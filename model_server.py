"""
A model worker with transformers libs executes the model.

Run BF16 inference with:

python model_server.py --host localhost --model-path THUDM/glm-4-voice-9b --port 10000 --dtype bfloat16 --device cuda:0

Run Int4 inference with:

python model_server.py --host localhost --model-path THUDM/glm-4-voice-9b --port 10000 --dtype int4 --device cuda:0

"""
import argparse
import json
import base64
from typing import Any
import torch.nn as nn

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
import torch
import uvicorn

from threading import Thread
from queue import Queue


class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class ModelWorker:
    def __init__(self, model_path, dtype="bfloat16", device='cuda'):
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config if self.bnb_config else None,
            device_map={"": 0}
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if top_p >= 1.0:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True

        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        sampled_local = torch.multinomial(filtered, num_samples=1)
        sampled = torch.gather(sorted_indices, -1, sampled_local).squeeze(-1)
        return sampled

    @staticmethod
    def _get_transformer_layers(model: Any):
        # Common explicit paths first.
        explicit_paths = [
            ("transformer", "layers"),
            ("model", "layers"),
            ("encoder", "layers"),
            ("decoder", "layers"),
            ("transformer", "encoder", "layers"),
            ("model", "encoder", "layers"),
            ("transformer", "h"),
            ("model", "h"),
        ]

        def _get_path(root: Any, path: tuple[str, ...]):
            cur = root
            for p in path:
                cur = getattr(cur, p, None)
                if cur is None:
                    return None
            return cur

        best = None
        best_len = -1

        for path in explicit_paths:
            layers = _get_path(model, path)
            if layers is None or not hasattr(layers, "__len__"):
                continue
            n = int(len(layers))
            if n > best_len:
                best = layers
                best_len = n

        # Fallback heuristic: choose the largest ModuleList that looks like transformer blocks.
        for _name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            n = int(len(module))
            if n <= best_len:
                continue
            if n == 0:
                continue
            first = module[0]
            attrs = set(dir(first))
            looks_like_block = any(
                k in attrs
                for k in ["self_attn", "attention", "mlp", "feed_forward", "input_layernorm", "post_attention_layernorm"]
            )
            if looks_like_block or n >= 8:
                best = module
                best_len = n

        if best is not None:
            return best

        if hasattr(model, "layers") and hasattr(model.layers, "__len__"):
            return model.layers
        raise RuntimeError("Unable to locate transformer layers on this model")

    @staticmethod
    def _pack_generated_hidden_layers(hidden_states: Any, prompt_len: int) -> dict[str, Any]:
        if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) == 0:
            raise RuntimeError("Model did not return hidden_states")

        # HF hidden_states usually includes embedding output at index 0.
        layer_states = list(hidden_states[1:]) if len(hidden_states) > 1 else list(hidden_states)
        per_layer = [h[0, prompt_len:, :] for h in layer_states]
        if len(per_layer) == 0:
            hidden_layers = torch.empty((0, 0, 0), dtype=torch.float16)
        else:
            hidden_layers = torch.stack(per_layer, dim=1).detach().cpu().to(torch.float16)

        last_hidden = hidden_layers[:, -1, :] if hidden_layers.shape[1] > 0 else torch.empty((0, 0), dtype=torch.float16)
        hidden_layers_np = hidden_layers.numpy()
        last_hidden_np = last_hidden.numpy()

        return {
            "hidden_dtype": "float16",
            "hidden_layers_shape": list(hidden_layers_np.shape),
            "hidden_layers_b64": base64.b64encode(hidden_layers_np.tobytes()).decode("ascii"),
            # Backward-compatible single-layer fields.
            "hidden_shape": list(last_hidden_np.shape),
            "hidden_b64": base64.b64encode(last_hidden_np.tobytes()).decode("ascii"),
        }

    def _capture_all_layer_hidden(self, model: Any, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        layers = self._get_transformer_layers(model)
        captured: list[torch.Tensor | None] = [None] * len(layers)
        handles = []

        def _mk_hook(idx: int):
            def _hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                    captured[idx] = hidden
                return output

            return _hook

        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(_mk_hook(idx)))

        try:
            model(input_ids=input_ids, return_dict=True)
        finally:
            for h in handles:
                h.remove()

        per_layer: list[torch.Tensor] = []
        for hidden in captured:
            if hidden is None:
                continue
            per_layer.append(hidden[0, prompt_len:, :])

        if len(per_layer) == 0:
            return torch.empty((0, 0, 0), dtype=torch.float16)

        return torch.stack(per_layer, dim=1).detach().cpu().to(torch.float16)

    @staticmethod
    def _pack_hidden_layers_tensor(hidden_layers: torch.Tensor) -> dict[str, Any]:
        if hidden_layers.dim() != 3:
            raise RuntimeError(f"Expected hidden_layers rank-3 tensor, got rank {hidden_layers.dim()}")

        if hidden_layers.shape[1] > 0:
            last_hidden = hidden_layers[:, -1, :]
        else:
            last_hidden = torch.empty((int(hidden_layers.shape[0]), 0), dtype=hidden_layers.dtype)

        hidden_layers_np = hidden_layers.numpy()
        last_hidden_np = last_hidden.numpy()
        return {
            "hidden_dtype": "float16",
            "hidden_layers_shape": list(hidden_layers_np.shape),
            "hidden_layers_b64": base64.b64encode(hidden_layers_np.tobytes()).decode("ascii"),
            "hidden_shape": list(last_hidden_np.shape),
            "hidden_b64": base64.b64encode(last_hidden_np.tobytes()).decode("ascii"),
        }

    @staticmethod
    def _register_layer_capture_hooks(layers):
        latest: list[torch.Tensor | None] = [None] * len(layers)
        handles = []

        def _mk_hook(idx: int):
            def _hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                    latest[idx] = hidden
                return output

            return _hook

        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(_mk_hook(idx)))

        return latest, handles

    @staticmethod
    def _collect_step_layer_hidden(latest: list[torch.Tensor | None]) -> torch.Tensor | None:
        step_layers: list[torch.Tensor] = []
        for hidden in latest:
            if hidden is None or hidden.dim() != 3:
                return None
            step_layers.append(hidden[0, -1, :].detach())

        if not step_layers:
            return None
        return torch.stack(step_layers, dim=0)

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(
            target=model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                streamer=streamer
            )
        )
        thread.start()
        for token_id in streamer:
            yield (json.dumps({"token_id": token_id, "error_code": 0}) + "\n").encode()

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": "Server Error",
                "error_code": 1,
            }
            yield (json.dumps(ret) + "\n").encode()

    @torch.inference_mode()
    def generate_with_hidden(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))

        inputs = tokenizer([prompt], return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        layers = self._get_transformer_layers(model)
        latest_hidden, cap_handles = self._register_layer_capture_hooks(layers)

        generated: list[int] = []
        step_hidden: list[torch.Tensor] = []
        past_key_values = None
        stop_token_id = tokenizer.convert_tokens_to_ids("<|user|>")

        try:
            cur_input_ids = input_ids
            cur_attention_mask = attention_mask

            for _step in range(max_new_tokens):
                outputs = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                step_h = self._collect_step_layer_hidden(latest_hidden)
                if step_h is not None:
                    step_hidden.append(step_h)

                logits = outputs.logits[:, -1, :]
                next_token = self._sample_top_p(logits, temperature=temperature, top_p=top_p)
                token_id = int(next_token.item())
                generated.append(token_id)

                if token_id == stop_token_id:
                    break

                past_key_values = outputs.past_key_values
                cur_input_ids = next_token.unsqueeze(0)
                if cur_attention_mask is not None:
                    cur_attention_mask = torch.cat(
                        [
                            cur_attention_mask,
                            torch.ones((1, 1), device=cur_attention_mask.device, dtype=cur_attention_mask.dtype),
                        ],
                        dim=1,
                    )
        finally:
            for h in cap_handles:
                h.remove()

        if step_hidden:
            hidden_layers = torch.stack(step_hidden, dim=0).to(torch.float16).cpu()
        else:
            hidden_layers = torch.empty((0, int(len(layers)), int(getattr(model.config, "hidden_size", 0))), dtype=torch.float16)

        hidden_payload = self._pack_hidden_layers_tensor(hidden_layers)

        return {
            "token_ids": generated,
            **hidden_payload,
            "error_code": 0,
        }

    def generate_with_hidden_gate(self, params):
        try:
            return self.generate_with_hidden(params)
        except Exception as e:
            print("Caught Unknown Error", e)
            return {
                "token_ids": [],
                "hidden_dtype": "float16",
                "hidden_layers_shape": [0, 0, 0],
                "hidden_layers_b64": "",
                "hidden_shape": [0, 0],
                "hidden_b64": "",
                "error_code": 1,
                "text": "Server Error",
            }

    @torch.inference_mode()
    def generate_with_steering(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]
        temperature = float(params.get("temperature", 0.2))
        top_p = float(params.get("top_p", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        inject_layer = int(params.get("inject_layer", -1))
        steering_map_raw = params.get("steering_map", {})
        return_hidden = bool(params.get("return_hidden", False))

        if not isinstance(steering_map_raw, dict):
            raise ValueError("steering_map must be a dict of step->vector")

        inputs = tokenizer([prompt], return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        layers = self._get_transformer_layers(model)
        num_layers = len(layers)
        if inject_layer < 0:
            inject_layer = num_layers + inject_layer
        if inject_layer < 0 or inject_layer >= num_layers:
            raise ValueError(f"inject_layer {inject_layer} out of range [0, {num_layers - 1}]")

        hidden_size = int(getattr(model.config, "hidden_size", 0))
        steering_map: dict[int, torch.Tensor] = {}
        for step_k, vec in steering_map_raw.items():
            if vec is None:
                continue
            step = int(step_k)
            v = torch.tensor(vec, device=self.device, dtype=torch.float32)
            if hidden_size > 0 and int(v.numel()) != hidden_size:
                raise ValueError(
                    f"steering vector dim mismatch at step {step}: got {int(v.numel())}, expected {hidden_size}"
                )
            steering_map[step] = v

        step_state = {"step": 0}

        def _steer_hook(_module, _inputs, output):
            vec = steering_map.get(step_state["step"])
            if vec is None:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
                if hidden.dim() == 3:
                    steered = hidden.clone()
                    steered[:, -1, :] = steered[:, -1, :] + vec.to(steered.dtype)
                    return (steered, *output[1:])
                return output

            if isinstance(output, torch.Tensor) and output.dim() == 3:
                steered = output.clone()
                steered[:, -1, :] = steered[:, -1, :] + vec.to(steered.dtype)
                return steered
            return output

        handle = layers[inject_layer].register_forward_hook(_steer_hook)
        latest_hidden = None
        cap_handles = []
        if return_hidden:
            latest_hidden, cap_handles = self._register_layer_capture_hooks(layers)

        generated: list[int] = []
        step_hidden: list[torch.Tensor] = []
        past_key_values = None
        stop_token_id = tokenizer.convert_tokens_to_ids("<|user|>")

        try:
            cur_input_ids = input_ids
            cur_attention_mask = attention_mask

            for step in range(max_new_tokens):
                step_state["step"] = step
                outputs = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                if return_hidden and latest_hidden is not None:
                    step_h = self._collect_step_layer_hidden(latest_hidden)
                    if step_h is not None:
                        step_hidden.append(step_h)

                logits = outputs.logits[:, -1, :]
                next_token = self._sample_top_p(logits, temperature=temperature, top_p=top_p)
                token_id = int(next_token.item())
                generated.append(token_id)

                if token_id == stop_token_id:
                    break

                past_key_values = outputs.past_key_values
                cur_input_ids = next_token.unsqueeze(0)
                if cur_attention_mask is not None:
                    cur_attention_mask = torch.cat(
                        [cur_attention_mask, torch.ones((1, 1), device=cur_attention_mask.device, dtype=cur_attention_mask.dtype)],
                        dim=1,
                    )

        finally:
            handle.remove()
            for h in cap_handles:
                h.remove()

        payload = {
            "token_ids": generated,
            "error_code": 0,
        }

        if return_hidden:
            if step_hidden:
                hidden_layers = torch.stack(step_hidden, dim=0).to(torch.float16).cpu()
            else:
                hidden_layers = torch.empty((0, int(len(layers)), hidden_size if hidden_size > 0 else 0), dtype=torch.float16)
            payload.update(self._pack_hidden_layers_tensor(hidden_layers))

        return payload

    def generate_with_steering_gate(self, params):
        try:
            return self.generate_with_steering(params)
        except Exception as e:
            print("Caught Unknown Error", e)
            return {
                "token_ids": [],
                "error_code": 1,
                "text": "Server Error",
            }


app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()

    generator = worker.generate_stream_gate(params)
    return StreamingResponse(generator)


@app.post("/generate_with_hidden")
async def generate_with_hidden(request: Request):
    params = await request.json()
    payload = worker.generate_with_hidden_gate(params)
    return JSONResponse(payload)


@app.post("/generate_with_steering")
async def generate_with_steering(request: Request):
    params = await request.json()
    payload = worker.generate_with_steering_gate(params)
    return JSONResponse(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    args = parser.parse_args()

    worker = ModelWorker(args.model_path, args.dtype, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
