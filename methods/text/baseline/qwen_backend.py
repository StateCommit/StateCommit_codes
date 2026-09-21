"""Greedy local Qwen backend shared by all text-agent methods."""

from __future__ import annotations

import time
from pathlib import Path

from .base import Generation


class QwenBackend:
    name = "qwen2.5-coder-instruct-greedy"

    def __init__(self, *, model_path: str | Path, device: str = "cuda:0") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("QwenBackend requires torch and transformers") from exc
        self._torch = torch
        self.device = device
        self.model_path = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()

    def generate(self, *, role: str, system: str, user: str, max_new_tokens: int) -> Generation:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]




        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        started = time.monotonic()
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        seconds = time.monotonic() - started
        completion = generated[0, encoded.input_ids.shape[1] :]
        raw = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        return Generation(
            role=role,
            raw=raw,
            seconds=seconds,
            input_tokens=int(encoded.input_ids.shape[1]),
            output_tokens=int(completion.shape[0]),
        )
