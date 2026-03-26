# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


def _build_quantization_config():
    """Build BitsAndBytes 4-bit quantization config if KIMODO_QUANTIZE=4bit is set."""
    quantize = os.environ.get("KIMODO_QUANTIZE", "").lower()
    if quantize != "4bit":
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            peft_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)

        extra_kwargs = {}
        quantization_config = _build_quantization_config()
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config
            extra_kwargs["device_map"] = "auto"
            print("[Kimodo] Using 4-bit quantization (NF4) for text encoder to reduce VRAM usage")

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **extra_kwargs,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._quantized = quantization_config is not None

    def to(self, device: torch.device):
        if not self._quantized:
            self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            encoded_text = self.model.encode(text, batch_size=len(text), show_progress_bar=False)

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths


class DummyTextEncoder:
    """Zero-vector text encoder for constraint-only generation without LLM weights.

    Activated by setting TEXT_ENCODER_MODE=dummy. Returns zero embeddings
    of the correct shape (llm_dim=4096), which the model treats as
    unconditional (same as empty-text in classifier-free guidance).

    When Llama access is available, remove TEXT_ENCODER_MODE=dummy to
    use the real LLM2Vec encoder with full text prompt support.
    """

    def __init__(self, llm_dim: int = 4096, device: str = "cuda:0") -> None:
        self.llm_dim = llm_dim
        self._device = torch.device(device)
        print("[Kimodo] Using DummyTextEncoder (zero embeddings, dim={})".format(llm_dim))
        print("[Kimodo] Text prompts will be IGNORED. Use constraints for motion control.")
        print("[Kimodo] To enable text prompts, remove TEXT_ENCODER_MODE=dummy after Llama access is granted.")

    def to(self, device: torch.device):
        self._device = torch.device(device)
        return self

    def eval(self):
        return self

    def get_device(self):
        return self._device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        encoded_text = torch.zeros(len(text), 1, self.llm_dim, device=self._device)
        lengths = np.ones(len(text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        return encoded_text, lengths
