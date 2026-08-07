# -*- coding: utf-8 -*-
"""
Llm.Qwen_Client — local Qwen2.5-Instruct client for netlist QA.

Native transformers, reuses torch already installed and GPU-configured
in pytorch251 — same reasoning as the TrOCR attempt, but this time for
a TEXT model (no image preprocessing needed at all, the input is
already a short netlist string).

Starts with Qwen2.5-1.5B-Instruct rather than the 7B used for the VLM
track: this task (looking up a fact in a ~15-line netlist) does not
obviously need a large model, and a fast first pass is more useful for
the fine-tune-or-not decision than committing to a slow one.

IMPORTANT — not executed end to end in the environment used to write
this file (torch unavailable in this sandbox, same constraint as
throughout the OCR work). The chat-template + generate() pattern
follows the current, stable Qwen2.5 documentation, verified via a
fresh fetch. Validate on a handful of questions before running the
full benchmark.
"""
from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class QwenClient:
    """Lazy-loaded local Qwen2.5-Instruct model for short, grounded
    answers over a netlist."""

    def __init__(self, model_name: str = DEFAULT_MODEL, gpu: bool = True,
                max_new_tokens: int = 32):
        self._model_name = model_name
        self._gpu = gpu
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if (self._gpu and torch.cuda.is_available()) else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype="auto").to(self._device)
        self._model.eval()

    def ask(self, messages: list[dict]) -> str:
        """Sends chat-format messages, returns the raw generated text
        (grading/normalization happens separately, in Llm.Prompting)."""
        self._ensure_model()
        import torch

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self._tokenizer(
            [text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs, max_new_tokens=self._max_new_tokens)
        generated_ids = [
            out[len(inp):] for inp, out in
            zip(model_inputs.input_ids, generated_ids)
        ]
        response = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response.strip()
