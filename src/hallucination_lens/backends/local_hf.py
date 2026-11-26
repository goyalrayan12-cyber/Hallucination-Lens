"""Run an open-weight model (e.g. Qwen2.5) on your own machine and read its token probabilities.

On an Apple-silicon Mac the model runs on the GPU through PyTorch's "mps" device. The model is
loaded lazily, so importing this file does not need PyTorch (handy for tests and CI).

Upgrades over the original notebook:
  * uses the model's chat template (instruct models are trained on it)
  * greedy decoding with fixed settings -> the same question always gives the same answer
  * stores the top-k candidates of every step (for entropy / margin features)
  * drops the final end-of-sequence token, so "length" means real answer tokens
"""

from __future__ import annotations

import numpy as np

from hallucination_lens.backends.base import GenerationResult
from hallucination_lens.config import GenerationConfig


def extract_token_stats(
    logits: np.ndarray, token_ids: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """From raw scores to log-probabilities.

    logits    : (T, V)  one row of scores per generated step, V = vocabulary size
    token_ids : (T,)    the token that was actually chosen at each step
    returns   : chosen log-prob per step (T,), and the top-k log-probs per step (T, k) sorted
                from most to least likely.
    """
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    log_norm = m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))  # log-sum-exp
    logp = x - log_norm  # log-softmax
    chosen = logp[np.arange(len(token_ids)), np.asarray(token_ids)]
    k = max(1, min(k, logp.shape[1]))
    idx = np.argpartition(logp, -k, axis=1)[:, -k:]
    top = np.take_along_axis(logp, idx, axis=1)
    order = np.argsort(-top, axis=1)
    return chosen, np.take_along_axis(top, order, axis=1)


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalHFBackend:
    name = "local"

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self._model = None
        self._tok = None
        self._device = "cpu"
        self._eos_ids: list[int] = []
        self._calls = 0

    # ---- loading --------------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = pick_device(self.cfg.device)
        if self.cfg.dtype == "auto":
            dtype = torch.float32 if self._device == "cpu" else torch.bfloat16
        else:
            dtype = getattr(torch, self.cfg.dtype)
        print(f"Loading {self.cfg.model_id} on '{self._device}' ({dtype}) ... first run downloads it.")
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, dtype=dtype)
        self._model = model.to(self._device).eval()

        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if self._tok.eos_token_id is not None and self._tok.eos_token_id not in eos:
            eos.append(self._tok.eos_token_id)
        self._eos_ids = eos

    def _prompt(self, question: str) -> str:
        if not self.cfg.use_chat_template:
            return question
        messages = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": question})
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---- generating -----------------------------------------------------------------------
    def generate(self, question: str) -> GenerationResult:
        import torch
        from transformers import GenerationConfig as HFGenerationConfig

        self._load()
        prompt = self._prompt(question)
        # the chat template already contains the special tokens, so do not add them twice
        inputs = self._tok(
            prompt, return_tensors="pt", add_special_tokens=not self.cfg.use_chat_template
        ).to(self._device)
        input_len = inputs["input_ids"].shape[1]

        gen_cfg = HFGenerationConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,  # greedy: deterministic and reproducible
            repetition_penalty=1.0,  # keep scores unmodified so log-probs are the model's own
            eos_token_id=self._eos_ids,
            pad_token_id=self._tok.pad_token_id
            if self._tok.pad_token_id is not None
            else self._eos_ids[0],
        )
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, generation_config=gen_cfg, return_dict_in_generate=True, output_scores=True
            )
        ids = out.sequences[0, input_len:].tolist()
        scores = torch.stack(out.scores, dim=0)[:, 0, :].float().cpu().numpy()  # (T, V)

        finished = bool(ids) and ids[-1] in self._eos_ids
        if finished:  # do not count the end-of-answer marker as an answer token
            ids, scores = ids[:-1], scores[:-1]
        if not ids:
            return GenerationResult("", [], [], finished, self.cfg.model_id)

        chosen, topk = extract_token_stats(scores, np.asarray(ids), self.cfg.top_k_logprobs)
        text = self._tok.decode(ids, skip_special_tokens=True).strip()

        self._calls += 1
        if self._device == "mps" and self._calls % 25 == 0:
            torch.mps.empty_cache()  # keep Apple unified memory tidy on long runs
        return GenerationResult(
            text=text,
            token_logprobs=[round(float(v), 6) for v in chosen],
            topk_logprobs=[[round(float(v), 6) for v in row] for row in topk],
            finished=finished,
            model=self.cfg.model_id,
        )
