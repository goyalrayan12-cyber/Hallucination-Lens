"""Run an MLX-quantized model natively on Apple Silicon and read its token log-probs."""

from __future__ import annotations

from hallucination_lens.backends.base import GenerationResult
from hallucination_lens.config import GenerationConfig


class LocalMLXBackend:
    name = "mlx"

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self._model = None
        self._tok = None
        self._eos_ids: set[int] = set()

    def _load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        print(f"Loading {self.cfg.model_id} on Apple MLX ... first run downloads it.")
        self._model, self._tok = load(self.cfg.model_id)

        eos = getattr(self._tok, "eos_token_ids", None)
        if eos:
            self._eos_ids = set(eos)
        elif getattr(self._tok, "eos_token_id", None) is not None:
            self._eos_ids = {int(self._tok.eos_token_id)}

    def _prompt(self, question: str) -> str:
        if not self.cfg.use_chat_template:
            return question
        messages = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": question})
        return self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, question: str) -> GenerationResult:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self._load()
        prompt = self._prompt(question)
        sampler = make_sampler(temp=0.0)  # greedy decoding (deterministic)

        chunks: list[str] = []
        token_logprobs: list[float] = []
        topk_logprobs: list[list[float]] = []
        finished = False

        for resp in stream_generate(
            self._model,
            self._tok,
            prompt=prompt,
            max_tokens=self.cfg.max_new_tokens,
            sampler=sampler,
        ):
            tok_id = int(resp.token)
            if tok_id in self._eos_ids:
                finished = True
                break

            lp = resp.logprobs
            chosen = round(float(lp[tok_id].item()), 6)
            k = max(1, min(self.cfg.top_k_logprobs, int(lp.shape[-1])))
            top_vals = mx.sort(mx.topk(lp, k))[::-1].tolist()

            chunks.append(resp.text)
            token_logprobs.append(chosen)
            topk_logprobs.append([round(float(v), 6) for v in top_vals])

            if resp.finish_reason == "stop":
                finished = True
                break

        return GenerationResult(
            text="".join(chunks).strip(),
            token_logprobs=token_logprobs,
            topk_logprobs=topk_logprobs,
            finished=finished,
            model=self.cfg.model_id,
        )