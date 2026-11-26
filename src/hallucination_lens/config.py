"""All settings live here as plain dataclasses, loaded from a YAML file in configs/.

Why: one place to change the model, the number of questions, alpha, etc. -- no hunting
through code. Every key has a default, so a YAML file only needs the keys you change.
You can also override from the command line:  --set data.n_questions=50
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from hallucination_lens.utils import slugify


@dataclass
class RunConfig:
    name: str | None = None  # label for this experiment; default is made from the model name
    data_dir: str = "data"  # cached generations + processed tables (git-ignored)
    runs_dir: str = "runs"  # results of each experiment (metrics, figures, model)


@dataclass
class DataConfig:
    dataset_id: str = "truthfulqa/truthful_qa"
    subset: str = "generation"
    split: str = "validation"
    n_questions: int | None = None  # None = all 817 questions


@dataclass
class GenerationConfig:
    backend: str = "local"  # "local" (Qwen on your Mac) or "gemini"
    max_new_tokens: int = 64
    top_k_logprobs: int = 5  # how many top candidates per token to keep (for entropy/margin)
    use_chat_template: bool = True  # the original report skipped this; instruct models need it
    system_prompt: str = "Answer the question truthfully in one or two short sentences."
    # --- local backend ---
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "auto"  # auto = bfloat16 on Apple GPU / CUDA, float32 on CPU
    device: str = "auto"  # auto = mps > cuda > cpu
    # --- gemini backend ---
    gemini_model: str = "gemini-2.5-flash"  # Gemini 3.x models do not return logprobs
    thinking_budget: int | None = 0  # 0 = switch "thinking" off (so all output tokens have logprobs)
    requests_per_minute: float = 10
    max_requests_per_run: int | None = None  # stop early to respect daily quota; re-run resumes
    max_retries: int = 5


@dataclass
class LabelConfig:
    method: str = "embedding"  # "embedding" (free, local) or "gemini_judge" (better, uses API)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    judge_model: str = "gemini-3.6-flash"
    judge_batch_size: int = 5  # answers judged per API call (saves quota)
    judge_requests_per_minute: float = 10
    judge_max_requests_per_run: int | None = None


@dataclass
class FeatureConfig:
    preset: str = "extended"  # "original4" (as in the report) or "extended"
    names: list[str] | None = None  # explicit list overrides the preset


@dataclass
class ModelConfig:
    name: str = "logreg"  # logreg | random_forest | gradient_boosting
    select_by_cv: bool = False  # True = pick the best of the three by cross-validated AUROC
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConformalConfig:
    alpha: float = 0.05  # allowed error rate; 0.05 = "95% guarantee"
    class_conditional: bool = True  # guarantee holds separately for each class (safer guardrail)
    repeated_splits: int = 200  # re-split the data this many times to test the guarantee; 0 = skip


@dataclass
class SplitConfig:
    train: float = 0.70
    calibration: float = 0.15
    test: float = 0.15
    seed: int = 42


_SECTIONS: dict[str, type] = {
    "run": RunConfig,
    "data": DataConfig,
    "generation": GenerationConfig,
    "label": LabelConfig,
    "features": FeatureConfig,
    "model": ModelConfig,
    "conformal": ConformalConfig,
    "split": SplitConfig,
}


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        data = data or {}
        unknown = set(data) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"Unknown config section(s): {sorted(unknown)}. Allowed: {sorted(_SECTIONS)}")
        kwargs = {}
        for name, section_cls in _SECTIONS.items():
            values = data.get(name) or {}
            allowed = {f.name for f in fields(section_cls)}
            bad = set(values) - allowed
            if bad:
                raise ValueError(f"Unknown key(s) in '{name}': {sorted(bad)}. Allowed: {sorted(allowed)}")
            kwargs[name] = section_cls(**values)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def validate(self) -> None:
        s = self.split
        if abs(s.train + s.calibration + s.test - 1.0) > 1e-6:
            raise ValueError("split.train + split.calibration + split.test must add up to 1.0")
        if not 0 < self.conformal.alpha < 1:
            raise ValueError("conformal.alpha must be between 0 and 1")
        if self.generation.backend not in ("local", "mlx", "gemini"):
            raise ValueError("generation.backend must be 'local', 'mlx', or 'gemini'")
        if self.label.method not in ("embedding", "gemini_judge"):
            raise ValueError("label.method must be 'embedding' or 'gemini_judge'")
        if self.model.name not in ("logreg", "random_forest", "gradient_boosting"):
            raise ValueError("model.name must be logreg, random_forest or gradient_boosting")


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML file (or defaults if path is None), then apply `section.key=value` overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Bad --set '{item}'. Use section.key=value, e.g. data.n_questions=50")
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        raw.setdefault(section, {})
        raw[section][name] = yaml.safe_load(value)
    return Config.from_dict(raw)


# ----- where files go -------------------------------------------------------------------------


def run_tag(cfg: Config) -> str:
    """Name used for this experiment's files. Set run.name to keep experiments separate."""
    if cfg.run.name:
        return cfg.run.name
    g = cfg.generation
    model = g.model_id if g.backend in ("local", "mlx") else g.gemini_model
    return f"{g.backend}_{slugify(model)}"


def generations_path(cfg: Config) -> Path:
    return Path(cfg.run.data_dir) / "generations" / f"{run_tag(cfg)}.jsonl"


def judgements_path(cfg: Config) -> Path:
    return Path(cfg.run.data_dir) / "judgements" / f"{run_tag(cfg)}.jsonl"


def dataset_path(cfg: Config) -> Path:
    return Path(cfg.run.data_dir) / "processed" / f"{run_tag(cfg)}.csv"


def run_dir(cfg: Config) -> Path:
    return Path(cfg.run.runs_dir) / run_tag(cfg)

