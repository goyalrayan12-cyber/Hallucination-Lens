from pathlib import Path

import pytest

from hallucination_lens.config import load_config, run_tag


def test_overrides_and_tags():
    cfg = load_config(None, ["data.n_questions=50", "generation.backend=gemini"])
    assert cfg.data.n_questions == 50
    assert run_tag(cfg) == "gemini_gemini-2.5-flash"
    assert run_tag(load_config(None)) == "local_qwen2.5-3b-instruct"


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(None, ["data.n_question=5"])
    with pytest.raises(ValueError, match="Bad --set"):
        load_config(None, ["nonsense"])
    with pytest.raises(ValueError, match="add up to 1.0"):
        load_config(None, ["split.train=0.9"])
    with pytest.raises(ValueError, match="backend"):
        load_config(None, ["generation.backend=openai"])


def test_every_shipped_config_loads():
    files = sorted(Path(__file__).parent.parent.glob("configs/*.yaml"))
    assert len(files) >= 4
    for f in files:
        load_config(f)
