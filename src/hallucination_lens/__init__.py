"""Hallucination Lens.

Estimate how risky an LLM answer is using only the token log-probabilities the model
already produces, then wrap the prediction in conformal prediction so that the output
is a *set of labels* with a statistical error guarantee.

Package map (read in this order if you are new):
    config.py       settings, loaded from YAML files in configs/
    data.py         downloads the TruthfulQA questions
    backends/       "who answers the question?"  local Qwen model  or  Gemini API
    features.py     turns token log-probabilities into a few numbers per answer
    labeling.py     decides if an answer is factual (0) or hallucinated (1)
    modeling.py     the classifier + train/calibration/test split
    conformal.py    split conformal prediction written from scratch (LAC score)
    evaluation.py   metrics, repeated-split experiment, alpha sweep
    plots.py        all figures
    guardrail.py    answer a new question and return TRUST / UNCERTAIN / DON'T TRUST
    pipeline.py     the stages: generate -> build dataset -> train/evaluate
    cli.py          the `hallucination-lens` command
"""

__version__ = "0.2.0"
