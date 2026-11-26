"""The `hallucination-lens` command. Run `hallucination-lens --help` to see everything.

Typical session:
    hallucination-lens smoke-test                       # does my install work? (no model/key)
    hallucination-lens run-all --config configs/default.yaml
    hallucination-lens ask --run runs/local_qwen2.5-3b-instruct "If you smash a mirror, what happens?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hallucination_lens import __version__
from hallucination_lens.config import Config, dataset_path, load_config
from hallucination_lens.errors import HallucinationLensError, LogprobsUnavailableError


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="YAML file from configs/ (default: built-in defaults)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override one setting, e.g. --set data.n_questions=50 (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hallucination-lens", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_ in [
        ("generate", "stage 1: let the model answer the questions and record token log-probabilities"),
        ("label", "stage 2: compute features + factual/hallucinated labels"),
        ("train", "stage 3: train classifier, calibrate conformal guardrail, evaluate, make figures"),
        ("run-all", "run stages 1-3 in a row"),
    ]:
        _common(sub.add_parser(name, help=help_))

    ask = sub.add_parser("ask", help="answer a question and say whether to trust the answer")
    ask.add_argument("question", nargs="?", help="omit to start an interactive prompt")
    ask.add_argument("--run", required=True, help="a finished run folder, e.g. runs/local_qwen2.5-3b-instruct")

    _common(sub.add_parser("gemini-check", help="test your Gemini key and whether the model returns logprobs"))
    sub.add_parser("compare", help="table of all finished runs").add_argument("--runs-dir", default="runs")
    sub.add_parser("smoke-test", help="10-second install check on synthetic data (no model, no key)")
    return parser


# ---------------------------------------------------------------------------------------------


def cmd_smoke_test() -> None:
    from hallucination_lens.pipeline import train_and_evaluate
    from hallucination_lens.synthetic import make_synthetic_dataset

    cfg = load_config(None, ["run.name=smoke_test", "conformal.repeated_splits=30"])
    df = make_synthetic_dataset(500)
    path = dataset_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Running the statistics half of the pipeline on SYNTHETIC data (numbers are meaningless).")
    train_and_evaluate(cfg, df=df)
    print("\nSmoke test passed: classifier, conformal calibration, figures and report all work.")


def cmd_gemini_check(cfg: Config) -> None:
    from hallucination_lens.backends.gemini import GeminiBackend, make_client

    client = make_client()
    print("[OK] Found a Gemini API key.")
    model = cfg.generation.gemini_model
    try:
        res = GeminiBackend(cfg.generation, client=client).generate("What is the capital of France?")
        extra = "top-k alternatives included" if res.topk_logprobs else "no top-k alternatives"
        print(f"[OK] {model} returned {len(res.token_logprobs)} token log-probs ({extra}). Answer: {res.text!r}")
    except LogprobsUnavailableError as exc:
        print(f"[FAIL] {model}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {model}: request failed -> {exc}")
    judge = cfg.label.judge_model
    try:
        reply = client.models.generate_content(model=judge, contents="Reply with the single word OK.")
        print(f"[OK] judge model {judge} responded: {(reply.text or '').strip()[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] judge model {judge}: {exc}\n       Change label.judge_model in your config to a model you can access.")


def cmd_compare(runs_dir: str) -> None:
    import pandas as pd

    rows = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = json.loads(path.read_text())
        if m["tag"] == "smoke_test":
            continue
        k = m["conformal"]
        g = k[k["guardrail_variant"]]
        rep = (m.get("repeated_splits") or {}).get(k["guardrail_variant"], {})
        rows.append({
            "run": m["tag"], "model": m["model"], "labels": m["label_method"], "n": m["n"]["total"],
            "n_test": m["n"]["test"], "acc": m["classifier"]["accuracy"],
            "majority": m["classifier"]["majority_baseline_accuracy"], "auroc": m["classifier"]["auroc"],
            "ece": m["classifier"]["ece"], "coverage": g["coverage"], "set_size": g["avg_set_size"],
            "trust_rate": g["rate_trust"], "wrongly_trust": g["false_reassurance_rate"],
            "resplit_cov": rep.get("coverage_mean"),
        })
    if not rows:
        print(f"No finished runs found under {runs_dir}/")
        return
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    out = Path(runs_dir) / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


def cmd_ask(run: str, question: str | None) -> None:
    from hallucination_lens.guardrail import Guardrail

    guard = Guardrail.from_run(run)
    if question:
        print(guard.check(question).format())
        return
    print("Type a question (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        print(guard.check(q).format())


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()  # reads .env (GEMINI_API_KEY etc.) before any heavy library is imported
    import matplotlib

    matplotlib.use("Agg")  # figures are saved to files, never popped up
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke-test":
            cmd_smoke_test()
        elif args.command == "compare":
            cmd_compare(args.runs_dir)
        elif args.command == "ask":
            cmd_ask(args.run, args.question)
        else:
            from hallucination_lens import pipeline

            cfg = load_config(args.config, args.overrides)
            if args.command == "gemini-check":
                cmd_gemini_check(cfg)
            elif args.command == "generate":
                pipeline.run_generate(cfg)
            elif args.command == "label":
                pipeline.build_dataset(cfg)
            elif args.command == "train":
                pipeline.train_and_evaluate(cfg)
            elif args.command == "run-all":
                pipeline.run_generate(cfg)
                pipeline.build_dataset(cfg)
                pipeline.train_and_evaluate(cfg)
    except (HallucinationLensError, ValueError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
