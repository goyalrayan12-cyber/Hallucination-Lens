"""The three stages of the project, as plain functions (the CLI just calls these).

    run_generate()        question -> model answer + token log-probabilities   (slow, cached)
    build_dataset()       answers  -> features + labels in one table           (fast)
    train_and_evaluate()  table    -> classifier + conformal guardrail + figures + report

Each stage saves its output to disk, so you can re-run a later stage without redoing the slow
one. Generation is resumable: kill it any time and run again to continue where it stopped.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from tqdm import tqdm

from hallucination_lens import __version__, plots
from hallucination_lens.backends import make_backend
from hallucination_lens.config import (
    Config,
    dataset_path,
    generations_path,
    judgements_path,
    run_dir,
    run_tag,
)
from hallucination_lens.conformal import ConformalPredictor
from hallucination_lens.data import load_truthfulqa, reference_answers
from hallucination_lens.errors import (
    DailyQuotaExceeded,
    HallucinationLensError,
    LogprobsUnavailableError,
)
from hallucination_lens.evaluation import (
    alpha_sweep,
    classifier_metrics,
    conformal_metrics,
    repeated_split_experiment,
    set_kinds,
    summarize_repeats,
)
from hallucination_lens.features import compute_features, resolve_feature_names
from hallucination_lens.labeling import EmbeddingLabeler, GeminiJudge, verdict_to_label
from hallucination_lens.modeling import build_model, compare_models, three_way_split
from hallucination_lens.report import format_summary, write_report
from hallucination_lens.utils import to_jsonable

# ---------------------------------------------------------------------------------------------
# Stage 1: generate
# ---------------------------------------------------------------------------------------------


def read_generations(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_generate(cfg: Config, backend=None, questions: pd.DataFrame | None = None) -> Path:
    path = generations_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    questions = questions if questions is not None else load_truthfulqa(cfg.data)
    done = {r["question_id"] for r in read_generations(path)}
    todo = questions[~questions["question_id"].isin(done)]
    if todo.empty:
        print(f"All {len(questions)} answers already generated -> {path}")
        return path

    backend = backend or make_backend(cfg.generation)
    budget = cfg.generation.max_requests_per_run if cfg.generation.backend == "gemini" else None
    made = 0
    with open(path, "a") as f:
        for row in tqdm(todo.itertuples(index=False), total=len(todo), desc="Generating answers"):
            if budget is not None and made >= budget:
                print(f"Reached max_requests_per_run={budget}. Run the same command again to continue.")
                break
            try:
                res = backend.generate(row.question)
            except DailyQuotaExceeded as exc:
                print(exc)
                break
            except LogprobsUnavailableError:
                raise  # applies to every question: stop and tell the user
            except HallucinationLensError as exc:
                tqdm.write(f"Skipping question {row.question_id}: {exc}")
                continue
            made += 1
            record = {
                "question_id": int(row.question_id),
                "question": row.question,
                "backend": backend.name,
                "model": res.model,
                "text": res.text,
                "token_logprobs": res.token_logprobs,
                "topk_logprobs": res.topk_logprobs,
                "finished": res.finished,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
    total = len(read_generations(path))
    print(f"{total}/{len(questions)} answers saved -> {path}")
    return path


# ---------------------------------------------------------------------------------------------
# Stage 2: features + labels
# ---------------------------------------------------------------------------------------------


def build_dataset(
    cfg: Config, questions: pd.DataFrame | None = None, labeler=None, judge=None
) -> pd.DataFrame:
    gens = read_generations(generations_path(cfg))
    if not gens:
        raise HallucinationLensError("No generations found. Run `hallucination-lens generate` first.")
    questions = questions if questions is not None else load_truthfulqa(cfg.data)
    qidx = questions.set_index("question_id")
    labeler = labeler or EmbeddingLabeler(cfg.label.embedding_model)

    rows, items, skipped = [], [], 0
    for g in tqdm(gens, desc="Features + labels"):
        qid = g["question_id"]
        if qid not in qidx.index:
            continue
        feats = compute_features(g["token_logprobs"], g.get("topk_logprobs"), g.get("finished", True))
        if feats is None:
            skipped += 1
            continue
        ref = qidx.loc[qid]
        correct, incorrect = reference_answers(ref)
        rows.append(
            {
                "question_id": qid,
                "question": g["question"],
                "category": ref.get("category", ""),
                "generated_text": g["text"],
                **feats,
                **labeler.label(g["text"], correct, incorrect),
            }
        )
        items.append(
            {"question_id": qid, "question": g["question"], "answer": g["text"],
             "correct": correct, "incorrect": incorrect}
        )
    df = pd.DataFrame(rows)
    df["label_judge"] = np.nan
    df["label_source"] = "embedding"
    df["is_hallucination"] = df["label_embedding"]

    audit: dict = {"method": cfg.label.method, "n_answers": len(df), "n_empty_skipped": skipped}
    if cfg.label.method == "gemini_judge":
        judge = judge or GeminiJudge(cfg.label, judgements_path(cfg))
        verdicts = judge.judge(items)
        df["label_judge"] = pd.to_numeric(
            df["question_id"].map(lambda q: verdict_to_label(verdicts.get(q, "unsure"))), errors="coerce"
        )
        has = df["label_judge"].notna()
        df.loc[has, "is_hallucination"] = df.loc[has, "label_judge"].astype(int)
        df.loc[has, "label_source"] = "gemini_judge"
        audit["n_judged"] = int(has.sum())
        audit["n_unsure_or_missing_fell_back_to_embedding"] = int((~has).sum())
        if has.sum() > 1 and df.loc[has, "label_judge"].nunique() > 1:
            both = df.loc[has]
            audit["agreement_with_embedding"] = float((both["label_judge"] == both["label_embedding"]).mean())
            audit["cohen_kappa_vs_embedding"] = float(
                cohen_kappa_score(both["label_judge"].astype(int), both["label_embedding"].astype(int))
            )
    df["is_hallucination"] = df["is_hallucination"].astype(int)

    out = dataset_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".label_audit.json").write_text(json.dumps(to_jsonable(audit), indent=2))
    print(f"{len(df)} rows saved -> {out}  (hallucinated share {df['is_hallucination'].mean():.1%})")
    if "agreement_with_embedding" in audit:
        print(
            f"Judge vs embedding labels: agreement {audit['agreement_with_embedding']:.1%}, "
            f"Cohen's kappa {audit['cohen_kappa_vs_embedding']:.2f}"
        )
    return df


# ---------------------------------------------------------------------------------------------
# Stage 3: train, calibrate, evaluate
# ---------------------------------------------------------------------------------------------


def train_and_evaluate(cfg: Config, df: pd.DataFrame | None = None, make_figures: bool = True) -> dict:
    if df is None:
        path = dataset_path(cfg)
        if not path.exists():
            raise HallucinationLensError(f"{path} not found. Run the `label` stage first.")
        df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    names = resolve_feature_names(cfg.features.preset, cfg.features.names)
    X = df[names]
    y = df["is_hallucination"].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 5:
        raise HallucinationLensError(
            "Need at least 5 answers of each class to train and calibrate. "
            f"Label counts: {y.value_counts().to_dict()}"
        )
    s = cfg.split
    tr, ca, te = three_way_split(y, s.train, s.calibration, s.test, s.seed)

    # --- classifier (optionally choose by cross-validation on the TRAIN part only) ---
    model_name, params, cv_table = cfg.model.name, cfg.model.params, None
    if cfg.model.select_by_cv:
        cv_table = compare_models(X.iloc[tr], y.iloc[tr], seed=s.seed)
        model_name = str(cv_table.iloc[0]["model"])
        params = cfg.model.params if model_name == cfg.model.name else None
    model = build_model(model_name, params, seed=s.seed).fit(X.iloc[tr], y.iloc[tr])
    proba = model.predict_proba(X)
    p_hall = proba[:, 1]

    # --- conformal calibration on the calibration split only ---
    alpha = cfg.conformal.alpha
    cp_marg = ConformalPredictor(alpha, False).calibrate(proba[ca], y.iloc[ca])
    cp_cc = ConformalPredictor(alpha, True).calibrate(proba[ca], y.iloc[ca])
    cp = cp_cc if cfg.conformal.class_conditional else cp_marg
    variant = "class_conditional" if cfg.conformal.class_conditional else "marginal"

    y_te = y.iloc[te].to_numpy()
    sets_marg, sets_cc = cp_marg.predict_set(proba[te]), cp_cc.predict_set(proba[te])
    sets_te = sets_cc if cfg.conformal.class_conditional else sets_marg

    clf_metrics = classifier_metrics(y_te, p_hall[te])
    try:  # baseline: how well does the raw average confidence alone rank hallucinations?
        clf_metrics["auroc_mean_logprob_only"] = float(roc_auc_score(y_te, -df["mean_logprob"].iloc[te]))
    except ValueError:
        clf_metrics["auroc_mean_logprob_only"] = float("nan")

    # --- extra evidence: repeated splits + alpha sweep ---
    repeats = None
    if cfg.conformal.repeated_splits > 0:
        print(f"Re-splitting {cfg.conformal.repeated_splits} times to test the guarantee ...")
        repeats = repeated_split_experiment(
            X, y, model_name, params, alpha, cfg.conformal.repeated_splits,
            (s.train, s.calibration, s.test), s.seed,
        )
    sweep = alpha_sweep(
        proba[ca], y.iloc[ca], proba[te], y_te, class_conditional=cfg.conformal.class_conditional
    )

    g = cfg.generation
    metrics = {
        "tag": run_tag(cfg),
        "created": datetime.now().isoformat(timespec="seconds"),
        "version": __version__,
        "backend": g.backend,
        "model": g.model_id if g.backend in ("local", "mlx") else g.gemini_model,
        "label_method": cfg.label.method,
        "label_sources": df["label_source"].value_counts().to_dict() if "label_source" in df else {},
        "classifier_name": model_name,
        "feature_names": names,
        "n": {"total": len(df), "train": len(tr), "calibration": len(ca), "test": len(te)},
        "hallucinated_share": float(y.mean()),
        "classifier": clf_metrics,
        "conformal": {
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "guardrail_variant": variant,
            "marginal": conformal_metrics(y_te, sets_marg),
            "class_conditional": conformal_metrics(y_te, sets_cc),
            "q_hat": cp.q_hat,
            "q_hat_by_class": cp.q_hat_by_class,
        },
        "repeated_splits": summarize_repeats(repeats) if repeats is not None else None,
        "cv_model_selection": cv_table.to_dict("records") if cv_table is not None else None,
    }

    # --- save everything ---
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    cfg.save(rd / "config.yaml")
    joblib.dump(model, rd / "model.joblib")
    (rd / "conformal.json").write_text(json.dumps(to_jsonable(cp.to_dict()), indent=2))
    (rd / "meta.json").write_text(json.dumps({"feature_names": names, "tag": run_tag(cfg)}, indent=2))
    (rd / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2))
    sweep.to_csv(rd / "alpha_sweep.csv", index=False)
    if repeats is not None:
        repeats.to_csv(rd / "repeated_splits.csv", index=False)

    split_name = np.empty(len(df), dtype=object)
    split_name[tr], split_name[ca], split_name[te] = "train", "calibration", "test"
    all_sets = cp.predict_set(proba)
    pd.DataFrame(
        {
            "question_id": df["question_id"],
            "split": split_name,
            "question": df["question"],
            "generated_text": df["generated_text"],
            "is_hallucination": y,
            "p_hallucination": p_hall,
            "set_kind": set_kinds(all_sets),
        }
    ).to_csv(rd / "predictions.csv", index=False)

    figures: list[str] = []
    if make_figures:
        fig_dir = rd / "figures"
        kinds_te = set_kinds(sets_te)

        def save(name, fig):
            plots.save_fig(fig, fig_dir / name)
            figures.append(name)

        save("class_balance.png", plots.plot_class_balance(y))
        save("feature_distributions.png", plots.plot_feature_distributions(df, names))
        save("reliability.png", plots.plot_reliability(y_te, p_hall[te]))
        save("verdicts.png", plots.plot_verdicts(y_te, kinds_te))
        save("confidence_vs_verdict.png",
             plots.plot_confidence_vs_verdict(df["mean_logprob"].iloc[te], y_te, kinds_te))
        save("alpha_sweep.png", plots.plot_alpha_sweep(sweep))
        if repeats is not None:
            save("coverage_distribution.png", plots.plot_coverage_distribution(repeats, alpha))
        if model_name == "logreg":
            save("coefficients.png", plots.plot_coefficients(names, model.named_steps["clf"].coef_[0]))
    write_report(rd, metrics, figures)
    print("\n" + format_summary(metrics))
    print(f"\nEverything saved in {rd}/  (open REPORT.md)")
    return metrics
