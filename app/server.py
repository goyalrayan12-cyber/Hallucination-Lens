"""Hallucination Lens Studio — FastAPI backend, SQLite history store, and M5-safe model runner."""

from __future__ import annotations

import csv
import inspect
import json
import math
import re
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "runs"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "ui_history.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

SUPPORTED_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25)


# =============================================================================
# 1. SQLite Persistent History Layer (WAL Mode)
# =============================================================================
class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,              -- 'single' or 'compare'
                    question TEXT NOT NULL,
                    primary_model TEXT NOT NULL,
                    secondary_model TEXT,
                    verdict_summary TEXT NOT NULL,   -- 'TRUST', 'UNCERTAIN', 'DONT_TRUST', 'DIVERGENT'
                    total_latency_ms REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at DESC)"
            )

    def add(
        self,
        *,
        mode: str,
        question: str,
        primary_model: str,
        secondary_model: str | None,
        verdict_summary: str,
        total_latency_ms: float,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO history (
                    created_at, mode, question, primary_model, secondary_model,
                    verdict_summary, total_latency_ms, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    mode,
                    question,
                    primary_model,
                    secondary_model,
                    verdict_summary,
                    round(total_latency_ms, 2),
                    json.dumps(payload),
                ),
            )
            row_id = cur.lastrowid
        return {
            "id": row_id,
            "created_at": created_at,
            "mode": mode,
            "question": question,
            "primary_model": primary_model,
            "secondary_model": secondary_model,
            "verdict_summary": verdict_summary,
            "total_latency_ms": round(total_latency_ms, 2),
            "payload": payload,
        }

    def list(
        self,
        q: str = "",
        mode: str = "all",
        verdict: str = "all",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if q.strip():
            clauses.append("(question LIKE ? OR payload_json LIKE ?)")
            like = f"%{q.strip()}%"
            params.extend([like, like])
        if mode in ("single", "compare"):
            clauses.append("mode = ?")
            params.append(mode)
        if verdict != "all":
            clauses.append("verdict_summary = ?")
            params.append(verdict)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM history {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "mode": r["mode"],
                    "question": r["question"],
                    "primary_model": r["primary_model"],
                    "secondary_model": r["secondary_model"],
                    "verdict_summary": r["verdict_summary"],
                    "total_latency_ms": r["total_latency_ms"],
                    "payload": json.loads(r["payload_json"]),
                }
            )
        return results

    def delete(self, item_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM history")
            return cur.rowcount


# =============================================================================
# 2. Run Discovery, Conformal Quantiles & M5 GPU-Serialized Guardrail Engine
# =============================================================================
def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".json":
        try:
            return json.loads(text)
        except Exception:
            return {}
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        return {}


def _compute_conformal_quantiles(run_dir: Path) -> dict[str, float]:
    """Compute exact finite-sample split conformal quantiles from predictions.csv."""
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        return {}

    scores: list[float] = []
    with pred_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("split", "")).strip().lower() != "calibration":
                continue
            try:
                y = int(float(row["is_hallucination"]))
                p1 = float(row["p_hallucination"])
            except (KeyError, TypeError, ValueError):
                continue
            p_true = p1 if y == 1 else (1.0 - p1)
            scores.append(1.0 - p_true)

    if not scores:
        return {}

    scores.sort()
    n = len(scores)
    quantiles: dict[str, float] = {}
    for alpha in SUPPORTED_ALPHAS:
        idx = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha)))) - 1
        quantiles[f"{alpha:.2f}"] = round(scores[idx], 4)
    return quantiles


def _apply_alpha_to_result(
    res: dict[str, Any], alpha: float, quantiles: dict[str, float]
) -> dict[str, Any]:
    """Recalculate the conformal prediction set and verdict for a requested alpha."""
    key = f"{alpha:.2f}"
    q_hat = quantiles.get(key)
    prob = res.get("prob_hallucination")
    res["alpha"] = round(alpha, 2)
    res["q_hat"] = q_hat

    if q_hat is None or prob is None:
        return res

    pred_set: list[int] = []
    if prob <= q_hat:          # score(Factual) = 1 - (1 - p1) = p1 <= q_hat
        pred_set.append(0)
    if (1.0 - prob) <= q_hat:  # score(Hallucinated) = 1 - p1 <= q_hat
        pred_set.append(1)

    verdict = _normalize_verdict("", pred_set)
    res["prediction_set"] = pred_set
    res["set_size"] = len(pred_set)
    res["verdict"] = verdict
    res["raw_verdict"] = verdict
    res["is_hallucination_flagged"] = verdict != "TRUST"
    return res


def discover_runs() -> list[dict[str, Any]]:
    """Scan runs/ for completed local trained guardrails."""
    if not RUNS_DIR.exists():
        return []

    discovered: list[dict[str, Any]] = []
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name == "smoke_test":
            continue

        cfg = (
            _read_yaml_or_json(entry / "config.yaml")
            or _read_yaml_or_json(entry / "config.json")
            or {}
        )
        metrics = _read_yaml_or_json(entry / "metrics.json") or {}
        gen_cfg = cfg.get("generation", {}) if isinstance(cfg, dict) else {}

        backend = gen_cfg.get("backend")
        if not backend:
            backend = "mlx" if "mlx" in entry.name.lower() else "local"

        model_id = gen_cfg.get("model_id") or entry.name
        label = (
            model_id.split("/")[-1]
            .replace("-Instruct", "")
            .replace("local_", "")
            .replace("mlx_", "")
        )

        test_m = metrics.get("test", metrics) if isinstance(metrics, dict) else {}
        discovered.append(
            {
                "run_name": entry.name,
                "run_path": str(entry.relative_to(ROOT_DIR)),
                "display_name": label,
                "model_id": model_id,
                "backend": backend,
                "backend_label": "Apple MLX · 4-bit" if backend == "mlx" else "PyTorch MPS",
                "metrics": {
                    "accuracy": test_m.get("accuracy"),
                    "auroc": test_m.get("auroc"),
                    "coverage": test_m.get("conformal_coverage")
                    or test_m.get("coverage"),
                    "avg_set_size": test_m.get("avg_set_size")
                    or test_m.get("mean_set_size"),
                },
                "quantiles": _compute_conformal_quantiles(entry),
            }
        )
    return discovered


def _normalize_verdict(raw_verdict: str, pred_set: list[int] | None) -> str:
    """Canonical verdict code: TRUST | UNCERTAIN | DONT_TRUST."""
    v = (raw_verdict or "").upper()
    if pred_set is not None:
        s = set(pred_set)
        if s == {0}:
            return "TRUST"
        if s == {1}:
            return "DONT_TRUST"
        return "UNCERTAIN"
    if "DON'T" in v or "DONT" in v or "HALLUCINAT" in v or "REJECT" in v or "FALSE" in v:
        if "UNCERTAIN" in v or "AMBIG" in v or "RISK" in v:
            return "UNCERTAIN"
        return "DONT_TRUST"
    if "UNCERTAIN" in v or "AMBIG" in v or "ABSTAIN" in v or "FLAG" in v:
        return "UNCERTAIN"
    if "TRUST" in v or "SAFE" in v or "VERIFIED" in v or "FACTUAL" in v:
        return "TRUST"
    return "UNCERTAIN"


def _parse_cli_output(stdout: str, run_info: dict[str, Any]) -> dict[str, Any]:
    """Fallback parser for `hallucination-lens ask --run ...` stdout."""
    lines = [ln.rstrip() for ln in stdout.splitlines()]
    answer = ""
    raw_verdict = ""
    pred_set: list[int] | None = None
    prob_hallucination: float | None = None
    features: dict[str, float] = {}

    for idx, line in enumerate(lines):
        low = line.lower().strip()
        if low.startswith(("answer:", "response:", "output:")):
            answer = line.split(":", 1)[1].strip()
            extra = []
            for nxt in lines[idx + 1 :]:
                if re.match(r"^[A-Za-z0-9 _\-()]+:\s*", nxt):
                    break
                if nxt.strip():
                    extra.append(nxt.strip())
            if extra and not answer:
                answer = " ".join(extra)
        elif low.startswith(("verdict:", "decision:", "guardrail:", "status:")):
            raw_verdict = line.split(":", 1)[1].strip()
        elif "prediction set" in low or "conformal set" in low:
            if "0" in line and "1" in line:
                pred_set = [0, 1]
            elif "factual" in low and "hallucinat" in low:
                pred_set = [0, 1]
            elif "hallucinat" in low or "{1}" in line or "[1]" in line:
                pred_set = [1]
            elif "factual" in low or "{0}" in line or "[0]" in line:
                pred_set = [0]
        elif "p(hallucination)" in low or "hallucination prob" in low or "risk" in low:
            m = re.search(r"([0-9]*\.?[0-9]+)\s*(%?)", line.split(":", 1)[-1])
            if m:
                val = float(m.group(1))
                prob_hallucination = val / 100.0 if m.group(2) == "%" or val > 1.0 else val

        for feat_key in (
            "mean_log_prob",
            "mean_logprob",
            "min_log_prob",
            "min_logprob",
            "var_log_prob",
            "var_logprob",
            "std_log_prob",
            "entropy",
            "mean_entropy",
            "output_length",
            "n_tokens",
        ):
            if feat_key in low:
                m = re.search(r"(-?[0-9]+\.?[0-9]*)", line.split(feat_key, 1)[-1])
                if m:
                    features[feat_key] = float(m.group(1))

    if not answer:
        answer = stdout.strip()

    verdict = _normalize_verdict(raw_verdict, pred_set)
    if pred_set is None:
        pred_set = [0] if verdict == "TRUST" else ([1] if verdict == "DONT_TRUST" else [0, 1])

    return {
        "run_name": run_info["run_name"],
        "display_name": run_info["display_name"],
        "model_id": run_info["model_id"],
        "backend": run_info["backend"],
        "backend_label": run_info["backend_label"],
        "answer": answer,
        "verdict": verdict,
        "raw_verdict": raw_verdict or verdict,
        "is_hallucination_flagged": verdict != "TRUST",
        "prediction_set": pred_set,
        "set_size": len(pred_set),
        "prob_hallucination": prob_hallucination,
        "features": features,
        "token_logprobs": [],
        "tokens": [],
        "raw_console": stdout.strip(),
    }


class GuardrailEngine:
    """Thread-safe model cache and Apple Silicon GPU execution lock."""

    def __init__(self) -> None:
        self._gpu_lock = threading.Lock()
        self._warm_cache: dict[str, Any] = {}

    def _try_load_warm(self, run_dir: Path) -> Any | None:
        """Attempt to instantiate & cache the native Python Guardrail class."""
        key = str(run_dir)
        if key in self._warm_cache:
            return self._warm_cache[key]

        try:
            import hallucination_lens.guardrail as gmod
        except Exception:
            return None

        for attr in ("Guardrail", "ConformalGuardrail", "HallucinationGuardrail"):
            cls = getattr(gmod, attr, None)
            if cls is None:
                continue
            for loader_name in ("load", "from_run", "from_dir"):
                loader = getattr(cls, loader_name, None)
                if callable(loader):
                    try:
                        obj = loader(run_dir)
                        self._warm_cache[key] = obj
                        return obj
                    except Exception:
                        pass
            try:
                sig = inspect.signature(cls.__init__)
                if len(sig.parameters) == 2:
                    obj = cls(run_dir)
                    self._warm_cache[key] = obj
                    return obj
            except Exception:
                pass

        for fn_name in ("load_guardrail", "build_guardrail"):
            fn = getattr(gmod, fn_name, None)
            if callable(fn):
                try:
                    obj = fn(run_dir)
                    self._warm_cache[key] = obj
                    return obj
                except Exception:
                    pass
        return None

    def _extract_from_warm_result(
        self, res: Any, run_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize a dataclass or dict returned by hallucination_lens.guardrail."""
        data = res if isinstance(res, dict) else getattr(res, "__dict__", {})

        def pick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if k in data and data[k] is not None:
                    return data[k]
                if hasattr(res, k) and getattr(res, k) is not None:
                    return getattr(res, k)
            return default

        answer = str(pick("answer", "text", "generated_text", "response", default=""))
        raw_verdict = str(pick("verdict", "decision", "action", "label", default=""))
        pred_set_raw = pick("prediction_set", "conformal_set", "label_set", "set")

        pred_set: list[int] | None = None
        if pred_set_raw is not None:
            pred_set = []
            for item in pred_set_raw:
                if isinstance(item, (int, float)):
                    pred_set.append(int(item))
                elif isinstance(item, str):
                    pred_set.append(1 if "halluc" in item.lower() or item == "1" else 0)
            pred_set = sorted(set(pred_set))

        prob = pick(
            "prob_hallucination",
            "p_hallucination",
            "hallucination_prob",
            "risk_score",
            "proba",
        )
        if isinstance(prob, (list, tuple)) and len(prob) >= 2:
            prob = float(prob[1])
        elif prob is not None:
            prob = float(prob)

        features_raw = pick("features", "feature_dict", default={}) or {}
        features = {
            str(k): round(float(v), 4)
            for k, v in features_raw.items()
            if isinstance(v, (int, float))
        }

        token_lps = pick("token_logprobs", "logprobs", default=[]) or []
        token_lps = [round(float(x), 4) for x in token_lps]

        tokens_raw = pick("tokens", "token_strs", "token_texts", default=[]) or []
        tokens = [str(t) for t in tokens_raw]

        verdict = _normalize_verdict(raw_verdict, pred_set)
        if pred_set is None:
            pred_set = [0] if verdict == "TRUST" else ([1] if verdict == "DONT_TRUST" else [0, 1])

        return {
            "run_name": run_info["run_name"],
            "display_name": run_info["display_name"],
            "model_id": run_info["model_id"],
            "backend": run_info["backend"],
            "backend_label": run_info["backend_label"],
            "answer": answer,
            "verdict": verdict,
            "raw_verdict": raw_verdict or verdict,
            "is_hallucination_flagged": verdict != "TRUST",
            "prediction_set": pred_set,
            "set_size": len(pred_set),
            "prob_hallucination": round(prob, 4) if prob is not None else None,
            "features": features,
            "token_logprobs": token_lps,
            "tokens": tokens,
            "engine_mode": "warm_memory",
        }

    def ask(self, run_name: str, question: str) -> dict[str, Any]:
        runs = {r["run_name"]: r for r in discover_runs()}
        if run_name not in runs:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{run_name}' not found in runs/. Available: {list(runs.keys())}",
            )
        run_info = runs[run_name]
        run_dir = RUNS_DIR / run_name

        with self._gpu_lock:
            t0 = time.perf_counter()

            # 1. Try hot in-memory guardrail first
            warm = self._try_load_warm(run_dir)
            if warm is not None:
                for method_name in ("ask", "check", "predict", "__call__"):
                    fn = getattr(warm, method_name, None)
                    if callable(fn):
                        try:
                            raw_res = fn(question)
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            out = self._extract_from_warm_result(raw_res, run_info)
                            out["latency_ms"] = round(elapsed_ms, 1)
                            n_toks = len(out["token_logprobs"]) or int(
                                out["features"].get("output_length", 0)
                                or out["features"].get("n_tokens", 0)
                            )
                            out["tokens_per_sec"] = (
                                round(n_toks / (elapsed_ms / 1000.0), 1)
                                if n_toks > 0 and elapsed_ms > 0
                                else None
                            )
                            return out
                        except Exception:
                            break

            # 2. Safe CLI fallback (guarantees first-run compatibility)
            cmd = [
                sys.executable,
                "-m",
                "hallucination_lens",
                "ask",
                "--run",
                str(run_dir),
                question,
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=180,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if proc.returncode != 0:
                err_msg = (proc.stderr or proc.stdout or "Unknown CLI error").strip()
                raise HTTPException(
                    status_code=500,
                    detail=f"Model '{run_name}' failed: {err_msg[-600:]}",
                )

            out = _parse_cli_output(proc.stdout, run_info)
            out["latency_ms"] = round(elapsed_ms, 1)
            out["engine_mode"] = "cli_subprocess"
            n_toks = int(
                out["features"].get("output_length", 0)
                or out["features"].get("n_tokens", 0)
            )
            out["tokens_per_sec"] = (
                round(n_toks / (elapsed_ms / 1000.0), 1)
                if n_toks > 0 and elapsed_ms > 0
                else None
            )
            return out


# =============================================================================
# 3. FastAPI Application & Routes
# =============================================================================
history_store = HistoryStore(DB_PATH)
engine = GuardrailEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Hallucination Lens Studio",
    version="2.2.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    run_name: str = Field(..., min_length=1)
    alpha: float = Field(default=0.05, ge=0.01, le=0.50)


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    run_a: str = Field(..., min_length=1)
    run_b: str = Field(..., min_length=1)
    alpha: float = Field(default=0.05, ge=0.01, le=0.50)


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    runs = discover_runs()
    return {"models": runs, "count": len(runs)}


@app.post("/api/ask")
def ask_single(req: AskRequest) -> dict[str, Any]:
    result = engine.ask(req.run_name, req.question.strip())
    quantiles = _compute_conformal_quantiles(RUNS_DIR / req.run_name)
    result = _apply_alpha_to_result(result, req.alpha, quantiles)
    record = history_store.add(
        mode="single",
        question=req.question.strip(),
        primary_model=result["display_name"],
        secondary_model=None,
        verdict_summary=result["verdict"],
        total_latency_ms=result["latency_ms"],
        payload={"result": result, "alpha": round(req.alpha, 2)},
    )
    return record


@app.post("/api/compare")
def ask_compare(req: CompareRequest) -> dict[str, Any]:
    if req.run_a == req.run_b:
        raise HTTPException(
            status_code=400, detail="Select two different models to compare."
        )

    question = req.question.strip()
    t_total = time.perf_counter()
    res_a = engine.ask(req.run_a, question)
    res_b = engine.ask(req.run_b, question)
    total_ms = round((time.perf_counter() - t_total) * 1000.0, 1)

    res_a = _apply_alpha_to_result(
        res_a, req.alpha, _compute_conformal_quantiles(RUNS_DIR / req.run_a)
    )
    res_b = _apply_alpha_to_result(
        res_b, req.alpha, _compute_conformal_quantiles(RUNS_DIR / req.run_b)
    )

    same_verdict = res_a["verdict"] == res_b["verdict"]
    verdict_summary = res_a["verdict"] if same_verdict else "DIVERGENT"

    lat_a, lat_b = res_a["latency_ms"], res_b["latency_ms"]
    if lat_a <= lat_b:
        faster = res_a["display_name"]
        speedup = round(lat_b / max(lat_a, 1.0), 2)
        saved_ms = round(lat_b - lat_a, 1)
    else:
        faster = res_b["display_name"]
        speedup = round(lat_a / max(lat_b, 1.0), 2)
        saved_ms = round(lat_a - lat_b, 1)

    prob_a, prob_b = res_a.get("prob_hallucination"), res_b.get("prob_hallucination")
    risk_delta = (
        round((prob_b - prob_a) * 100.0, 1)
        if prob_a is not None and prob_b is not None
        else None
    )

    comparison = {
        "agreement": same_verdict,
        "faster_model": faster,
        "speedup_ratio": speedup,
        "time_saved_ms": saved_ms,
        "risk_delta_pct": risk_delta,
    }

    record = history_store.add(
        mode="compare",
        question=question,
        primary_model=res_a["display_name"],
        secondary_model=res_b["display_name"],
        verdict_summary=verdict_summary,
        total_latency_ms=total_ms,
        payload={
            "result_a": res_a,
            "result_b": res_b,
            "comparison": comparison,
            "alpha": round(req.alpha, 2),
        },
    )
    return record


@app.get("/api/history")
def get_history(
    q: str = Query(default=""),
    mode: str = Query(default="all"),
    verdict: str = Query(default="all"),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    items = history_store.list(q=q, mode=mode, verdict=verdict, limit=limit)
    return {"items": items, "count": len(items)}


@app.delete("/api/history/{item_id}")
def delete_history_item(item_id: int) -> dict[str, Any]:
    deleted = history_store.delete(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History entry not found.")
    return {"deleted": True, "id": item_id}


@app.delete("/api/history")
def clear_history() -> dict[str, Any]:
    removed = history_store.clear()
    return {"cleared": removed}


@app.get("/api/history/export")
def export_history() -> JSONResponse:
    items = history_store.list(limit=1000)
    return JSONResponse(
        content={"exported_at": datetime.now(timezone.utc).isoformat(), "items": items},
        headers={
            "Content-Disposition": 'attachment; filename="hallucination_lens_history.json"'
        },
    )