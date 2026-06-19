#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scipy.stats as sstats  # type: ignore
except Exception:
    sstats = None

ROOT = Path("/home/dineth/focus_measure/journal/Regression")
DATA_OUT = ROOT / "data" / "out_final_phase"
TABLES_ROOT = ROOT / "tables" / "FINAL_PHASE"

TRACKS = ["smears", "biopsy"]


@dataclass
class EvalPaths:
    track: str
    sign_dir: Path
    reg_dir: Path
    eval_dir: Path
    tables_dir: Path
    log_file: Path


def parse_k_values(raw: list[str] | None) -> list[int]:
    if not raw:
        return [3, 5, 7]
    return [int(x) for x in raw]


def ensure_track(track: str) -> None:
    if track not in TRACKS:
        raise ValueError(f"Unsupported track={track}. Expected one of {TRACKS}")


def get_paths(track: str) -> EvalPaths:
    ensure_track(track)
    sign_dir = DATA_OUT / track / "sign"
    reg_dir = DATA_OUT / track / "regression"
    eval_dir = DATA_OUT / track / "evaluation"
    tables_dir = TABLES_ROOT / track.upper()

    eval_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return EvalPaths(
        track=track,
        sign_dir=sign_dir,
        reg_dir=reg_dir,
        eval_dir=eval_dir,
        tables_dir=tables_dir,
        log_file=eval_dir / "evaluation_log.txt",
    )


def log(paths: EvalPaths, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    with paths.log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {what}: {path}")
    return path


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_table(df: pd.DataFrame, csv_path: Path, save_latex: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if save_latex:
        tex_path = csv_path.with_suffix(".tex")
        with tex_path.open("w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))


def safe_to_numeric(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column '{name}' could not be converted to numeric values")
    return out


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank-based AUROC (Mann-Whitney U)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = u / (n_pos * n_neg)
    return float(auc)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = float(0.5 * (tpr + tnr))
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-12))

    out = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    if y_prob is not None:
        out["auroc"] = float(auroc_score(y_true, np.asarray(y_prob, dtype=float)))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    abs_err = np.abs(err)

    out = {
        "mae_um": float(np.mean(abs_err)) if abs_err.size else float("nan"),
        "rmse_um": float(np.sqrt(np.mean(np.square(err)))) if err.size else float("nan"),
        "median_abs_error_um": float(np.median(abs_err)) if abs_err.size else float("nan"),
        "p95_abs_error_um": float(np.percentile(abs_err, 95)) if abs_err.size else float("nan"),
    }
    return out


def bin_edges_from_width(values: np.ndarray, width: float) -> np.ndarray:
    vmax = float(np.nanmax(values)) if values.size else 0.0
    vmax = max(vmax, width)
    n_bins = int(math.ceil(vmax / width))
    edges = np.arange(0, (n_bins + 1) * width + 1e-12, width, dtype=float)
    if edges[-1] < vmax:
        edges = np.append(edges, vmax)
    return edges


def per_bin_classification(
    abs_delta: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_width: float,
) -> pd.DataFrame:
    abs_delta = np.asarray(abs_delta, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    edges = bin_edges_from_width(abs_delta, bin_width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (abs_delta >= lo) & (abs_delta < hi)
        n = int(idx.sum())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "accuracy": np.nan, "balanced_accuracy": np.nan})
            continue
        m = classification_metrics(y_true[idx], y_pred[idx])
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": n,
                "accuracy": float(m["accuracy"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def per_bin_regression(abs_delta: np.ndarray, abs_err: np.ndarray, bin_width: float) -> pd.DataFrame:
    abs_delta = np.asarray(abs_delta, dtype=float)
    abs_err = np.asarray(abs_err, dtype=float)

    edges = bin_edges_from_width(abs_delta, bin_width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (abs_delta >= lo) & (abs_delta < hi)
        n = int(idx.sum())
        mae = float(np.mean(abs_err[idx])) if n > 0 else np.nan
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": n, "mae_um": mae})
    return pd.DataFrame(rows)


def weighted_median(values: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float | None:
    vals = np.asarray(values, dtype=float)
    wts = np.asarray(weights, dtype=float)

    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not mask.any():
        return None

    vals = vals[mask]
    wts = wts[mask]

    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]

    total = float(np.sum(wts))
    if total <= eps:
        return None

    cutoff = 0.5 * total
    csum = np.cumsum(wts)
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = min(max(idx, 0), len(vals) - 1)
    return float(vals[idx])


def bootstrap_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = []
    n = len(values)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        stats.append(metric_fn(sample))
    stats = np.asarray(stats, dtype=float)

    point = float(metric_fn(values))
    lo = float(np.percentile(stats, 100 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2.0)))
    return point, lo, hi


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[np.isfinite(diff)]
    diff = diff[~np.isclose(diff, 0.0)]

    if diff.size == 0:
        return {"statistic": np.nan, "pvalue": np.nan, "method": "wilcoxon_no_data"}

    if sstats is not None:
        try:
            stat, p = sstats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
            return {"statistic": float(stat), "pvalue": float(p), "method": "scipy_wilcoxon"}
        except Exception:
            pass

    # Fallback: sign test approximation (binomial normal approximation)
    n_pos = int((diff > 0).sum())
    n = int(diff.size)
    z = (n_pos - 0.5 * n) / math.sqrt(max(n * 0.25, 1e-12))
    # two-sided normal p-value
    p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"statistic": float(z), "pvalue": float(p), "method": "fallback_sign_test"}


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(y_pred_a).astype(int)
    b = np.asarray(y_pred_b).astype(int)

    a_ok = a == y_true
    b_ok = b == y_true

    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())

    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "chi2": np.nan, "pvalue": np.nan, "method": "mcnemar_no_discordant"}

    chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)

    if sstats is not None:
        try:
            p = 1.0 - sstats.chi2.cdf(chi2, df=1)
            return {"n01": n01, "n10": n10, "chi2": float(chi2), "pvalue": float(p), "method": "chi2_cc"}
        except Exception:
            pass

    # Fallback normal approx
    z = (abs(n01 - n10) - 1.0) / math.sqrt(max(n01 + n10, 1e-12))
    p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "pvalue": float(p), "method": "fallback_normal"}


def friedman_nemenyi(scores_by_model: dict[str, np.ndarray]) -> dict:
    names = list(scores_by_model.keys())
    arrays = [np.asarray(scores_by_model[n], dtype=float) for n in names]

    n = min(len(a) for a in arrays) if arrays else 0
    if n == 0 or len(names) < 3:
        return {
            "method": "insufficient_data",
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
            "avg_ranks": {},
            "critical_difference": np.nan,
            "pairwise_rank_diff": {},
        }

    arrays = [a[:n] for a in arrays]

    # lower score => better rank 1
    rank_mat = np.vstack([pd.Series(row).rank(method="average", ascending=True).to_numpy() for row in np.vstack(arrays).T])
    avg_ranks = rank_mat.mean(axis=0)
    avg_ranks_dict = {names[i]: float(avg_ranks[i]) for i in range(len(names))}

    if sstats is not None:
        try:
            stat, p = sstats.friedmanchisquare(*arrays)
            fried_stat = float(stat)
            fried_p = float(p)
        except Exception:
            fried_stat, fried_p = np.nan, np.nan
    else:
        fried_stat, fried_p = np.nan, np.nan

    k = len(names)
    # Approximate q_alpha for alpha=0.05, large Nemenyi table.
    q_alpha = 2.569
    cd = q_alpha * math.sqrt((k * (k + 1)) / (6.0 * n))

    pair = {}
    for i in range(k):
        for j in range(i + 1, k):
            key = f"{names[i]}_vs_{names[j]}"
            pair[key] = float(abs(avg_ranks[i] - avg_ranks[j]))

    return {
        "method": "friedman_nemenyi",
        "friedman_stat": fried_stat,
        "friedman_p": fried_p,
        "avg_ranks": avg_ranks_dict,
        "critical_difference": float(cd),
        "pairwise_rank_diff": pair,
    }


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.unique(y_score)[::-1]
    tpr = []
    fpr = []
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    pos = max(int(pos), 1)
    neg = max(int(neg), 1)

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        tpr.append(tp / pos)
        fpr.append(fp / neg)

    # append endpoints
    tpr = np.array([0.0] + tpr + [1.0], dtype=float)
    fpr = np.array([0.0] + fpr + [1.0], dtype=float)
    return fpr, tpr


def save_plot(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_phase5_index(track: str) -> pd.DataFrame:
    p = DATA_OUT / track / "regression" / "index_phase5.csv"
    require_file(p, "phase5 index")
    df = pd.read_csv(p, low_memory=False)
    req = ["roi_uid", "dataset", "fov_id", "defocus_um", "y_sign", "y_mag_um"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Phase5 index missing columns {missing}: {p}")
    return df


def infer_dataset_for_fov(index_df: pd.DataFrame) -> pd.DataFrame:
    # most frequent dataset per fov
    tmp = (
        index_df.groupby(["fov_id", "dataset"]).size().reset_index(name="n").sort_values(["fov_id", "n"], ascending=[True, False])
    )
    tmp = tmp.drop_duplicates(subset=["fov_id"], keep="first").reset_index(drop=True)
    return tmp[["fov_id", "dataset"]]


def default_eval_cli(parser):
    parser.add_argument("--track", required=True, choices=TRACKS)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-latex", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip execution when expected outputs already exist.")
    parser.add_argument("--bins", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--k-values", nargs="+", default=["3", "5", "7"])
    return parser
