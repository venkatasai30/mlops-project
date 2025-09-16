import json
import math
from datetime import datetime
from typing import Dict, List, Tuple
import os

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None


def _safe_perc(arr: np.ndarray) -> np.ndarray:
    s = arr.sum()
    if s == 0:
        return np.zeros_like(arr, dtype=float)
    return arr / s


def compute_categorical_dist(series: pd.Series) -> Dict[str, float]:
    counts = series.astype(str).value_counts(dropna=False)
    perc = (counts / counts.sum()).to_dict()
    return {str(k): float(v) for k, v in perc.items()}


def compute_numeric_bins(series: pd.Series, bins: int = 10) -> Tuple[List[float], np.ndarray]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    # Quantile-based bins are robust to skew
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(clean, qs))
    if len(edges) < 2:
        # fallback to min/max range if degenerate
        mn = float(clean.min()) if len(clean) else 0.0
        mx = float(clean.max()) if len(clean) else 1.0
        edges = np.linspace(mn, mx, bins + 1)
    hist, _ = np.histogram(clean, bins=edges)
    return list(map(float, edges)), hist.astype(float)


def psi(train_p: np.ndarray, current_p: np.ndarray, eps: float = 1e-8) -> float:
    # population stability index
    tp = np.clip(train_p, eps, None)
    cp = np.clip(current_p, eps, None)
    return float(np.sum((cp - tp) * np.log(cp / tp)))


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-8) -> float:
    # KL(q || p): current vs baseline
    keys = set(p.keys()) | set(q.keys())
    s = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        s += qk * math.log(qk / pk)
    return float(s)


def compute_baseline(X: pd.DataFrame) -> Dict:
    baseline: Dict = {"created": datetime.utcnow().isoformat()}
    # Categorical
    if "rideable_type" in X.columns:
        baseline["rideable_type_dist"] = compute_categorical_dist(X["rideable_type"])
    # Numeric
    if "trip_distance" in X.columns:
        edges, hist = compute_numeric_bins(X["trip_distance"], bins=10)
        baseline["trip_distance_edges"] = edges
        baseline["trip_distance_hist"] = _safe_perc(hist).tolist()
        baseline["trip_distance_mean"] = float(pd.to_numeric(X["trip_distance"], errors="coerce").dropna().mean())
        baseline["trip_distance_std"] = float(pd.to_numeric(X["trip_distance"], errors="coerce").dropna().std(ddof=0))
    baseline["n_train"] = int(len(X))
    return baseline


def compute_drift_metrics(baseline: Dict, X: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # KL for categorical
    if "rideable_type_dist" in baseline and "rideable_type" in X.columns:
        curr = compute_categorical_dist(X["rideable_type"])
        metrics["kl_rideable_type"] = kl_divergence(baseline["rideable_type_dist"], curr)
    # PSI + z-score for numeric
    if "trip_distance_edges" in baseline and "trip_distance" in X.columns:
        edges = np.array(baseline["trip_distance_edges"])  # type: ignore[arg-type]
        hist_train = np.array(baseline["trip_distance_hist"], dtype=float)
        clean = pd.to_numeric(X["trip_distance"], errors="coerce").dropna()
        hist_curr, _ = np.histogram(clean, bins=edges)
        hist_curr = _safe_perc(hist_curr.astype(float))
        metrics["psi_trip_distance"] = psi(hist_train, hist_curr)
        mu_t = baseline.get("trip_distance_mean", 0.0)
        sd_t = max(baseline.get("trip_distance_std", 1e-8), 1e-8)
        mu_c = float(clean.mean()) if len(clean) else float("nan")
        metrics["z_trip_distance"] = abs(mu_c - mu_t) / sd_t
    metrics["n_current"] = float(len(X))
    return metrics


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def generate_drift_plots(baseline: Dict, current: pd.DataFrame, out_dir: str) -> List[str]:
    """Generate small overlay plots for categorical and numeric features.

    Returns list of saved file paths. Safe no-op if matplotlib unavailable.
    """
    paths: List[str] = []
    if plt is None:
        return paths
    os.makedirs(out_dir, exist_ok=True)

    # Categorical: rideable_type
    if "rideable_type_dist" in baseline and "rideable_type" in current.columns:
        base_dist = baseline["rideable_type_dist"]
        curr_dist = compute_categorical_dist(current["rideable_type"])  # type: ignore
        keys = sorted(set(base_dist.keys()) | set(curr_dist.keys()))
        base_vals = [base_dist.get(k, 0.0) for k in keys]
        curr_vals = [curr_dist.get(k, 0.0) for k in keys]
        x = np.arange(len(keys))
        w = 0.4
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(x - w / 2, base_vals, width=w, label="train")
        ax.bar(x + w / 2, curr_vals, width=w, label="current")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_ylabel("share")
        ax.set_title("rideable_type distribution")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "rideable_type_dist.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # Numeric: trip_distance
    if "trip_distance_edges" in baseline and "trip_distance" in current.columns:
        edges = np.array(baseline["trip_distance_edges"], dtype=float)
        hist_train = np.array(baseline["trip_distance_hist"], dtype=float)
        clean = pd.to_numeric(current["trip_distance"], errors="coerce").dropna()
        hist_curr, _ = np.histogram(clean, bins=edges)
        hist_curr = _safe_perc(hist_curr.astype(float))
        mids = (edges[:-1] + edges[1:]) / 2.0
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(mids, hist_train, drawstyle="steps-mid", label="train")
        ax.plot(mids, hist_curr, drawstyle="steps-mid", label="current")
        ax.set_xlabel("trip_distance")
        ax.set_ylabel("share")
        ax.set_title("trip_distance distribution")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "trip_distance_hist.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    return paths
