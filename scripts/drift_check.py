#!/usr/bin/env python
import sys, os as _os

if "PYTHONPATH" not in _os.environ:
    _os.environ["PYTHONPATH"] = _os.getcwd()
    sys.path.insert(0, _os.getcwd())
import argparse
import json
import os
from typing import Dict

import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from utils.drift import compute_drift_metrics, compute_baseline, generate_drift_plots
from haversine import haversine, Unit


def load_current(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv"}:
        return pd.read_csv(path)
    if ext in {".parquet"}:
        return pd.read_parquet(path)
    if ext in {".json"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {ext}")


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce feature frame with at least rideable_type and trip_distance."""
    out = pd.DataFrame()
    if "rideable_type" in df.columns:
        out["rideable_type"] = df["rideable_type"].astype(str)
    # trip_distance present or compute from lat/lng
    if "trip_distance" in df.columns:
        out["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")
    elif {"start_lat", "start_lng", "end_lat", "end_lng"}.issubset(df.columns):
        out = df.copy()
        def _dist(row):
            try:
                return haversine((row["start_lat"], row["start_lng"]), (row["end_lat"], row["end_lng"]), unit=Unit.MILES)
            except Exception:
                return float("nan")
        out["trip_distance"] = out.apply(_dist, axis=1)
        out = out[["rideable_type", "trip_distance"]]
    else:
        # fallback with NaNs
        out["trip_distance"] = pd.Series(dtype=float)
    return out


def load_baseline_from_run(run_id: str) -> Dict:
    # Download artifact baseline/baseline.json
    local_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/baseline")
    path = os.path.join(local_dir, "baseline.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_id_from_model(model_name: str, stage: str) -> str:
    c = MlflowClient()
    versions = c.get_latest_versions(model_name, [stage])
    if not versions:
        raise SystemExit(f"No versions for model {model_name} at stage {stage}")
    return versions[0].run_id


def main():
    ap = argparse.ArgumentParser(description="Lightweight drift detection and logging")
    ap.add_argument("--tracking_uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    ap.add_argument("--current_file", required=True, help="Path to current data (csv/parquet/json)")
    ap.add_argument("--baseline_file", help="Optional baseline data file (csv/parquet/json); if set, skip fetching from MLflow")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--run_id", help="Baseline run id (contains baseline artifact)")
    group.add_argument("--model_name", help="Model name to resolve baseline from")
    ap.add_argument("--stage", default="Staging", help="Model stage when using --model_name")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    baseline = None
    run_id = None
    if args.baseline_file:
        base_df = load_current(args.baseline_file)
        base_df = ensure_features(base_df)
        baseline = compute_baseline(base_df)
    else:
        if not (args.run_id or args.model_name):
            raise SystemExit("Provide --baseline_file or one of --run_id/--model_name")
        run_id = args.run_id or resolve_run_id_from_model(args.model_name, args.stage)
        baseline = load_baseline_from_run(run_id)
    current = ensure_features(load_current(args.current_file))

    metrics = compute_drift_metrics(baseline, current)

    # Log drift metrics as a separate run
    exp_name = f"drift-monitoring"
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f"drift-{args.model_name or run_id}"):
        for k, v in metrics.items():
            if v is not None and v == v:  # filter NaN
                mlflow.log_metric(k, float(v))
        # thresholds as params
        mlflow.log_param("psi_threshold_warn", 0.2)
        mlflow.log_param("psi_threshold_alert", 0.3)
        # Save detailed report
        report = {
            "baseline_run_id": run_id,
            "current_file": args.current_file,
            "metrics": metrics,
        }
        tmp = "drift_report.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(tmp, artifact_path="drift")
        # Generate small plots and log them
        try:
            out_dir = "drift_plots"
            paths = generate_drift_plots(baseline, current, out_dir)
            for p in paths:
                mlflow.log_artifact(p, artifact_path="drift")
        except Exception as e:
            print(f"WARN: failed to generate drift plots: {e}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
