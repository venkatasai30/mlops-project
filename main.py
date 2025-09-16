import os
import time
import argparse
from datetime import datetime

import mlflow
from prefect import flow, task, get_run_logger
from sklearn.impute import SimpleImputer
from prefect.context import get_run_context
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ToRecords(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None
        # Pandas DataFrame → list of dicts
        if pd is not None and isinstance(X, pd.DataFrame):
            return X.to_dict(orient="records")
        # Pandas Series → DataFrame → list of dicts
        if pd is not None and isinstance(X, pd.Series):
            return X.to_frame().to_dict(orient="records")
        # Already a list of dicts
        if isinstance(X, list) and (len(X) == 0 or isinstance(X[0], dict)):
            return X
        # Fallback: wrap single dict
        if isinstance(X, dict):
            return [X]
        raise TypeError(f"Unsupported input type for ToRecords: {type(X)}")
from prefect.task_runners import SequentialTaskRunner
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

from utils.prepare import process_data
from utils.drift import compute_baseline, save_json


@task(name="Run models")
def run_models(X_train, y_train, X_valid, y_valid):
    for model_class in (Ridge, GradientBoostingRegressor, RandomForestRegressor):
        with mlflow.start_run():

            # Build and Train model
            model = make_pipeline(
                # Ensure DataFrame inputs work with pyfunc/HTTP by converting rows to dicts
                ToRecords(),
                DictVectorizer(),
                SimpleImputer(),
                model_class(random_state=42),
            )
            # Pass DataFrames; the pipeline converts rows to dicts internally
            model.fit(X_train, y_train)

            # Also attach baseline artifact to this model run for downstream drift checks
            try:
                baseline = compute_baseline(X_train)
                os.makedirs("artifacts", exist_ok=True)
                baseline_path = os.path.join("artifacts", "baseline.json")
                save_json(baseline_path, baseline)
                mlflow.log_artifact(baseline_path, artifact_path="baseline")
            except Exception:
                pass

            # MLflow logging
            start_time = time.time()
            y_pred_train = model.predict(X_train)
            y_pred_valid = model.predict(X_valid)
            inference_time = time.time() - start_time

            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_valid = mean_absolute_error(y_valid, y_pred_valid)
            rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
            rmse_valid = mean_squared_error(y_valid, y_pred_valid, squared=False)

            mlflow.set_tag("author/developer", "PatrickCmd")
            mlflow.set_tag("Model", f"{model_class}")

            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mae_valid", mae_valid)
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("rmse_valid", rmse_valid)
            mlflow.log_metric(
                "inference_time",
                inference_time / (len(y_pred_train) + len(y_pred_valid)),
            )


@flow(name="mlflow-training", task_runner=SequentialTaskRunner())
def main(train_file, valid_file):
    # Set and run experiment
    ctx = get_run_context()
    # Allow overriding tracking server via env var
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    # Allow overriding experiment name; default to date-stamped name
    default_date = None
    try:
        default_date = ctx.flow_run.expected_start_time.strftime("%Y-%m-%d")
    except Exception:
        default_date = datetime.utcnow().strftime("%Y-%m-%d")
    EXPERIMENT_NAME = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        f"citibikes-experiment-{default_date}",
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    logger = get_run_logger()
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")
    logger.info("Process data features for model training and validation")
    X_train, y_train, X_valid, y_valid = process_data(train_file, valid_file)
    logger.info(
        f"Train and Validation df shapes: {X_train.shape}, {y_train.shape}, {X_valid.shape}, {y_valid.shape}"
    )

    # Run models
    logger.info("Training models")
    # Log baseline distribution as an artifact for drift checks
    try:
        baseline = compute_baseline(X_train)
        os.makedirs("artifacts", exist_ok=True)
        baseline_path = os.path.join("artifacts", "baseline.json")
        save_json(baseline_path, baseline)
        mlflow.log_artifact(baseline_path, artifact_path="baseline")
    except Exception as e:
        logger.warning(f"Failed to compute/log baseline distribution: {e}")
    run_models(X_train, y_train, X_valid, y_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="file for training data.")
    parser.add_argument("--valid_file", help="file for validation data.")
    args = parser.parse_args()

    parameters = {
        "train_file": args.train_file,
        "valid_file": args.valid_file,
    }
    main(**parameters)
