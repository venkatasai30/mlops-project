import os
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def main():
    tracking_uri = os.environ.get("TRACKING_URI", os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    today = datetime.utcnow().strftime("%Y-%m-%d")
    experiment_name = os.environ.get("EXPERIMENT_NAME", os.environ.get("MLFLOW_EXPERIMENT_NAME", f"citibikes-experiment-{today}"))

    print(f"Using tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise SystemExit(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.rmse_valid ASC"],
        max_results=5,
    )
    if not runs:
        raise SystemExit("No runs found to stage.")

    best = runs[0]
    run_id = best.info.run_id
    name = f"CITIBIKESDurationModel-{run_id}"
    print(f"Best run: {run_id} rmse_valid={best.data.metrics.get('rmse_valid')}")

    try:
        registered = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=name)
    except Exception as e:
        # Fallback if the registered model already exists
        print(f"register_model exception: {e}; creating version via client...")
        try:
            client.create_registered_model(name)
        except Exception:
            pass
        registered = client.create_model_version(name=name, source=f"runs:/{run_id}/model", run_id=run_id)

    client.transition_model_version_stage(name=name, version=registered.version, stage="Staging")
    client.update_model_version(
        name=name,
        version=registered.version,
        description=f"[{datetime.now()}] Version {registered.version} from '{experiment_name}' transitioned to Staging.",
    )
    print(f"Staged model: {name} v{registered.version}")


if __name__ == "__main__":
    main()

