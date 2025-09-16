
PORT ?= 5000
MLFLOW_PORT ?= 5001

.PHONY: quality_checks setup mlflow train kill-port serve-run predict drift drift-flow drift-deploy


quality_checks:
	isort .
	black .

setup:
	pipenv install
	pipenv install --dev

mlflow:
	PORT=$(MLFLOW_PORT) ./scripts/run_mlflow_local.sh


train:
	./scripts/train_model.sh

kill-port:
	bash scripts/kill_port.sh $(PORT)

stage:
	./scripts/stage_best.sh

serve-run:
	PORT=$(SERVE_PORT) bash scripts/serve_run.sh

predict:
	@HOST=$${HOST:-127.0.0.1}; PORT=$${PORT:-1237}; \
	JSON='{"dataframe_split":{"columns":["rideable_type","start_end_id","trip_distance"],"data":[["electric_bike","123_456",2.7]]}}'; \
	curl -sS -X POST http://$${HOST}:$${PORT}/invocations -H 'Content-Type: application/json' -d "$${JSON}"

drift:
	@TRACKING_URI=$${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}; \
	CURRENT=$${CURRENT_FILE:-./data/202205-capitalbikeshare-tripdata.parquet}; \
	MODEL_NAME=$${MODEL_NAME:-CITIBIKESDurationModel}; \
	if [ -n "$$MODEL_NAME" ]; then \
		pipenv run python scripts/drift_check.py --tracking_uri $$TRACKING_URI --current_file $$CURRENT --model_name $$MODEL_NAME --stage Staging; \
	else \
		echo "Please set MODEL_NAME or pass --run_id in a custom call"; \
	fi

drift-flow:
	PREFECT_HOME="$(PWD)/.prefect" \
		pipenv run python -m flows.drift_monitor --tracking_uri $${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001} --current_file $${CURRENT_FILE:-./data/202205-capitalbikeshare-tripdata.parquet}

drift-deploy:
	PREFECT_HOME="$(PWD)/.prefect" \
		pipenv run prefect deployment build flows/drift_monitor.py:monitor_drift -n daily-drift --cron "0 9 * * *" --timezone "$(shell date +%Z)" -o drift-deployment.yaml
	PREFECT_HOME="$(PWD)/.prefect" pipenv run prefect deployment apply drift-deployment.yaml
