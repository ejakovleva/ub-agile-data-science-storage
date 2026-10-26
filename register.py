import mlflow

best = mlflow.search_runs(
 experiment_names=["trip-duration"],
 order_by=["metrics.mae ASC"],
 max_results=1,
).iloc[0]
version = mlflow.register_model(f"runs:/{best.run_id}/model",
"trip-duration")
print("version", version.version)
client = mlflow.MlflowClient()
client.set_registered_model_alias("trip-duration", "champion", version.version)
