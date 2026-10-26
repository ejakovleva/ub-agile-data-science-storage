import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from features import load_trips

X_train, X_test, y_train, y_test = load_trips()

mlflow.set_experiment("trip-duration")
mlflow.sklearn.autolog()

for max_depth in [2, 5, 10, 20]:
    for est in [10,50]:
        with mlflow.start_run(run_name=f"rf-depth-{max_depth}"):
            params = {"n_estimators": est, "max_depth": max_depth}
            mlflow.log_params(params)
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(
            model, name="model", input_example=X_test[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            print(f"max_depth={max_depth:2} n_estimators={est:2} MAE: {mae:.2f} minutes")