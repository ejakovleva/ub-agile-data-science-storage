import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("models:/trip-duration@champion")
trips = pd.DataFrame({
    "trip_distance": [1.0, 10.0],
    "pickup_hour": [8.0, 3.0],
    "pickup_weekday": [0.0, 6.0],
    "PULocationID": [161.0, 132.0],
    "DOLocationID": [237.0, 230.0],
    })
print(model.predict(trips))