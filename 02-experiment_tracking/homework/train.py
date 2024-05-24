import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # Starting MLflow run
    with mlflow.start_run():
        # Loading training and validation data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Creating and training the RandomForestRegressor model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # Calculating RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Logging parameters and metrics to MLflow
        mlflow.log_params(rf.get_params())
        mlflow.log_metric("RMSE", rmse)


if __name__ == '__main__':
    run_train()