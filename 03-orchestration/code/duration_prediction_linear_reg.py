#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import mlflow
import pandas as pd
import pickle
import argparse
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = []
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.log_param("model_type", "LinearRegression")

        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("intercept", lr.intercept_)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        with open("models/linear_regression.b", "wb") as f_out:
            pickle.dump(lr, f_out)
        mlflow.log_artifact("models/linear_regression.b", artifact_path="models_mlflow")

        model_size = os.path.getsize("models/linear_regression.b")
        mlflow.log_metric("model_size_bytes", model_size)

        print("Intercept of the model:", lr.intercept_)
        print("Model_size_bytes:", model_size)

        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    print("Starting the training process...")

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print(f"Training completed. Run ID: {run_id}")