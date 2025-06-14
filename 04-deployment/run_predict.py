import os
import pickle
import argparse
import numpy as np
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def prepare_predictions(df, dv, model, categorical, year, month):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred
    
    return df_result, y_pred


def run(year: int, month: int, model_path: str):
    categorical = ['PULocationID', 'DOLocationID']
    taxi_type = 'yellow'
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_{year:04d}-{month:02d}.parquet'
    
    dv, model = load_model(model_path)
    df = read_data(input_file, categorical)
    df_result, y_pred = prepare_predictions(df, dv, model, categorical, year, month)
    
    return {
        'mean_predicted_duration': float(np.mean(y_pred)),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year to process"
    )
    parser.add_argument(
        "--month",
        type=int,
        default=3,
        help="Month to process"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./app/model.bin",
        help="Path to the model file"
    )
    
    args = parser.parse_args()
    
    results = run(args.year, args.month, args.model_path)
    print(f"Mean predicted duration: {results['mean_predicted_duration']:.2f}")

