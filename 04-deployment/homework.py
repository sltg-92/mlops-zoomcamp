#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn import __version__ as sklearn_version

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    # Assuming the data filename follows a specific pattern like 'yellow_tripdata_{year}-{month}.parquet'
    filename = rf'C:\Users\HP\Documents\Python_Projects\mlops-zoomcamp\04-deployment\yellow_tripdata_{year}-{month:02}.parquet'
    
    df = read_data(filename)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Calculate mean predicted duration
    mean_predicted_duration = np.mean(y_pred)
    
    print(f"The mean predicted duration for {year}-{month:02} is: {mean_predicted_duration:.2f} minutes")
    
    # Creating ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Creating results DataFrame with ride_id and predicted_duration
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    # Saving DataFrame to Parquet file
    output_file = 'output.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Checking file size
    import os
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size} bytes ({file_size / (1024 ** 2):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process year and month for prediction.')
    parser.add_argument('year', type=int, help='The year (e.g., 2023)')
    parser.add_argument('month', type=int, choices=range(1, 13), help='The month (1-12)')
    args = parser.parse_args()

    main(args.year, args.month)


