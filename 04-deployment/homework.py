#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[6]:


import pickle
import pandas as pd


# In[8]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[9]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[11]:


df = read_data(r'C:\Users\HP\Documents\Python_Projects\mlops-zoomcamp\03-orchestration\dataset\yellow_tripdata_2023-03.parquet')


# In[12]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[14]:


import numpy as np
# Calculate the standard deviation
std_dev = np.std(y_pred)

print(f"The standard deviation of the predicted duration is: {std_dev}")


# In[16]:


import pyarrow.parquet as pq

# Suponiendo que 'df' es tu DataFrame original y 'y_pred' son las predicciones
# Asegúrate de que year y month estén definidos con los valores correctos
year = 2023
month = 3

# Crear la columna 'ride_id'
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Crear el DataFrame de resultados con 'ride_id' y 'y_pred'
df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'predicted_duration': y_pred
})

# Guardar el DataFrame en un archivo Parquet
output_file = 'output.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Comprobar el tamaño del archivo
import os
file_size = os.path.getsize(output_file)
print(f"File size: {file_size} bytes ({file_size / (1024 ** 2):.2f} MB)")

