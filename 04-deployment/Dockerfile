# Use the base image provided
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your Python script into the container
COPY homework.py .

# Copiar el archivo yellow_tripdata_2023-05.parquet al directorio de trabajo dentro del contenedor
COPY yellow_tripdata_2023-05.parquet /app/yellow_tripdata_2023-05.parquet

# Install required Python packages
RUN pip install pandas pyarrow scikit-learn==1.4.2

# Command to run the script
CMD ["python", "homework.py", "2023", "05"]
