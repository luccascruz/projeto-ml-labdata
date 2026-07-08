# Imagem custom do Airflow com as dependencias do pipeline (pandas, s3fs, sklearn...).
FROM apache/airflow:2.10.5-python3.12

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
