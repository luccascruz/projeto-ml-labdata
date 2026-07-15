# Imagem custom do Airflow com as dependencias do pipeline (pandas, s3fs, sklearn...).
FROM apache/airflow:2.10.5-python3.12

COPY requirements.txt /requirements.txt
# --trusted-host: contorna falha de SSL em redes corporativas com inspecao
# HTTPS (proxy/antivirus). Ver MLOps/README.md.
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r /requirements.txt
