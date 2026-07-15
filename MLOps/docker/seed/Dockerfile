# Seeder: sobe os CSVs brutos (montados em /data) para o bucket `raw` do MinIO.
FROM python:3.12-slim

# --trusted-host: contorna falha de SSL em redes corporativas com inspecao
# HTTPS (proxy/antivirus). Ver MLOps/README.md.
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org boto3==1.35.36 pyyaml==6.0.2

WORKDIR /app

COPY seed_minio.py /app/seed_minio.py
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r /requirements.txt

ENTRYPOINT ["python", "seed_minio.py"]
