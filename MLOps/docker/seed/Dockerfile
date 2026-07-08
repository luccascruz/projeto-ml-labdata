# Seeder: sobe os CSVs brutos (montados em /data) para o bucket `raw` do MinIO.
FROM python:3.12-slim

RUN pip install --no-cache-dir boto3==1.35.36 pyyaml==6.0.2

WORKDIR /app
COPY seed_minio.py /app/seed_minio.py

ENTRYPOINT ["python", "seed_minio.py"]
