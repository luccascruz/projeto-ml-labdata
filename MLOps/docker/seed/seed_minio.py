"""Carga inicial: sobe os CSVs brutos de /data para o bucket `raw` do MinIO.

Descobre QUAIS arquivos sao o raw e qual e o bucket lendo DataPipeline/config.yml
(data.raw_files e storage.buckets.raw), sem hardcoding — o config e a unica fonte
do nome do bucket. Idempotente: pula chaves que ja existem (a menos de
SEED_OVERWRITE). Endpoint e credenciais vem de variaveis de ambiente.
"""
import os
import sys

import boto3
import yaml
from botocore.client import Config
from botocore.exceptions import ClientError

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yml")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
OVERWRITE = os.environ.get("SEED_OVERWRITE", "false").lower() == "true"

with open(CONFIG_PATH, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
raw_files = cfg["data"]["raw_files"]  # {logico: nome_do_arquivo}
BUCKET = os.environ.get("RAW_BUCKET") or cfg["storage"]["buckets"]["raw"]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["MINIO_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    config=Config(signature_version="s3v4"),
)


def key_exists(key: str) -> bool:
    """True se a key existe; propaga erros que nao sejam 'nao encontrado'
    (credenciais invalidas, bucket ausente, rede fora) em vez de mascara-los."""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def main() -> None:
    available = 0
    for _, fname in raw_files.items():
        src = os.path.join(DATA_DIR, fname)
        if not os.path.exists(src):
            if key_exists(fname):
                print(f"[seed] ausente localmente mas ja existe em s3://{BUCKET}/{fname}")
                available += 1
            else:
                print(f"[seed] ausente, pulando: {src}")
            continue
        if key_exists(fname) and not OVERWRITE:
            print(f"[seed] ja existe em s3://{BUCKET}/{fname}, pulando")
            available += 1
            continue
        print(f"[seed] enviando {src} -> s3://{BUCKET}/{fname}")
        s3.upload_file(src, BUCKET, fname)
        available += 1

    if available == 0:
        print(f"[seed] ERRO: nenhum arquivo de {list(raw_files.values())} disponivel em {DATA_DIR} ou em s3://{BUCKET}")
        sys.exit(1)
    print("[seed] concluido")


if __name__ == "__main__":
    main()
