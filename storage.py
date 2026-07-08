"""Helper de I/O no MinIO (S3) para o pipeline.

Todo o pipeline (raw -> clean -> abt -> models) le e grava em buckets do MinIO.
Nao ha modo local de filesystem: a I/O do pipeline e sempre S3.

Os nomes dos buckets vem do config (bloco `storage.buckets`); o endpoint e as
credenciais vem SEMPRE de variaveis de ambiente (MINIO_ENDPOINT,
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) — nunca do config.yml versionado.
Portavel (Linux/macOS/Windows): so depende de pandas + s3fs.
"""
from __future__ import annotations

import os


class Storage:
    """Resolve caminhos e opcoes de acesso aos buckets do MinIO."""

    def __init__(self, cfg: dict):
        self._buckets = cfg["storage"]["buckets"]

    def path(self, layer: str, filename: str) -> str:
        """URI s3://bucket/arquivo para a camada (raw | clean | abt | models)."""
        return f"s3://{self._buckets[layer]}/{filename}"

    @property
    def storage_options(self) -> dict:
        """Credenciais/endpoint para pandas + s3fs, lidos de variaveis de ambiente."""
        return {
            "key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
            "client_kwargs": {"endpoint_url": os.environ["MINIO_ENDPOINT"]},
        }

    def io_kwargs(self) -> dict:
        """kwargs para pd.read_* / DataFrame.to_* (sempre com storage_options)."""
        return {"storage_options": self.storage_options}

    def open(self, layer: str, filename: str, mode: str = "rb"):
        """File-like via s3fs para artefatos binarios (joblib / matplotlib)."""
        import s3fs

        fs = s3fs.S3FileSystem(**self.storage_options)
        return fs.open(self.path(layer, filename), mode)


def get_storage(cfg: dict) -> Storage:
    return Storage(cfg)
