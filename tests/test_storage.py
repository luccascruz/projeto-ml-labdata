"""Testes do resolvedor de storage (S3-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import get_storage  # noqa: E402

CFG = {
    "storage": {
        "buckets": {"raw": "raw", "clean": "clean", "abt": "abt", "models": "models"},
    }
}


def test_path_monta_uri_s3():
    st = get_storage(CFG)
    assert st.path("raw", "application_train.csv") == "s3://raw/application_train.csv"
    assert st.path("clean", "bureau.parquet") == "s3://clean/bureau.parquet"
    # subpasta na key (artefatos do modelo)
    assert st.path("models", "modelo_risco_credito/m.pkl") == "s3://models/modelo_risco_credito/m.pkl"


def test_storage_options_vem_de_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    st = get_storage(CFG)
    opts = st.io_kwargs()["storage_options"]
    assert opts["key"] == "k"
    assert opts["secret"] == "s"
    assert opts["client_kwargs"]["endpoint_url"] == "http://minio:9000"
