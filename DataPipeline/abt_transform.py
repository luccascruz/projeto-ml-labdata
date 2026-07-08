"""Construção da ABT (clean -> abt).

Lê as bases sanitizadas do bucket `clean` (MinIO/S3), agrega o histórico
(bureau e previous_application) para uma linha por cliente e junta tudo na
tabela principal. Nomes de coluna, agregações, renomeações e buckets vêm do
config.yml — nada chumbado. Saída: `abt.csv` no bucket `abt`.
"""

import sys
import pandas as pd
from pathlib import Path
import yaml

# Raiz do projeto = pasta-pai de DataPipeline/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"

# storage.py fica na raiz do projeto: garante o import ao rodar como script
sys.path.insert(0, str(PROJECT_ROOT))
from storage import get_storage


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_abt(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    store = get_storage(cfg)
    kw = store.io_kwargs()

    clean_files = cfg["data"]["clean_files"]
    id_col = cfg["project"]["id_column"]

    print("--- Construindo a ABT Final ---")

    # 1. Carregar bases sanitizadas (parquet) do bucket `clean`
    app = pd.read_parquet(store.path("clean", clean_files["application"]), **kw)
    bureau = pd.read_parquet(store.path("clean", clean_files["bureau"]), **kw)
    prev_app = pd.read_parquet(store.path("clean", clean_files["previous_application"]), **kw)

    # 2. Agregações (config-driven), uma linha por id_col
    aggs = cfg["abt"]["aggregations"]
    rename = cfg["abt"]["rename"]
    bureau_agg = bureau.groupby(id_col).agg(aggs["bureau"]).rename(columns=rename)
    prev_agg = prev_app.groupby(id_col).agg(aggs["previous_application"]).rename(columns=rename)

    # 3. Merge na tabela principal
    print("Realizando os merges...")
    abt = app.merge(bureau_agg, on=id_col, how="left")
    abt = abt.merge(prev_agg, on=id_col, how="left")

    # 4. Clientes sem histórico recebem o valor configurado
    abt = abt.fillna(cfg["abt"]["fill_missing_after_merge"])

    # 5. Salvar ABT no bucket `abt`
    output_abt = store.path("abt", cfg["data"]["abt_file"])
    abt.to_csv(output_abt, index=False, **kw)
    print(f"ABT Final construída com sucesso! Shape: {abt.shape}")
    print(f"Salva em: {output_abt}")
    return abt


if __name__ == "__main__":
    build_abt()
