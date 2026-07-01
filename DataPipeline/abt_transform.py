"""Construção da ABT (clean -> abt).

Lê as bases sanitizadas, agrega o histórico (bureau e previous_application)
para uma linha por cliente e junta tudo na tabela principal. Todos os caminhos,
nomes de coluna, agregações e renomeações vêm do config.yml — nada chumbado.
Caminhos resolvidos relativos à raiz do projeto (portável). Saída: Dados/abt.csv.
"""

import pandas as pd
from pathlib import Path
import yaml

# Raiz do projeto = pasta-pai de DataPipeline/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_abt(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()

    clean_dir = PROJECT_ROOT / cfg["paths"]["clean_dir"]
    abt_dir = PROJECT_ROOT / cfg["paths"]["abt_dir"]

    abt_dir.mkdir(parents=True, exist_ok=True)

    clean_files = cfg["data"]["clean_files"]
    id_col = cfg["project"]["id_column"]

    print("--- Construindo a ABT Final ---")

    # 1. Carregar bases sanitizadas
    app = pd.read_csv(clean_dir / clean_files["application"])
    bureau = pd.read_csv(clean_dir / clean_files["bureau"])
    prev_app = pd.read_csv(clean_dir / clean_files["previous_application"])

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

    # 5. Salvar ABT
    output_abt = abt_dir / cfg["data"]["abt_file"]
    abt.to_csv(output_abt, index=False)
    print(f"ABT Final construída com sucesso! Shape: {abt.shape}")
    print(f"Salva em: {output_abt}")
    return abt


if __name__ == "__main__":
    build_abt()
