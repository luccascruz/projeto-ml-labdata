"""Sanitização dos dados brutos (raw -> clean).

Todos os parâmetros (caminhos, nomes de arquivo, estratégias de imputação,
limites de nulos, valor alvo) vêm do config.yml — nada é chumbado aqui.
Os caminhos são resolvidos relativos à raiz do projeto via pathlib, de forma
portável entre Linux, macOS e Windows. Saídas gravadas direto em /Dados (flat),
seguindo a estrutura de entrega.
"""

import pandas as pd
from pathlib import Path
import yaml

# Raiz do projeto = pasta-pai de DataPipeline/ (este arquivo vive em DataPipeline/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Carrega o config.yml como dicionário."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fill_numeric(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Preenche nulos das colunas numéricas conforme a estratégia configurada."""
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        else:  # "zero"
            df[col] = df[col].fillna(0)
    return df


def sanitize_application(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Limpeza da tabela principal (application_train)."""
    san = cfg["sanitization"]

    # Remoção de colunas acima do limite de nulos configurado
    limite = san["null_threshold_pct"]
    percentual_nulos = (df.isnull().sum() / len(df)) * 100
    df = df.drop(columns=percentual_nulos[percentual_nulos > limite].index.tolist())

    # Numéricas pela estratégia configurada; categóricas pelo valor configurado
    df = _fill_numeric(df, san["numeric_fill_strategy"])
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(san["categorical_fill_value"])

    # Feature engineering: idade a partir de DAYS_BIRTH (se habilitado)
    if san.get("feature_engineering", {}).get("days_birth_to_age") and "DAYS_BIRTH" in df.columns:
        df["IDADE_ANOS"] = df["DAYS_BIRTH"].abs() / 365
        df = df.drop(columns=["DAYS_BIRTH"])

    return df


def sanitize_bureau(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Limpeza do bureau: colunas monetárias preenchidas com 0."""
    keyword = cfg["sanitization"]["monetary_keyword"]
    cols_monetarias = [col for col in df.columns if keyword in col]
    df[cols_monetarias] = df[cols_monetarias].fillna(0)
    return df


def sanitize_previous_app(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Limpeza do previous_application: numéricas com 0, categóricas com o valor configurado."""
    fill_val = cfg["sanitization"]["categorical_fill_value"]
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(fill_val)
    return df


def run_sanitization(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    clean_dir = PROJECT_ROOT / cfg["paths"]["clean_dir"]

    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_files = cfg["data"]["raw_files"]
    clean_files = cfg["data"]["clean_files"]

    # Mapeamento: (arquivo bruto, arquivo limpo, função de limpeza)
    tasks = [
        (raw_files["application"], clean_files["application"], sanitize_application),
        (raw_files["bureau"], clean_files["bureau"], sanitize_bureau),
        (raw_files["previous_application"], clean_files["previous_application"], sanitize_previous_app),
    ]

    for raw_name, clean_name, clean_func in tasks:
        input_path = raw_dir / raw_name
        if input_path.exists():
            print(f"--- Sanitizando: {raw_name} ---")
            df = pd.read_csv(input_path)
            df_clean = clean_func(df, cfg)
            output_path = clean_dir / clean_name
            df_clean.to_csv(output_path, index=False)
            print(f"Salvo em: {output_path}")
        else:
            print(f"Arquivo não encontrado: {input_path}")


if __name__ == "__main__":
    run_sanitization()
