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


def optimize_dtypes(df:pd.DataFrame) -> pd.DataFrame:
    """
    Otimiza os tipos de dados para reduzir o consumo de memória.

    Operações realizadas:
        - float64 -> float32
        - int64 -> menor tipo inteiro possível (downcast)
    """

    # Converte colunas float64 para float32
    colunas_float = df.select_dtypes(include="float64").columns

    if len(colunas_float) > 0:
        df[colunas_float] = df[colunas_float].apply(
            pd.to_numeric,
            downcast="float"
        )

    # Converte colunas inteiras para o menor tipo possível
    colunas_int = df.select_dtypes(include="int64").columns

    for coluna in colunas_int:
        df[coluna] = pd.to_numeric(
            df[coluna],
            downcast="integer"
        )

    return df


def clean_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """
    Executa operações genéricas de limpeza aplicáveis a qualquer DataFrame.

    Operações realizadas:
        - Remove linhas duplicadas
        - Remove colunas duplicadas
        - Remove espaços em branco no início e fim de colunas de texto
    """

    # Remove linhas completamente duplicadas
    df = df.drop_duplicates()

    # Remove espaços em branco das colunas de texto
    colunas_texto = df.select_dtypes(include="object").columns

    for coluna in colunas_texto:
        df[coluna] = df[coluna].str.strip()

    # Remove colunas duplicadas, caso existam
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def sanitize_application(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza da tabela principal (application_train)"""
    # Corrige valor sentinela utilizado para representar ausência de informação
    df["DAYS_EMPLOYED"] = (
        df["DAYS_EMPLOYED"]
        .replace(365243, pd.NA)
    )

   # Valida a coluna alvo
    valores = set(df["TARGET"].dropna())

    if not valores.issubset({0, 1}):
        raise ValueError("TARGET contém valores inválidos.")

    return df


def sanitize_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitiza a tabela bureau.

    Operações realizadas:
        - Substitui valores monetários negativos por NA
        - Substitui contagens negativas de prorrogação por NA
        - Substitui datas futuras (dias positivos) por NA
    """

    # Valores monetários não podem ser negativos
    monetary_cols = [
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
    ]

    for col in monetary_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # Quantidade de prorrogações não pode ser negativa
    if "CNT_CREDIT_PROLONG" in df.columns:
        df.loc[df["CNT_CREDIT_PROLONG"] < 0, "CNT_CREDIT_PROLONG"] = pd.NA

    # Datas em bureau representam dias relativos ao momento da aplicação.
    # Valores positivos indicariam eventos futuros.
    day_cols = [
        "DAYS_CREDIT",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
    ]

    for col in day_cols:
        if col in df.columns:
            df.loc[df[col] > 0, col] = pd.NA

    return df

def sanitize_previous_app(df: pd.DataFrame) -> pd.DataFrame:
    
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
