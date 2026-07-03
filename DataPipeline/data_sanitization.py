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


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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


def sanitize_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitiza a tabela previous_application.

    Operações realizadas:
        - Substitui o valor sentinela (365243) por NA nas colunas de dias
        - Substitui valores monetários negativos por NA
        - Substitui quantidade negativa de parcelas por NA
    """

    # Valor sentinela 365243 utilizado para representar ausência de informação
    day_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]

    for col in day_cols:
        if col in df.columns:
            df[col] = df[col].replace(365243, pd.NA)

    # Valores monetários não podem ser negativos
    monetary_cols = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
    ]

    for col in monetary_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # Número de parcelas não pode ser negativo
    if "CNT_PAYMENT" in df.columns:
        df.loc[df["CNT_PAYMENT"] < 0, "CNT_PAYMENT"] = pd.NA

    return df


def run_sanitization(cfg: dict | None = None) -> None:
    """
    Executa a sanitização de todas as tabelas brutas.

    Pipeline:
        1. Carrega o arquivo bruto
        2. Executa limpeza genérica
        3. Executa sanitização específica da tabela
        4. Otimiza tipos de dados
        5. Salva o resultado
    """

    cfg = cfg or load_config()

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    clean_dir = PROJECT_ROOT / cfg["paths"]["clean_dir"]

    clean_dir.mkdir(parents=True, exist_ok=True)

    raw_files = cfg["data"]["raw_files"]
    clean_files = cfg["data"]["clean_files"]

    tasks = {
        "application": sanitize_application,
        "bureau": sanitize_bureau,
        "previous_application": sanitize_previous_application,
    }

    for dataset, sanitize_func in tasks.items():

        input_path = raw_dir / raw_files[dataset]
        output_path = clean_dir / clean_files[dataset]

        if not input_path.exists():
            print(f"Arquivo não encontrado: {input_path}")
            continue

        print(f"\n--- Sanitizando {dataset} ---")

        df = pd.read_csv(input_path)

        # Limpeza genérica
        df = clean_dataframe(df)

        # Regras específicas da tabela
        df = sanitize_func(df)

        # Otimização de memória
        df = optimize_dtypes(df)

        df.to_csv(output_path, index=False)

        print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    run_sanitization()
