"""Construção da ABT (clean -> abt)."""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from storage import get_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def drop_rare_rows(df: pd.DataFrame, drop_rules: dict) -> pd.DataFrame:
    """Remove linhas com categorias raras/inválidas (config: abt.drop_rows)."""
    for col, values in (drop_rules or {}).items():
        if col in df.columns:
            before = len(df)
            df = df[~df[col].isin(values)]
            removed = before - len(df)
            if removed:
                print(f"  Removidas {removed} linhas com {col} em {values}")
    return df.reset_index(drop=True)


def build_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Cria features a partir da tabela application_train."""
    app = app.copy()
    app["AGE"] = abs(app["DAYS_BIRTH"]) / 365.25

    # Flag de ausência de histórico de emprego (desempregados/pensionistas
    # ficam com DAYS_EMPLOYED nulo desde a sanitização) + imputação por mediana,
    # em vez de deixar EMPLOYMENT_YEARS como NaN.
    app["FLAG_SEM_HISTORICO_EMPREGO"] = app["DAYS_EMPLOYED"].isna().astype("int8")
    employment_years = abs(app["DAYS_EMPLOYED"]) / 365.25
    app["EMPLOYMENT_YEARS"] = employment_years.fillna(
        employment_years.median())

    app["CREDIT_INCOME_RATIO"] = app["AMT_CREDIT"] / \
        app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"] / \
        app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["ANNUITY_CREDIT_RATIO"] = app["AMT_ANNUITY"] / \
        app["AMT_CREDIT"].replace(0, np.nan)
    app["GOODS_CREDIT_RATIO"] = app["AMT_GOODS_PRICE"] / \
        app["AMT_CREDIT"].replace(0, np.nan)

    ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    app["EXT_SOURCE_MEAN"] = app[ext_sources].mean(axis=1)
    app["EXT_SOURCE_STD"] = app[ext_sources].std(axis=1)
    app["EXT_SOURCE_MIN"] = app[ext_sources].min(axis=1)
    app["EXT_SOURCE_MAX"] = app[ext_sources].max(axis=1)
    app["EXT_SOURCE_MISSING"] = app[ext_sources].isna().sum(axis=1)

    docs = [col for col in app.columns if col.startswith("FLAG_DOCUMENT_")]
    app["DOCUMENT_COUNT"] = app[docs].sum(axis=1)

    return app.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL", *docs], errors="ignore")


def build_previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Cria features agregadas da tabela previous_application."""
    prev = prev.copy()
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / \
        prev["AMT_CREDIT"].replace(0, np.nan)

    return prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_RATE=("NAME_CONTRACT_STATUS",
                            lambda x: (x == "Approved").mean()),
        PREV_REFUSED_RATE=("NAME_CONTRACT_STATUS",
                           lambda x: (x == "Refused").mean()),
        PREV_MEAN_CREDIT=("AMT_CREDIT", "mean"),
        PREV_MAX_CREDIT=("AMT_CREDIT", "max"),
        PREV_MEAN_CNT_PAYMENT=("CNT_PAYMENT", "mean"),
        PREV_MEAN_APP_CREDIT_RATIO=("APP_CREDIT_RATIO", "mean"),
        PREV_LAST_APPLICATION=("DAYS_DECISION", "max")
    ).reset_index()


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Cria features agregadas da tabela bureau com granularidade."""
    bureau = bureau.copy()

    # Agregação completa de estatísticas do bureau
    bureau_features = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
            TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
            TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
            MEAN_DEBT=("AMT_CREDIT_SUM_DEBT", "mean"),
            MAX_DEBT=("AMT_CREDIT_SUM_DEBT", "max"),
            TOTAL_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            MEAN_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "mean"),
            MEAN_DAYS_CREDIT=("DAYS_CREDIT", "mean"),
            LAST_CREDIT=("DAYS_CREDIT", "max"),
            CREDIT_TYPES=("CREDIT_TYPE", "nunique"),
        )
    )

    # Cálculo da taxa de empréstimos ativos
    active_rate = (
        bureau
        .assign(ACTIVE=bureau["CREDIT_ACTIVE"].eq("Active"))
        .groupby("SK_ID_CURR")["ACTIVE"]
        .mean()
        .rename("ACTIVE_LOAN_RATE")
    )

    # Une as agregações com a taxa de ativos e reseta o índice
    return bureau_features.join(active_rate).reset_index()


def build_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features utilizando informações de múltiplas tabelas.

    Pré-requisito: TOTAL_DEBT, BUREAU_LOAN_COUNT, EXT_SOURCE_2 e EXT_SOURCE_3
    já devem estar com os NaNs de "sem histórico" tratados antes de chamar esta
    função — caso contrário DTI_RATIO e AVG_DEBT_PER_LOAN saem inconsistentes
    (ver explicação no chat).
    """
    df = df.copy()
    df["DTI_RATIO"] = df["TOTAL_DEBT"] / \
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["AVG_DEBT_PER_LOAN"] = df["TOTAL_DEBT"] / \
        df["BUREAU_LOAN_COUNT"].replace(0, np.nan)

    # Interações cruzadas de fontes externas
    df["INCOME_x_EXT_SOURCE_2"] = df["AMT_INCOME_TOTAL"] * df["EXT_SOURCE_2"]
    df["INCOME_x_EXT_SOURCE_3"] = df["AMT_INCOME_TOTAL"] * df["EXT_SOURCE_3"]

    return df


def prepare_model_data(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    test_size: float = 0.20,
    val_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza o encoding das variáveis categóricas e divide
    os dados em treino, validação e teste (holdout).
    """

    features = df.drop(
        columns=[
            target,
            "SK_ID_CURR",
        ],
        errors="ignore",
    )

    features = pd.get_dummies(
        features,
        dummy_na=True,
    )

    features[target] = df[target]

    # 1ª divisão: separa o TEST (holdout) do resto. Só deve ser tocado
    # uma vez, no final, para avaliar o modelo já escolhido.
    train_val, test = train_test_split(
        features,
        test_size=test_size,
        stratify=features[target],
        random_state=random_state,
    )

    # 2ª divisão: separa TREINO de VALIDAÇÃO dentro do que sobrou.
    # val_size é fração do dataset ORIGINAL, então reconvertemos para
    # fração do que sobrou depois de tirar o test.
    val_relative_size = val_size / (1 - test_size)

    train, validation = train_test_split(
        train_val,
        test_size=val_relative_size,
        stratify=train_val[target],
        random_state=random_state,
    )

    # Imputação por mediana — calculada SOMENTE no treino e aplicada em
    # ambos os splits. Isso resolve dois problemas do fillna(0) anterior:
    # (1) mediana preserva melhor a distribuição de colunas como EXT_SOURCE
    # e as razões financeiras do que forçar tudo pra 0; (2) calcular a
    # mediana depois do split (e só com o treino) evita vazamento de
    # informação da validação para dentro do treino.
    numeric_cols = train.drop(columns=[target]).select_dtypes(
        include="number").columns
    medianas_treino = train[numeric_cols].median()

    for split in (train, validation, test):
        split[numeric_cols] = split[numeric_cols].fillna(medianas_treino)
        # Fallback: se alguma coluna do treino for 100% nula, a mediana
        # também sai NaN — preenche com 0 pra garantir que nenhum NaN
        # sobrevive antes do .fit() do modelo.
        split[numeric_cols] = split[numeric_cols].fillna(0)

    return train, validation, test


def build_abt() -> None:
    """Constrói a Analytical Base Table (ABT) e salva no MinIO."""
    cfg = load_config()
    store = get_storage(cfg)
    kw = store.io_kwargs()

    print("--- Construindo ABT no MinIO ---")
    app = pd.read_parquet(store.path(
        "clean", cfg["data"]["clean_files"]["application"]), **kw)
    prev = pd.read_parquet(store.path(
        "clean", cfg["data"]["clean_files"]["previous_application"]), **kw)
    bur = pd.read_parquet(store.path(
        "clean", cfg["data"]["clean_files"]["bureau"]), **kw)

    app = build_application_features(drop_rare_rows(
        app, cfg.get("abt", {}).get("drop_rows")))

    df = app.merge(build_previous_features(prev), on="SK_ID_CURR", how="left") \
            .merge(build_bureau_features(bur), on="SK_ID_CURR", how="left")

    df["FLAG_SEM_HISTORICO_PREVIA"] = df["PREV_COUNT"].isna().astype("int8")
    df["FLAG_SEM_HISTORICO_BUREAU"] = df["BUREAU_LOAN_COUNT"].isna().astype("int8")

    # Preenchimento automático de todas as colunas de agregação
    agg_cols = [c for c in df.columns if c.startswith(
        ("PREV_", "BUREAU_", "TOTAL_"))]
    df[agg_cols] = df[agg_cols].fillna(0)

    df = build_final_features(df).replace([np.inf, -np.inf], np.nan)

    train, val, test = prepare_model_data(
        df, cfg["project"]["target"], cfg["project"]["random_state"])

    for df_name, df_obj in [("train.parquet", train), ("val.parquet", val), ("test.parquet", test)]:
        df_obj.to_parquet(store.path("abt", df_name), index=False, **kw)

    print("ABT Finalizada e salva no MinIO.")


if __name__ == "__main__":
    build_abt()