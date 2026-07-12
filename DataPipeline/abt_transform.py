from storage import get_storage
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

# Configuração de caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --- FUNÇÕES DE FEATURE ENGINEERING ---
def drop_rare_rows(df: pd.DataFrame, drop_rules: dict) -> pd.DataFrame:
    for col, values in (drop_rules or {}).items():
        if col in df.columns:
            df = df[~df[col].isin(values)]
    return df.reset_index(drop=True)


def build_application_features(app: pd.DataFrame) -> pd.DataFrame:
    app = app.copy()
    app["AGE"] = abs(app["DAYS_BIRTH"]) / 365.25
    app["FLAG_SEM_HISTORICO_EMPREGO"] = app["DAYS_EMPLOYED"].isna().astype("int8")
    employment_years = abs(app["DAYS_EMPLOYED"]) / 365.25
    app["EMPLOYMENT_YEARS"] = employment_years.fillna(
        employment_years.median())

    # Razões Financeiras
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
    bureau = bureau.copy()
    bureau_features = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        TOTAL_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum")
    )
    active_rate = bureau.assign(ACTIVE=bureau["CREDIT_ACTIVE"].eq(
        "Active")).groupby("SK_ID_CURR")["ACTIVE"].mean()
    return bureau_features.join(active_rate).reset_index()


def build_final_features(df: pd.DataFrame) -> pd.DataFrame:
    df["DTI_RATIO"] = df["TOTAL_DEBT"] / \
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["AVG_DEBT_PER_LOAN"] = df["TOTAL_DEBT"] / \
        df["BUREAU_LOAN_COUNT"].replace(0, np.nan)
    return df


def prepare_model_data(df, target, random_state):
    features = pd.get_dummies(
        df.drop(columns=["SK_ID_CURR"], errors="ignore"), dummy_na=True)
    features[target] = df[target]
    train_val, test = train_test_split(
        features, test_size=0.2, stratify=features[target], random_state=random_state)
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val[target], random_state=random_state)

    numeric_cols = train.drop(columns=[target]).select_dtypes(
        include="number").columns
    medianas = train[numeric_cols].median()
    for split in (train, val, test):
        split[numeric_cols] = split[numeric_cols].fillna(medianas).fillna(0)
    return train, val, test


# --- FLUXO PRINCIPAL ---
def build_abt() -> None:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "DataPipeline" / "config.yml"))
    store = get_storage(cfg)
    kw = store.io_kwargs()

    print("--- Construindo ABT no MinIO ---")

    # 1. Leitura
    app = pd.read_parquet(store.path(
        "clean", cfg["data"]["clean_files"]["application"]), **kw)
    prev = pd.read_parquet(store.path(
        "clean", cfg["data"]["clean_files"]["previous_application"]), **kw)
    bur = pd.read_parquet(store.path(
        "clean", cfg["data"]["clean_files"]["bureau"]), **kw)

    # 2. Transformação
    app = build_application_features(drop_rare_rows(
        app, cfg.get("abt", {}).get("drop_rows")))
    df = app.merge(build_previous_features(prev), on="SK_ID_CURR", how="left") \
            .merge(build_bureau_features(bur), on="SK_ID_CURR", how="left")

    # Flags de histórico e preenchimento
    df["FLAG_SEM_HISTORICO_PREVIA"] = df["PREV_COUNT"].isna().astype("int8")
    df["FLAG_SEM_HISTORICO_BUREAU"] = df["BUREAU_LOAN_COUNT"].isna().astype("int8")

    cols_to_fill = ["PREV_COUNT", "BUREAU_LOAN_COUNT", "TOTAL_DEBT"]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    df = build_final_features(df).replace([np.inf, -np.inf], np.nan)

    # 3. Model Data & Escrita
    train, val, test = prepare_model_data(
        df, cfg["project"]["target"], cfg["project"]["random_state"])

    for df_name, df_obj in [("train.parquet", train), ("val.parquet", val), ("test.parquet", test)]:
        df_obj.to_parquet(store.path("abt", df_name), index=False, **kw)

    print("ABT Finalizada e salva no MinIO.")


if __name__ == "__main__":
    build_abt()
