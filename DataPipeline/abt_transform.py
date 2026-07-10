"""Construção da ABT (clean -> abt)."""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

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
    app["EMPLOYMENT_YEARS"] = employment_years.fillna(employment_years.median())

    app["CREDIT_INCOME_RATIO"] = (
        app["AMT_CREDIT"] /
        app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    app["ANNUITY_INCOME_RATIO"] = (
        app["AMT_ANNUITY"] /
        app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    app["ANNUITY_CREDIT_RATIO"] = (
        app["AMT_ANNUITY"] /
        app["AMT_CREDIT"].replace(0, np.nan)
    )

    app["GOODS_CREDIT_RATIO"] = (
        app["AMT_GOODS_PRICE"] /
        app["AMT_CREDIT"].replace(0, np.nan)
    )

    ext_sources = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    app["EXT_SOURCE_MEAN"] = app[ext_sources].mean(axis=1)
    app["EXT_SOURCE_STD"] = app[ext_sources].std(axis=1)
    app["EXT_SOURCE_MIN"] = app[ext_sources].min(axis=1)
    app["EXT_SOURCE_MAX"] = app[ext_sources].max(axis=1)
    app["EXT_SOURCE_MISSING"] = (
    app[
        ["EXT_SOURCE_1",
         "EXT_SOURCE_2",
         "EXT_SOURCE_3"]
    ]
    .isna()
    .sum(axis=1)
    )

    docs = [col for col in app.columns if col.startswith("FLAG_DOCUMENT_")]

    app["DOCUMENT_COUNT"] = app[docs].sum(axis=1)

    app = app.drop(
        columns=[
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "FLAG_MOBIL",
            *docs,
        ],
        errors="ignore",
    )

    return app


def build_previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Cria features agregadas da tabela previous_application."""

    prev = prev.copy()

    prev["APP_CREDIT_RATIO"] = (
        prev["AMT_APPLICATION"] /
        prev["AMT_CREDIT"].replace(0, np.nan)
    )

    previous_features = (
        prev
        .groupby("SK_ID_CURR")
        .agg(
            PREV_COUNT=("SK_ID_PREV", "count"),
            PREV_APPROVED_RATE=(
                "NAME_CONTRACT_STATUS",
                lambda x: (x == "Approved").mean(),
            ),
            PREV_REFUSED_RATE=(
                "NAME_CONTRACT_STATUS",
                lambda x: (x == "Refused").mean(),
            ),
            PREV_MEAN_CREDIT=("AMT_CREDIT", "mean"),
            PREV_MAX_CREDIT=("AMT_CREDIT", "max"),
            PREV_MEAN_APPLICATION=("AMT_APPLICATION", "mean"),
            PREV_MEAN_CNT_PAYMENT=("CNT_PAYMENT", "mean"),
            PREV_MAX_CNT_PAYMENT=("CNT_PAYMENT", "max"),
            PREV_MEAN_DOWN_PAYMENT=("RATE_DOWN_PAYMENT", "mean"),
            PREV_MEAN_APP_CREDIT_RATIO=("APP_CREDIT_RATIO", "mean"),
            PREV_LAST_APPLICATION=("DAYS_DECISION", "max"),
            PREV_MEAN_DECISION=("DAYS_DECISION", "mean"),
        )
        .reset_index()
    )

    return previous_features


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Cria features agregadas da tabela bureau."""

    bureau = bureau.copy()

    # Flags auxiliares
    bureau["IS_ACTIVE"] = bureau["CREDIT_ACTIVE"].eq("Active").astype("int8")
    bureau["IS_CLOSED"] = bureau["CREDIT_ACTIVE"].eq("Closed").astype("int8")
    bureau["HAS_OVERDUE"] = (
        bureau["AMT_CREDIT_SUM_OVERDUE"]
        .fillna(0)
        .gt(0)
        .astype("int8")
    )

    # Dívida apenas de contratos ativos
    bureau["ACTIVE_DEBT_VALUE"] = (
        bureau["AMT_CREDIT_SUM_DEBT"]
        .where(bureau["IS_ACTIVE"] == 1, 0)
    )

    bureau_features = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),

            TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),

            TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
            MEAN_DEBT=("AMT_CREDIT_SUM_DEBT", "mean"),
            MAX_DEBT=("AMT_CREDIT_SUM_DEBT", "max"),

            ACTIVE_DEBT=("ACTIVE_DEBT_VALUE", "sum"),

            TOTAL_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            MEAN_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "mean"),

            MEAN_DAYS_CREDIT=("DAYS_CREDIT", "mean"),
            LAST_CREDIT=("DAYS_CREDIT", "max"),

            CREDIT_TYPES=("CREDIT_TYPE", "nunique"),

            ACTIVE_LOANS=("IS_ACTIVE", "sum"),
            CLOSED_LOANS=("IS_CLOSED", "sum"),

            ACTIVE_LOAN_RATE=("IS_ACTIVE", "mean"),
            OVERDUE_RATE=("HAS_OVERDUE", "mean"),
        )
        .reset_index()
    )


    return bureau_features


def build_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features utilizando informações de múltiplas tabelas.

    Pré-requisito: TOTAL_DEBT, BUREAU_LOAN_COUNT, EXT_SOURCE_2 e EXT_SOURCE_3
    já devem estar com os NaNs de "sem histórico" tratados antes de chamar esta
    função — caso contrário DTI_RATIO e AVG_DEBT_PER_LOAN saem inconsistentes
    (ver explicação no chat).
    """

    df = df.copy()

    df["DTI_RATIO"] = (
        df["TOTAL_DEBT"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["AVG_DEBT_PER_LOAN"] = (
        df["TOTAL_DEBT"] /
        df["BUREAU_LOAN_COUNT"].replace(0, np.nan)
    )

    df["INCOME_x_EXT_SOURCE_2"] = (
        df["AMT_INCOME_TOTAL"] *
        df["EXT_SOURCE_2"]
    )

    df["INCOME_x_EXT_SOURCE_3"] = (
        df["AMT_INCOME_TOTAL"] *
        df["EXT_SOURCE_3"]
    )

    return df


def prepare_model_data(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza o encoding das variáveis categóricas e divide
    os dados em treino e validação.
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

    train, validation = train_test_split(
        features,
        test_size=test_size,
        stratify=features[target],
        random_state=random_state,
    )

    # Imputação por mediana — calculada SOMENTE no treino e aplicada em
    # ambos os splits. Isso resolve dois problemas do fillna(0) anterior:
    # (1) mediana preserva melhor a distribuição de colunas como EXT_SOURCE
    # e as razões financeiras do que forçar tudo pra 0; (2) calcular a
    # mediana depois do split (e só com o treino) evita vazamento de
    # informação da validação para dentro do treino.
    numeric_cols = train.drop(columns=[target]).select_dtypes(include="number").columns
    medianas_treino = train[numeric_cols].median()

    train[numeric_cols] = train[numeric_cols].fillna(medianas_treino)
    validation[numeric_cols] = validation[numeric_cols].fillna(medianas_treino)

    # Remove features com somente 1 valor única
    X_train = train.drop(columns=[target])

    constant_cols = X_train.columns[X_train.nunique() <= 1]

    train = train.drop(columns=constant_cols)
    validation = validation.drop(columns=constant_cols)

    # Fallback: se alguma coluna do treino for 100% nula, a mediana também
    # sai NaN — nesse caso extremo, preenche com 0 pra garantir que nenhum
    # NaN sobrevive antes do .fit() do modelo.
    train[numeric_cols] = train[numeric_cols].fillna(0)
    validation[numeric_cols] = validation[numeric_cols].fillna(0)

    return train, validation


def build_abt(cfg: dict | None = None) -> None:
    """
    Constrói a Analytical Base Table (ABT).
    """

    cfg = cfg or load_config()

    clean_dir = PROJECT_ROOT / cfg["paths"]["clean_dir"]
    abt_dir = PROJECT_ROOT / cfg["paths"]["abt_dir"]
    abt_dir.mkdir(parents=True, exist_ok=True)

    clean_files = cfg["data"]["clean_files"]
    abt_files = cfg["data"].get("abt_files")
    target = cfg["project"]["target"]
    random_state = cfg["project"]["random_state"]
    test_size = cfg.get("abt", {}).get("test_size", 0.20)

    print("Carregando tabelas...")

    application = pd.read_parquet(clean_dir / clean_files["application"])
    previous = pd.read_parquet(clean_dir / clean_files["previous_application"])
    bureau = pd.read_parquet(clean_dir / clean_files["bureau"])

    print("Removendo linhas raras/inválidas...")
    application = drop_rare_rows(application, cfg.get("abt", {}).get("drop_rows"))

    print("Criando features da application...")
    application = build_application_features(application)

    print("Criando features da previous_application...")
    previous_features = build_previous_features(previous)

    print("Criando features da bureau...")
    bureau_features = build_bureau_features(bureau)

    print("Realizando merges...")

    df = (
        application
        .merge(previous_features, on="SK_ID_CURR", how="left")
        .merge(bureau_features, on="SK_ID_CURR", how="left")
    )

    # Flags de "sem histórico" — capturadas ANTES do fillna dos agregados,
    # pois depois do fillna(0) não dá mais para distinguir "0 empréstimos
    # prévios" de "não tinha registro nenhum" (não é bem a mesma coisa, mas
    # aqui coincide: quem não aparece no merge tem count NaN -> 0 depois).
    df["FLAG_SEM_HISTORICO_PREVIA"] = df["PREV_COUNT"].isna().astype("int8")
    df["FLAG_SEM_HISTORICO_BUREAU"] = df["BUREAU_LOAN_COUNT"].isna().astype("int8")

    # Preenche TODAS as colunas agregadas de previous_application e bureau,
    # não só uma lista parcial — senão colunas como PREV_MEAN_CREDIT,
    # MEAN_DEBT, LAST_CREDIT etc. ficam com NaN pros clientes sem histórico.
    aggregated_cols = [
        c for c in previous_features.columns if c != "SK_ID_CURR"
    ] + [
        c for c in bureau_features.columns if c != "SK_ID_CURR"
    ]
    df[aggregated_cols] = df[aggregated_cols].fillna(0)

    print("Criando features finais...")

    # Só agora, com os agregados já preenchidos, é seguro calcular as
    # features cruzadas (senão DTI_RATIO e AVG_DEBT_PER_LOAN saem NaN mesmo
    # quando TOTAL_DEBT e BUREAU_LOAN_COUNT já deveriam ser 0).
    df = build_final_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    train, validation = prepare_model_data(
        df=df,
        target=target,
        random_state=random_state,
        test_size=test_size,
    )

    print("Salvando ABTs...")

    train_ref = abt_dir / abt_files["train"]
    val_ref = abt_dir / abt_files["val"]

    train.to_parquet(train_ref, index=False)
    validation.to_parquet(val_ref, index=False)

    print("-" * 50)
    print("ABT criada com sucesso.")
    print(f"Treino: {train.shape}")
    print(f"Validação: {validation.shape}")
    print(f"Salvo em: {train_ref} e {val_ref}")


if __name__ == "__main__":
    build_abt()