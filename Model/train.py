"""Treino do modelo de risco de crédito (abt -> modelo).

Lê o config próprio de /Model (Model/config.yml): hiperparâmetros, seleção de
variáveis, estratégia de balanceamento, threshold e buckets — nada chumbado.
Lê a ABT do bucket `abt` (MinIO/S3) e salva `.pkl`/artefatos no bucket `models`
via storage.py. Métrica oficial reportada: ROC AUC.
"""

import sys
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem display: funciona em servidor/headless (Docker)
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# Raiz do projeto = pasta-pai de Model/ (este arquivo vive em Model/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "Model" / "config.yml"

# storage.py fica na raiz do projeto: garante o import ao rodar como script
sys.path.insert(0, str(PROJECT_ROOT))
from storage import get_storage


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()

    target = cfg["project"]["target"]
    seed = cfg["project"]["random_state"]
    mcfg = cfg["model"]

    store = get_storage(cfg)
    kw = store.io_kwargs()

    print("--- Iniciando Pipeline com ABT Finalizada ---")
    df = pd.read_csv(store.path("abt", cfg["data"]["abt_file"]), **kw)

    # Artefatos do modelo vão para o bucket `models`, sob a subpasta do modelo
    model_name = mcfg["name"]
    prefix = f"{model_name}/"

    # 1. Mantém apenas colunas numéricas + o target
    y = df[target]
    df = df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    df = df.fillna(0)
    df[target] = y

    # 2. Feature engineering (configurável)
    if mcfg["feature_engineering"].get("renda_por_familia") \
            and "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
        df["RENDA_POR_FAMILIA"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=mcfg["test_size"], random_state=seed
    )
    # 3. Seleção de variáveis por importância (configurável)
    fs = mcfg["feature_selection"]
    if fs["enabled"]:
        print("Rankeando variáveis...")
        ranker = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=mcfg["hyperparameters"]["n_jobs"]
        )
        ranker.fit(X_train, y_train)
        importances = pd.Series(ranker.feature_importances_, index=X_train.columns)
        top_features = importances.nlargest(fs["top_n"]).index.tolist()
        X_train = X_train[top_features]
        X_test = X_test[top_features]
        print(f"Features selecionadas para treino: {top_features}")
    else:
        top_features = X_train.columns.tolist()

    # 3.1 Salva conjunto de teste (X_test, y_test) no bucket `models` p/ avaliação futura
    X_test.to_csv(store.path("models", prefix + "X_test.csv"), index=False, **kw)
    y_test.to_csv(store.path("models", prefix + "y_test.csv"), index=False, **kw)


    # 4. Balanceamento (configurável: smote | class_weight | none)
    balancing = mcfg["balancing"]
    if balancing.get("method") == "smote":
        X_train, y_train = SMOTE(random_state=seed).fit_resample(X_train, y_train)

    class_weight = balancing.get("class_weight")
    if class_weight is not None:
        # YAML pode carregar as chaves como str; garante int (rótulos das classes)
        class_weight = {int(k): v for k, v in class_weight.items()}

    hp = mcfg["hyperparameters"]
    model = RandomForestClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        n_jobs=hp["n_jobs"],
        class_weight=class_weight,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # 5. Avaliação no conjunto de TESTE (held-out) — métrica oficial: ROC AUC
    probs = model.predict_proba(X_test)[:, 1]
    threshold = mcfg["threshold"]
    y_pred = (probs >= threshold).astype(int)

    print("\n--- Relatório Final (conjunto de teste) ---")
    print(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"Threshold aplicado: {threshold}")
    print(classification_report(y_test, y_pred))

    # 6. Persistência do modelo no bucket `models`
    model_ref = store.path("models", prefix + mcfg["filename"])
    with store.open("models", prefix + mcfg["filename"], "wb") as fh:
        joblib.dump(model, fh)
    print(f"Modelo salvo em: {model_ref}")

    # 7. Importância das variáveis do modelo final
    fig_ref = store.path("models", prefix + "feature_importance_final.png")
    pd.Series(model.feature_importances_, index=top_features).nlargest(10).plot(kind="barh")
    plt.title("Top 10 Variáveis (após merge Bureau + PrevApp)")
    plt.tight_layout()
    with store.open("models", prefix + "feature_importance_final.png", "wb") as fh:
        plt.savefig(fh, format="png")
    print(f"Gráfico salvo em: {fig_ref}")


if __name__ == "__main__":
    train_model()
