"""Treino do modelo de risco de crédito (abt -> modelo).

Lê o config próprio de /Model (Model/config.yml): hiperparâmetros, seleção de
variáveis, estratégia de balanceamento, threshold e caminhos — nada chumbado.
Caminhos resolvidos relativos à raiz do projeto (portável). Lê a ABT de
/Dados/abt.csv. Métrica oficial reportada: ROC AUC.
"""

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


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()

    target = cfg["project"]["target"]
    seed = cfg["project"]["random_state"]
    mcfg = cfg["model"]

    data_dir = PROJECT_ROOT / cfg["data"]["data_dir"]
    data_file = data_dir / cfg["data"]["abt_file"]

    print("--- Iniciando Pipeline com ABT Finalizada ---")
    df = pd.read_csv(data_file)

    #Atualização em 01/07, criando pasta antes para não sobescrever na persistência do modelo
    model_name = mcfg["name"]
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"] / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

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

    # 3.1 Salva conjunto de teste (X_test, y_test) para avaliação futura do modelo em evaluation.ipynb
    X_test.to_csv(model_dir / "X_test.csv", index=False)
    y_test.to_csv(model_dir / "y_test.csv", index=False)


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

    # 6. Persistência do modelo no diretório configurado
    model_path = model_dir / mcfg["filename"]
    joblib.dump(model, model_path)
    print(f"Modelo salvo em: {model_path}")

    # 7. Importância das variáveis do modelo final
    fig_path = model_dir / "feature_importance_final.png"
    pd.Series(model.feature_importances_, index=top_features).nlargest(10).plot(kind="barh")
    plt.title("Top 10 Variáveis (após merge Bureau + PrevApp)")
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Gráfico salvo em: {fig_path}")


if __name__ == "__main__":
    train_model()
