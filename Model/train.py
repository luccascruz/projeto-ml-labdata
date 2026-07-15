"""
Módulo de Treinamento e Avaliação Dinâmica de Modelos (Versão Docker Otimizada).
"""

from storage import get_storage
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import joblib
import yaml
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')

# --- CONFIGURAÇÃO ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with open(PROJECT_ROOT / "Model" / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_feature_importance(model, feature_names, store, model_name):
    """Gera gráfico de importância e salva no MinIO para explicabilidade."""
    est = model.named_steps['model'] if isinstance(model, Pipeline) else model
    if hasattr(est, 'feature_importances_'):
        imp = pd.Series(est.feature_importances_,
                        index=feature_names).nlargest(15)
        plt.figure(figsize=(8, 6))
        imp.sort_values().plot(kind='barh', color='skyblue')
        plt.title(f'Top 15 Features: {model_name}')
        plt.tight_layout()
        with store.open("models", f"models/{model_name}/feature_importance.png", "wb") as fh:
            plt.savefig(fh)
        plt.close()


def prepare_model_data(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    test_size: float,
    val_size: float,
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


def build_models(cfg, y_train):
    """Constrói modelos dinamicamente a partir do config.yml."""
    random_state = cfg["project"]["random_state"]
    models_cfg = cfg["models"]
    models = {}

    for name, info in models_cfg.items():
        if not info.get("enabled", True):
            continue
        params = info.get("params", {}).copy()
        params["random_state"] = random_state

        if info["class"] == "XGBClassifier":
            neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
            params["scale_pos_weight"] = neg / pos
            models[name] = xgb.XGBClassifier(**params)
        elif info["class"] == "RandomForestClassifier":
            models[name] = RandomForestClassifier(**params)
        elif info["class"] == "LogisticRegression":
            base = LogisticRegression(**params)
            models[name] = Pipeline([("scaler", StandardScaler()), ("model", base)]) if info.get(
                "use_scaler") else base
    return models


def train_and_evaluate():
    cfg = load_config()
    store = get_storage(cfg)
    kw = store.io_kwargs()
    target = cfg["project"]["target"]

    print("Carregando ABT do MinIO...")
    df = pd.read_parquet(store.path("abt", cfg["abt_files"]["abt"]), **kw)

    # Separação da ABT para treino, validação, e teste (holdout)
    split_cfg = cfg.get("split", {})
    train_df, val_df, test_df = prepare_model_data(
        df,
        target,
        cfg["project"]["random_state"],
        test_size=split_cfg.get("test_size"),
        val_size=split_cfg.get("val_size"),
    )
    
    # --- Pseudo logging pra observar
    print("\n--- Divisão dos Dados ---")
    total = len(df)
    for name, split in [
        ("Treino", train_df),
        ("Validação", val_df),
        ("Teste", test_df),
    ]:
        print(
            f"{name:<10}: {len(split):>7,} amostras "
            f"({len(split)/total:.1%}) | "
            f"TARGET=1: {split[target].mean():.2%}"
        )
    print("-------------------------\n")

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    # --- Persistência de Artefatos de Suporte ---
    print("Salvando artefatos de inferência e auditoria...")

    artifacts = {
        "evaluation_data/medianas.pkl": X_train.median(numeric_only=True),
        "evaluation_data/X_train.pkl": X_train,
        "evaluation_data/y_train.pkl": y_train,
        "evaluation_data/feature_names.pkl": X_train.columns.tolist(),
        "evaluation_data/X_val.pkl": X_val,
        "evaluation_data/y_val.pkl": y_val,
        "evaluation_data/X_test.pkl": X_test,
        "evaluation_data/y_test.pkl": y_test,
    }
    for path, obj in artifacts.items():
        with store.open("models", path, "wb") as fh:
            joblib.dump(obj, fh)

    # --- Loop de Treinamento ---
    modelos = build_models(cfg, y_train)
    resultados = []

    # Configuração de limites: Otimiza para Recall > min_recall_target dentro do range [min, max]
    min_recall = cfg.get("evaluation", {}).get("min_recall_target", 0.70)
    range_cfg = cfg.get("evaluation", {}).get(
        "threshold_range", {"min": 0.1, "max": 0.9})

    for nome, model in modelos.items():
        print(f"Treinando {nome}...")
        if isinstance(model, xgb.XGBClassifier):
            model.fit(X_train, y_train, eval_set=[
                      (X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        prec, rec, thresh = precision_recall_curve(y_val, probs)

        # Otimização: Acha o threshold que maximiza a precisão, mantendo recall >= min_recall
        # E respeitando os limites [min, max] definidos no YAML
        valid_mask = (thresh >= range_cfg["min"]) & (
            thresh <= range_cfg["max"])
        recall_mask = rec[:-1] >= min_recall

        candidates = np.where(valid_mask & recall_mask)[0]

        if len(candidates) > 0:
            # Pega o índice que tem a maior precisão entre os candidatos válidos
            best_idx = candidates[np.argmax(prec[candidates])]
            best_thresh = thresh[best_idx]
        else:
            # Fallback seguro caso nenhum ponto atenda a restrição
            best_thresh = range_cfg.get("min", 0.3)
            print(
                f"AVISO: {nome} não atingiu o Recall mínimo no range. Usando threshold mínimo.")

        resultados.append(
            {'Modelo': nome, 'AUC': auc, 'Threshold': best_thresh})

        # Persistência
        with store.open("models", f"models/{nome}/model.pkl", "wb") as fh:
            joblib.dump(model, fh)
        with store.open("models", f"models/{nome}/threshold.txt", "w") as f:
            f.write(str(best_thresh))

        save_feature_importance(model, X_train.columns, store, nome)

    with store.open("models", "leaderboard.csv", "w") as fh:
        pd.DataFrame(resultados).sort_values(
            by='AUC', ascending=False).to_csv(fh, index=False)

    print("Treino e persistência finalizados.")


if __name__ == "__main__":
    train_and_evaluate()