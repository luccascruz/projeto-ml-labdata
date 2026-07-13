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
    train_df = pd.read_parquet(store.path(
        "abt", cfg["abt_files"]["train"]), **kw)
    val_df = pd.read_parquet(store.path("abt", cfg["abt_files"]["val"]), **kw)
    test_df = pd.read_parquet(store.path(
        "abt", cfg["abt_files"]["test"]), **kw)

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    # --- Persistência de Artefatos de Suporte ---
    print("Salvando artefatos de inferência e auditoria...")

    with store.open("models", "evaluation_data/medianas.pkl", "wb") as fh:
        joblib.dump(X_train.median(numeric_only=True), fh)
    with store.open("models", "evaluation_data/feature_names.pkl", "wb") as fh:
        joblib.dump(X_train.columns.tolist(), fh)
    with store.open("models", "evaluation_data/X_val.pkl", "wb") as fh:
        joblib.dump(X_val, fh)
    with store.open("models", "evaluation_data/y_val.pkl", "wb") as fh:
        joblib.dump(y_val, fh)
    with store.open("models", "evaluation_data/X_test.pkl", "wb") as fh:
        joblib.dump(X_test, fh)
    with store.open("models", "evaluation_data/y_test.pkl", "wb") as fh:
        joblib.dump(y_test, fh)

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
