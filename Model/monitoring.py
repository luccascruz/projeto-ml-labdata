"""Monitoramento do modelo em produção (entregável individual, item "c").

Cobre as três frentes exigidas, todas parametrizadas por `Model/config.yml`
(bloco `monitoring`) e com saída persistida no bucket `monitoring` do MinIO:

1. **Performance** — quando o rótulo real matura (pagou/não pagou), recalcula
   ROC AUC, KS e precisão/recall/F1 da classe inadimplente. Acurácia NÃO é
   usada como métrica de decisão: com ~8% de inadimplência, um modelo que nunca
   prevê inadimplente "acerta" ~92% e é inútil.
2. **Data drift** — PSI (Population Stability Index) das features e da
   distribuição do score, comparando a janela de produção contra a referência
   de treino. Não depende do rótulo, que só matura meses depois.
3. **Falhas** — o script sai com código != 0 quando a performance cai abaixo do
   mínimo ou o drift estoura o limiar, fazendo a task do Airflow falhar e
   disparar o alerta/re-treino (ver MLOps/dags/monitoring_pipeline.py).

Nesta entrega a "janela de produção" é o conjunto de teste held-out — dado que
o modelo nunca viu no fit. É a simulação honesta possível sem tráfego real: o
mecanismo de cálculo é o mesmo que rodaria sobre o lote de produção.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from storage import get_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "Model" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def calcular_psi(referencia: pd.Series, producao: pd.Series, n_buckets: int) -> float:
    """PSI entre a distribuição de referência (treino) e a de produção.

    Leitura usual de mercado: < 0.1 estável | 0.1–0.2 atenção | > 0.2 drift.
    Usa quantis da referência como cortes; colunas constantes retornam 0.
    """
    ref = pd.to_numeric(referencia, errors="coerce").dropna()
    prod = pd.to_numeric(producao, errors="coerce").dropna()
    if ref.empty or prod.empty or ref.nunique() <= 1:
        return 0.0

    # Cortes por quantil da referência (robusto a assimetria das variáveis
    # financeiras, que são muito enviesadas nesta base).
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, n_buckets + 1)))
    if len(cortes) <= 2:
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=cortes)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=cortes)[0] / len(prod)

    # Epsilon evita divisão por zero / log(0) em buckets vazios.
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    prod_pct = np.clip(prod_pct, eps, None)

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def calcular_ks(y_true: pd.Series, scores: np.ndarray) -> float:
    """KS: máxima separação entre as acumuladas de bons e maus."""
    from scipy.stats import ks_2samp

    return float(ks_2samp(scores[y_true == 1], scores[y_true == 0]).statistic)


def metricas_performance(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    """Métricas de performance à luz do desbalanceamento da TARGET."""
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "ks": calcular_ks(y_true, scores),
        # Classe 1 = inadimplente: é a classe que dói no negócio.
        "precision_inadimplente": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_inadimplente": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_inadimplente": float(f1_score(y_true, y_pred, zero_division=0)),
        "taxa_aprovacao": float((y_pred == 0).mean()),
        # KPI de negócio: inadimplência da carteira que o modelo aprovaria.
        # É o número que sustenta "o modelo é o meio, não o fim".
        "inadimplencia_carteira_aprovada": float(y_true[y_pred == 0].mean()),
        "inadimplencia_carteira_sem_modelo": float(y_true.mean()),
    }


def rodar_monitoramento(cfg: dict | None = None) -> dict:
    """Executa o ciclo de monitoramento e persiste o relatório no MinIO.

    Retorna um dicionário com o status e os indicadores apurados.
    """
    from Model.predict import carregar_modelo_minio

    cfg = cfg or load_config()
    mon_cfg = cfg["monitoring"]
    store = get_storage(cfg)

    print("Carregando artefatos de referência e da janela de produção...")
    with store.open("models", "evaluation_data/X_train.pkl", "rb") as fh:
        X_ref = joblib.load(fh)
    with store.open("models", "evaluation_data/X_test.pkl", "rb") as fh:
        X_prod = joblib.load(fh)
    with store.open("models", "evaluation_data/y_test.pkl", "rb") as fh:
        y_prod = joblib.load(fh)

    model, threshold, _, feature_names = carregar_modelo_minio(cfg=cfg)

    # ---------------------------------------------------------------- drift
    n_buckets = mon_cfg["psi_buckets"]
    monitorar = [c for c in mon_cfg["features_monitoradas"] if c in X_ref.columns]
    faltando = set(mon_cfg["features_monitoradas"]) - set(monitorar)
    if faltando:
        print(f"AVISO: features do config ausentes na ABT, ignoradas: {sorted(faltando)}")

    psi_features = {
        col: calcular_psi(X_ref[col], X_prod[col], n_buckets) for col in monitorar
    }

    scores_ref = model.predict_proba(X_ref)[:, 1]
    scores_prod = model.predict_proba(X_prod)[:, 1]
    psi_score = calcular_psi(pd.Series(scores_ref), pd.Series(scores_prod), n_buckets)

    # --------------------------------------------------------- performance
    perf = metricas_performance(y_prod, scores_prod, threshold)

    # ------------------------------------------------------------ vereditos
    limiar_psi = mon_cfg["psi_alerta"]
    auc_minimo = mon_cfg["roc_auc_minimo"]

    features_em_drift = {c: v for c, v in psi_features.items() if v > limiar_psi}
    drift_detectado = bool(features_em_drift) or psi_score > limiar_psi
    performance_ok = perf["roc_auc"] >= auc_minimo

    # ----------------------------------------------------- relatório MinIO
    relatorio = pd.DataFrame(
        [{"indicador": f"psi__{c}", "valor": v, "limiar": limiar_psi} for c, v in psi_features.items()]
        + [{"indicador": "psi__score", "valor": psi_score, "limiar": limiar_psi}]
        + [{"indicador": k, "valor": v, "limiar": auc_minimo if k == "roc_auc" else None}
           for k, v in perf.items()]
    )
    destino = mon_cfg["relatorio"]
    with store.open("monitoring", destino, "w") as fh:
        relatorio.to_csv(fh, index=False)

    print("\n--- Monitoramento ---")
    print(relatorio.to_string(index=False))
    print(f"\nRelatório salvo em: {store.path('monitoring', destino)}")
    print(f"ROC AUC: {perf['roc_auc']:.4f} (mínimo aceito: {auc_minimo})")
    print(f"PSI do score: {psi_score:.4f} (limiar: {limiar_psi})")
    if features_em_drift:
        print(f"Features em drift: {features_em_drift}")

    status = {
        "drift_detectado": drift_detectado,
        "performance_ok": performance_ok,
        "psi_score": psi_score,
        "features_em_drift": features_em_drift,
        **perf,
    }

    # Falha explícita = task vermelha no Airflow = alerta + gatilho de re-treino,
    # em vez de degradar silenciosamente em produção.
    if not performance_ok:
        raise ValueError(
            f"Performance abaixo do mínimo: ROC AUC {perf['roc_auc']:.4f} < {auc_minimo}"
        )
    if drift_detectado:
        raise ValueError(
            f"Data drift detectado (PSI > {limiar_psi}): "
            f"score={psi_score:.4f}, features={features_em_drift}"
        )

    print("\nStatus: modelo saudável (sem drift, performance dentro do mínimo).")
    return status


if __name__ == "__main__":
    rodar_monitoramento()
