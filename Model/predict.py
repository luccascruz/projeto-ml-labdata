"""Inferência do modelo de risco de crédito (entregável individual).

Carrega os artefatos do bucket `models` do MinIO (modelo, threshold, medianas e
feature_names) e faz a predição para um cliente novo.

Ponto central deste módulo — `montar_features`: o modelo NÃO é treinado sobre os
campos crus do formulário (`DAYS_BIRTH`, `CODE_GENDER`, ...), e sim sobre o
espaço de features da ABT (`AGE`, `CODE_GENDER_M`, `CREDIT_INCOME_RATIO`, ...).
Por isso o input do cliente precisa ser traduzido para esse espaço com as MESMAS
derivações usadas em `DataPipeline/abt_transform.py` (fonte da verdade das
regras). Sem essa tradução, os campos do formulário não encontram coluna
correspondente, são silenciosamente descartados e o score fica praticamente
insensível ao que o usuário digita.

Os parâmetros de negócio (idade, comprometimento de renda, escala da renda) vêm
de `Model/config.yml`, bloco `inference` — nada é chumbado aqui.
"""
from pathlib import Path

import joblib
import pandas as pd
import yaml

from storage import get_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "Model" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Carrega o config.yml do domínio de modelo."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_modelo_minio(model_name: str | None = None, cfg: dict | None = None):
    """Carrega os artefatos de inferência do MinIO.

    Sem `model_name`, usa o campeão do `leaderboard.csv` (maior AUC), que é
    gravado ordenado por `Model/train.py`.
    """
    cfg = cfg or load_config()
    store = get_storage(cfg)

    if model_name is None:
        with store.open("models", "leaderboard.csv", "r") as f:
            model_name = pd.read_csv(f).iloc[0]["Modelo"]

    with store.open("models", f"models/{model_name}/model.pkl", "rb") as fh:
        model = joblib.load(fh)
    with store.open("models", f"models/{model_name}/threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    with store.open("models", "evaluation_data/medianas.pkl", "rb") as fh:
        medianas = joblib.load(fh)
    with store.open("models", "evaluation_data/feature_names.pkl", "rb") as fh:
        feature_names = joblib.load(fh)

    return model, threshold, medianas, feature_names


def top_features(model, feature_names, n: int = 10) -> pd.DataFrame | None:
    """Features mais importantes do modelo (explicabilidade).

    Retorna None para modelos sem `feature_importances_` (ex.: LogisticRegression,
    que expõe coeficientes, não importâncias).
    """
    clf = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        return None
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": clf.feature_importances_}
        )
        .sort_values(by="importance", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def montar_features(
    dados_cliente: dict,
    medianas: pd.Series,
    feature_names: list,
    cfg: dict,
) -> pd.DataFrame:
    """Traduz o input cru do cliente para o espaço de features da ABT.

    Espera em `dados_cliente` (unidades do formulário):
        genero            -> "M" ou "F"
        idade             -> anos
        anos_emprego      -> anos no emprego atual
        renda_mensal      -> R$/mês
        valor_credito     -> R$ (valor total do empréstimo)
        parcela_mensal    -> R$/mês
        divida_total      -> R$ (dívida registrada no bureau)
        qtd_emprestimos   -> contagem de empréstimos no bureau

    As colunas não informadas ficam na mediana do treino — é o comportamento
    desejado: representam "cliente típico" para o que não foi perguntado.
    As derivações abaixo espelham `DataPipeline/abt_transform.py`.
    """
    inf_cfg = cfg["inference"]
    meses = inf_cfg["meses_por_ano"]

    # Base: mediana do treino para TODAS as features, na ordem exata do fit.
    df = pd.DataFrame([medianas], columns=feature_names)

    # A ABT trabalha com renda/anuidade ANUAIS (base Kaggle); o form é mensal.
    renda_anual = dados_cliente["renda_mensal"] * meses
    anuidade_anual = dados_cliente["parcela_mensal"] * meses
    credito = dados_cliente["valor_credito"]
    divida = dados_cliente["divida_total"]
    qtd_emprestimos = dados_cliente["qtd_emprestimos"]

    # Features derivadas de EXT_SOURCE ficam na mediana (o cliente não informa
    # score de bureau externo), mas as INTERAÇÕES com renda precisam ser
    # recalculadas: deixá-las na mediana enquanto AMT_INCOME_TOTAL muda faz o
    # modelo ver um cliente incoerente (renda alta + interação de renda baixa).
    ext_source_2 = medianas.get("EXT_SOURCE_2", 0.0)
    ext_source_3 = medianas.get("EXT_SOURCE_3", 0.0)

    valores = {
        # --- application ---
        "AGE": dados_cliente["idade"],
        "EMPLOYMENT_YEARS": dados_cliente["anos_emprego"],
        "FLAG_SEM_HISTORICO_EMPREGO": 0,
        "AMT_INCOME_TOTAL": renda_anual,
        "AMT_CREDIT": credito,
        "AMT_ANNUITY": anuidade_anual,
        # --- razões financeiras (espelham build_application_features) ---
        "CREDIT_INCOME_RATIO": credito / renda_anual if renda_anual else None,
        "ANNUITY_INCOME_RATIO": anuidade_anual / renda_anual if renda_anual else None,
        "ANNUITY_CREDIT_RATIO": anuidade_anual / credito if credito else None,
        # --- bureau (espelham build_bureau_features) ---
        "TOTAL_DEBT": divida,
        "MEAN_DEBT": divida / qtd_emprestimos if qtd_emprestimos else 0.0,
        "MAX_DEBT": divida,
        "BUREAU_LOAN_COUNT": qtd_emprestimos,
        "FLAG_SEM_HISTORICO_BUREAU": int(qtd_emprestimos == 0),
        # --- cruzadas (espelham build_final_features) ---
        "DTI_RATIO": divida / renda_anual if renda_anual else None,
        "AVG_DEBT_PER_LOAN": divida / qtd_emprestimos if qtd_emprestimos else 0.0,
        "INCOME_x_EXT_SOURCE_2": renda_anual * ext_source_2,
        "INCOME_x_EXT_SOURCE_3": renda_anual * ext_source_3,
    }

    for coluna, valor in valores.items():
        if coluna in df.columns and valor is not None:
            df.loc[:, coluna] = valor

    # Gênero virou dummy no treino (pd.get_dummies): zera todas as variantes e
    # liga só a do cliente, em vez de tentar escrever em "CODE_GENDER" (que não
    # existe no modelo).
    dummies_genero = [c for c in df.columns if c.startswith("CODE_GENDER_")]
    if dummies_genero:
        df.loc[:, dummies_genero] = 0
        alvo = f"CODE_GENDER_{dados_cliente['genero']}"
        if alvo in df.columns:
            df.loc[:, alvo] = 1

    return df


def validar_regras_negocio(dados_cliente: dict, cfg: dict) -> str | None:
    """Aplica as regras de negócio anteriores ao modelo.

    Retorna a mensagem de recusa, ou None se o cliente pode seguir para o score.
    Regras (idade e comprometimento) vêm do config — alteráveis sem tocar no app.
    """
    regras = cfg["inference"]["regras_negocio"]
    idade = dados_cliente["idade"]

    if idade < regras["idade_minima"] or idade > regras["idade_maxima"]:
        return (
            f"Serviço disponível apenas para clientes entre "
            f"{regras['idade_minima']} e {regras['idade_maxima']} anos. "
            f"Idade informada: {idade} anos."
        )

    renda_anual = dados_cliente["renda_mensal"] * cfg["inference"]["meses_por_ano"]
    teto = regras["max_credito_sobre_renda"]
    if renda_anual and dados_cliente["valor_credito"] / renda_anual > teto:
        return (
            f"Valor solicitado (R$ {dados_cliente['valor_credito']:,.2f}) excede "
            f"o limite de {teto}x a renda anual (R$ {renda_anual:,.2f})."
        )

    return None


def decidir(prob: float, threshold: float, cfg: dict) -> str:
    """Traduz a probabilidade em ação de negócio (ver MLOps/README.md).

    Fora da zona cinzenta a decisão é automática; dentro dela vai para análise
    humana, em vez de o modelo decidir sozinho perto do corte.
    """
    zona = cfg["inference"]["zona_cinzenta"]
    if prob >= threshold + zona:
        return "NEGAR"
    if prob <= threshold - zona:
        return "APROVAR"
    return "ANALISE_HUMANA"


def prever_risco(
    model,
    threshold: float,
    medianas: pd.Series,
    feature_names: list,
    dados_cliente: dict,
    cfg: dict | None = None,
) -> tuple[float, bool, str]:
    """Prediz o risco de inadimplência de um cliente.

    Retorna (probabilidade, is_inadimplente, acao) — a ação já é a decisão de
    negócio (APROVAR / NEGAR / ANALISE_HUMANA).
    """
    cfg = cfg or load_config()
    df_modelo = montar_features(dados_cliente, medianas, feature_names, cfg)

    prob = float(model.predict_proba(df_modelo)[0, 1])
    return prob, bool(prob >= threshold), decidir(prob, threshold, cfg)
