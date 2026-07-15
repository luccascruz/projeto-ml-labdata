"""Serviço de predição — Streamlit (entregável individual).

Coleta os dados do cliente, aplica as regras de negócio (config), chama o modelo
campeão publicado no MinIO e exibe probabilidade + ação recomendada.

As regras de negócio e o mapeamento das features vivem em `Model/predict.py` e
`Model/config.yml` — este arquivo é só a interface.
"""
import datetime
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Model.predict import (  # noqa: E402
    carregar_modelo_minio,
    load_config,
    prever_risco,
    top_features,
    validar_regras_negocio,
)

st.set_page_config(page_title="Análise de Risco", page_icon="💳", layout="wide")
st.title("💳 Sistema de Análise de Risco de Crédito")

cfg = load_config()
regras = cfg["inference"]["regras_negocio"]
meses = cfg["inference"]["meses_por_ano"]


@st.cache_resource(show_spinner="Carregando modelo campeão do MinIO...")
def _carregar():
    """Carrega o modelo uma vez por sessão (evita reler o .pkl a cada submit)."""
    return carregar_modelo_minio(cfg=cfg)


model, threshold, medianas, feature_names = _carregar()


def _padrao(coluna: str, divisor: float = 1.0, fallback: float = 0.0) -> float:
    """Valor default do formulário = mediana do treino.

    Ancorar os defaults na mediana faz o "cliente base" do demo ser o cliente
    TÍPICO da base. Sem isso, valores arbitrários caem fora da distribuição de
    treino e o modelo extrapola — gerando leituras sem sentido (ex.: renda maior
    aumentando o risco).
    """
    if coluna not in medianas.index:
        return fallback
    return round(float(medianas[coluna]) / divisor, 2)


hoje = datetime.date.today()
cem_anos_atras = datetime.date(hoje.year - 100, 1, 1)
idade_padrao = int(_padrao("AGE", fallback=40))
data_padrao = datetime.date(hoje.year - idade_padrao, hoje.month, hoje.day)

# --- SIDEBAR E INPUTS ---
with st.sidebar:
    st.header("📋 Dados do Cliente")
    st.caption(
        "Os valores iniciais são as **medianas da base de treino** — o cliente "
        "típico. Valores muito fora dessa escala levam o modelo a extrapolar."
    )
    with st.form("form_inputs"):
        genero = st.radio("Gênero", ["Masculino", "Feminino"], horizontal=True)
        data_nasc = st.date_input(
            "Data de nascimento",
            value=data_padrao,
            min_value=cem_anos_atras,
            max_value=hoje,
        )
        anos_emprego = st.number_input(
            "Anos no emprego atual", min_value=0, value=int(_padrao("EMPLOYMENT_YEARS", fallback=5))
        )
        st.divider()
        renda_mensal = st.number_input(
            "Renda mensal", min_value=1.0, value=_padrao("AMT_INCOME_TOTAL", meses, 12000.0)
        )
        valor_credito = st.number_input(
            "Valor total do empréstimo", min_value=1.0, value=_padrao("AMT_CREDIT", fallback=500000.0)
        )
        parcela_mensal = st.number_input(
            "Parcela mensal desejada", min_value=1.0, value=_padrao("AMT_ANNUITY", meses, 2000.0)
        )
        divida_total = st.number_input(
            "Dívida total atual (bureau)", min_value=0.0, value=_padrao("TOTAL_DEBT", fallback=0.0)
        )
        qtd_emprestimos = st.number_input(
            "Qtd. de empréstimos ativos", min_value=0, value=int(_padrao("BUREAU_LOAN_COUNT", fallback=1))
        )
        submitted = st.form_submit_button("Analisar Risco")

# --- LÓGICA DE PREDIÇÃO ---
if submitted:
    idade = (
        hoje.year
        - data_nasc.year
        - ((hoje.month, hoje.day) < (data_nasc.month, data_nasc.day))
    )

    dados_cliente = {
        "genero": "M" if genero == "Masculino" else "F",
        "idade": idade,
        "anos_emprego": anos_emprego,
        "renda_mensal": renda_mensal,
        "valor_credito": valor_credito,
        "parcela_mensal": parcela_mensal,
        "divida_total": divida_total,
        "qtd_emprestimos": qtd_emprestimos,
    }

    # 1. Regras de negócio ANTES do modelo (não gasta score em caso barrado).
    erro = validar_regras_negocio(dados_cliente, cfg)
    if erro:
        st.error(f"Recusado por regra de negócio: {erro}")
        st.stop()

    try:
        # 2. Predição (o mapeamento p/ features da ABT é feito no predict.py).
        prob, is_mau, acao = prever_risco(
            model, threshold, medianas, feature_names, dados_cliente, cfg=cfg
        )

        # 3. Resultado + ação automatizada.
        st.subheader("Resultado da Análise")
        col1, col2, col3 = st.columns(3)
        # "Score de risco", não "probabilidade": o treino usa rebalanceamento
        # (scale_pos_weight / class_weight) por causa dos ~8% de inadimplência,
        # então a saída é um score ordenador — NÃO uma probabilidade calibrada.
        # O que vale é a posição frente ao threshold, não o valor absoluto.
        col1.metric("Score de risco", f"{prob:.2%}")
        col2.metric("Threshold do modelo", f"{threshold:.2%}")

        if acao == "NEGAR":
            col3.error("Ação automática: NEGAR")
            st.warning("⚠️ Risco elevado — crédito negado automaticamente.")
        elif acao == "APROVAR":
            col3.success("Ação automática: APROVAR")
            st.success("✅ Risco baixo — crédito aprovado automaticamente.")
        else:
            col3.info("Ação: ANÁLISE HUMANA")
            st.info(
                "🔍 Score na zona cinzenta em torno do threshold "
                f"(±{cfg['inference']['zona_cinzenta']:.0%}) — encaminhado para "
                "análise manual em vez de decisão automática."
            )

        # 4. Explicabilidade.
        importancias = top_features(model, feature_names)
        if importancias is not None:
            with st.expander("Ver features que mais pesam na decisão do modelo"):
                st.dataframe(importancias, use_container_width=True)
                st.caption(
                    "Importância global do modelo (não é a contribuição deste "
                    "cliente específico). O score é ordenador, não calibrado: "
                    "o rebalanceamento do treino desloca o valor absoluto — por "
                    "isso a decisão sai da comparação com o threshold."
                )

    except Exception as e:
        st.error(f"Erro crítico: {e}")
        st.write("Verifique se o pipeline (DAG `home-credit-pipeline`) já rodou e publicou o modelo no MinIO.")

st.caption("Pipeline MLOps — Inteligência Artificial para Decisão de Crédito")
