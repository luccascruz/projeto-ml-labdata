import streamlit as st
import pandas as pd
import numpy as np
import sys
import datetime
from pathlib import Path
from Model.predict import carregar_modelo_minio, prever_risco

# --- CONFIGURAÇÃO ---
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# --- INTERFACE ---
st.set_page_config(page_title="Análise de Risco", page_icon="💳", layout="wide")
st.title("💳 Sistema de Análise de Risco de Crédito")
hoje = datetime.date.today()
cem_anos_atras = datetime.date(hoje.year - 100, 1, 1)
data_limite_18_anos = datetime.date(hoje.year - 18, hoje.month, hoje.day)

# --- SIDEBAR E INPUTS ---
with st.sidebar:
    st.header("📋 Dados do Cliente")
    with st.form("form_inputs"):
        genero = st.radio("Gênero", ["Masculino", "Feminino"], horizontal=True)
        data_nasc = st.date_input(
            "Data de nascimento", 
            value=data_limite_18_anos, 
            min_value=cem_anos_atras, 
            max_value=hoje
        )
        anos_emprego = st.number_input("Anos no emprego atual", min_value=0, value=1)
        st.divider()
        amt_income = st.number_input("Renda Mensal (R$)", min_value=1621.0, value=2000.0)
        amt_credit = st.number_input("Valor total do empréstimo (R$)", min_value=1000.0, value=1500.0)
        amt_annuity = st.number_input("Parcela mensal desejada (R$)", min_value=100.0, value=200.0)
        amt_debt = st.number_input("Dívida total atual (R$)", min_value=0.0, value=100.0)
        bureau_count = st.number_input("Qtd. de empréstimos ativos", min_value=0, value=1)
        submitted = st.form_submit_button("Analisar Risco")

# --- LÓGICA DE PREDIÇÃO ---
if submitted:
    # 1. Validação de Idade (Regra de Negócio)
    hoje = datetime.date.today()
    idade = hoje.year - data_nasc.year - ((hoje.month, hoje.day) < (data_nasc.month, data_nasc.day))
    
    if idade < 18 or idade > 65:
        st.error(f"Erro: O serviço está disponível apenas para clientes entre 18 e 65 anos. Sua idade: {idade} anos.")
        st.stop()

    # 2. Validação de Comprometimento de Renda
    if amt_credit / amt_income > 15:
        st.error(f"Erro: O valor solicitado ({amt_credit:,.2f}) excede nosso limite máximo de 15x a sua renda.")
        st.stop()

    try:
        # 3. Carrega artefatos do MinIO
        model, threshold, medianas, feature_names = carregar_modelo_minio()
        
        # 4. Mapeamento para o formato do modelo
        input_dict = {
            'CODE_GENDER': 1 if genero == "Masculino" else 0,
            'DAYS_BIRTH': -(pd.to_datetime('today') - pd.to_datetime(data_nasc)).days,
            'DAYS_EMPLOYED': -(anos_emprego * 365),
            'AMT_INCOME_TOTAL': amt_income,
            'AMT_CREDIT': amt_credit,
            'AMT_ANNUITY': amt_annuity,
            'AMT_CREDIT_SUM_DEBT': amt_debt,
            'BUREAU_LOAN_COUNT': bureau_count
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # 5. Predição
        prob, is_mau = prever_risco(model, threshold, medianas, feature_names, input_df)
        
        # 6. Exibição
        st.subheader("Resultado da Análise")
        col1, col2 = st.columns(2)
        col1.metric("Probabilidade de Risco", f"{prob:.2%}")
        
        if is_mau:
            col2.error("Classificação: Risco Elevado")
            st.warning("⚠️ Recomendação: Negar crédito.")
        else:
            col2.success("Classificação: Aprovado")
            st.balloons()
            
    except Exception as e:
        st.error(f"Erro crítico: {e}")
        st.write("Verifique se o modelo e os artefatos de inferência estão disponíveis no MinIO.")

st.caption("Pipeline MLOps - Inteligência Artificial para Decisão de Crédito")