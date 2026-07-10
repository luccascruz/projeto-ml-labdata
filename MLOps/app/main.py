import streamlit as st
import pandas as pd
import numpy as np
import sys
import datetime
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# --- CONFIGURAÇÃO ---
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from Model.predict import carregar_melhor_modelo, prever_risco

# --- CLASSE DE FEATURE ENGINEERING ---
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in ['AMT_CREDIT_SUM_DEBT', 'AMT_INCOME_TOTAL', 'BUREAU_LOAN_COUNT']:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X['DTI_RATIO'] = X['AMT_CREDIT_SUM_DEBT'] / X['AMT_INCOME_TOTAL'].replace(0, np.nan)
        X['AVG_DEBT_PER_LOAN'] = X['AMT_CREDIT_SUM_DEBT'] / X['BUREAU_LOAN_COUNT'].replace(0, np.nan)
        return X.replace([np.inf, -np.inf], np.nan)

# --- INTERFACE ---
st.set_page_config(page_title="Análise de Risco", page_icon="💳", layout="wide")
st.title("💳 Sistema de Análise de Risco de Crédito")

# Data limite para 18 anos atrás
data_limite = datetime.date(datetime.date.today().year - 18, 12, 31)

with st.sidebar:
    st.header("📋 Dados do Cliente")
    with st.form("form_inputs"):
        genero = st.radio("Gênero", ["Masculino", "Feminino"], horizontal=True)
        data_nasc = st.date_input("Data de nascimento", value=data_limite, max_value=data_limite)
        anos_emprego = st.number_input("Anos no emprego atual", min_value=0, value=5)
        
        st.divider()
        
        amt_income = st.number_input("Renda Mensal (R$)", min_value=1.0, value=5000.0)
        amt_credit = st.number_input("Valor total do empréstimo (R$)", min_value=1000.0, value=20000.0)
        amt_annuity = st.number_input("Parcela mensal desejada (R$)", min_value=100.0, value=1000.0)
        amt_debt = st.number_input("Dívida total atual (R$)", min_value=0.0, value=1000.0)
        bureau_count = st.number_input("Qtd. de empréstimos ativos", min_value=0, value=1)
        
        submitted = st.form_submit_button("Analisar Risco")

if submitted:
    # 1. Trava de Negócio: Proporção Renda vs Crédito
    if amt_credit / amt_income > 15:
        st.error(f"Erro: O valor solicitado ({amt_credit:,.2f}) excede nosso limite máximo de 15x a sua renda.")
        st.stop()

    try:
        # 2. Carrega modelo dinâmico
        model, threshold = carregar_melhor_modelo()
        expected_cols = model.named_steps['preprocessor'].feature_names_in_
        
        # 3. Inicializa DataFrame com colunas esperadas
        input_data = pd.DataFrame(np.nan, index=[0], columns=expected_cols)
        
        # 4. Mapeamento dos campos
        if 'CODE_GENDER' in input_data: input_data['CODE_GENDER'] = 1 if genero == "Masculino" else 0
        input_data['DAYS_BIRTH'] = -(pd.to_datetime('today') - pd.to_datetime(data_nasc)).days
        input_data['DAYS_EMPLOYED'] = -(anos_emprego * 365)
        input_data['AMT_INCOME_TOTAL'] = amt_income
        input_data['AMT_CREDIT'] = amt_credit
        input_data['AMT_ANNUITY'] = amt_annuity
        input_data['AMT_CREDIT_SUM_DEBT'] = amt_debt
        input_data['BUREAU_LOAN_COUNT'] = bureau_count
        
        # 5. Predição
        prob, is_mau = prever_risco(model, input_data, threshold)
        
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
        st.error(f"Erro no processamento: {e}")

st.caption("Pipeline MLOps - Inteligência Artificial para Decisão de Crédito")