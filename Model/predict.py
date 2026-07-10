from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from storage import get_storage

# --- CLASSE DE FEATURE ENGINEERING ---
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Garante tratamento de tipos antes do cálculo
        cols = ['AMT_CREDIT_SUM_DEBT', 'AMT_INCOME_TOTAL', 'BUREAU_LOAN_COUNT']
        for col in cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        X['DTI_RATIO'] = X['AMT_CREDIT_SUM_DEBT'] / X['AMT_INCOME_TOTAL'].replace(0, np.nan)
        X['AVG_DEBT_PER_LOAN'] = X['AMT_CREDIT_SUM_DEBT'] / X['BUREAU_LOAN_COUNT'].replace(0, np.nan)
        return X.replace([np.inf, -np.inf], np.nan)

def carregar_melhor_modelo():
    """
    Identifica dinamicamente o melhor modelo via leaderboard.csv
    e carrega o pipeline e o threshold otimizado.
    """
    try:
        PROJECT_ROOT = Path(__file__).parents[1]
        with open(PROJECT_ROOT / "Model" / "config.yml", "r") as f:
            cfg = yaml.safe_load(f)
        
        store = get_storage(cfg)
        
        # 1. Carregar Leaderboard da raiz do bucket
        with store.open("models", "leaderboard.csv", "r") as f:
            ranking = pd.read_csv(f)
        
        melhor_nome = ranking.iloc[0]['Modelo']
        
        # 2. Carregar o Modelo da subpasta correspondente
        # A estrutura agora é: models/XGBOOST/model.pkl
        with store.open("models", f"{melhor_nome}/model.pkl", "rb") as fh:
            model = joblib.load(fh)
            
        # 3. Carregar o Threshold da mesma subpasta
        with store.open("models", f"{melhor_nome}/threshold.txt", "r") as f:
            threshold_str = f.read().strip()
            threshold = float(threshold_str)
            
        return model, threshold
        
    except Exception as e:
        print(f"Erro ao carregar modelo do Storage: {e}")
        return None, None

def prever_risco(model, dados_input: pd.DataFrame, threshold: float):
    """
    Realiza a predição utilizando o threshold carregado dinamicamente.
    """
    try:
        # Probabilidade da classe 1 (Inadimplente)
        prob = model.predict_proba(dados_input)[:, 1][0]
        
        # Aplica o threshold carregado do arquivo
        is_mau = bool(prob >= threshold)
        
        return float(prob), is_mau
        
    except Exception as e:
        print(f"Erro crítico na inferência: {e}")
        return 0.0, False