import sys
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve

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

# --- CONFIGURAÇÃO ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from storage import get_storage

def save_feature_importance(model, feature_names, store, nome):
    est = model.named_steps['model']
    if hasattr(est, 'feature_importances_'):
        importances = est.feature_importances_
    elif hasattr(est, 'coef_'):
        importances = np.abs(est.coef_[0])
    else: return

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='skyblue')
    plt.title(f'Top 20 Features: {nome}')
    plt.tight_layout()
    
    # Salva na pasta raiz do modelo
    with store.open("models", f"{nome}/feature_importance.png", "wb") as fh:
        plt.savefig(fh)
        plt.close()

def train_and_evaluate():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "Model" / "config.yml"))
    store = get_storage(cfg)
    
    df = pd.read_csv(store.path("abt", "train.csv"), **store.io_kwargs())
    target = cfg["project"]["target"]
    
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in cols_numericas: cols_numericas.append(target)
    
    X_train = df[cols_numericas].drop(columns=[target])
    y_train = df[target]
    
    feat_eng = FeatureEngineering()
    X_train_trans = feat_eng.transform(X_train)
    feature_names = X_train_trans.columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), 
         make_column_selector(dtype_include=np.number))
    ], remainder='passthrough')

    modelos_grid = {
        'XGBOOST': {
            'model': xgb.XGBClassifier(n_jobs=1, scale_pos_weight=5), # Exemplo de peso para inadimplentes
            'params': {'model__n_estimators': [100]}
        },
        'RANDOM_FOREST': {'model': RandomForestClassifier(n_jobs=1), 'params': {'model__n_estimators': [100]}},
        'LOGISTIC_REGRESSION': {'model': LogisticRegression(max_iter=1000), 'params': {'model__C': [0.1]}}
    }

    leaderboard = []
    for nome, config in modelos_grid.items():
        print(f"Treinando {nome}...")
        pipe = Pipeline([('feat_eng', feat_eng), ('preprocessor', preprocessor), ('model', config['model'])])
        grid = GridSearchCV(pipe, config['params'], cv=2, scoring='roc_auc', n_jobs=1)
        grid.fit(X_train, y_train)
        
        melhor_modelo = grid.best_estimator_
        
        # Calcular Threshold Ótimo (Critério de Youden)
        probs = melhor_modelo.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, probs)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        
        # SALVAR ARTEFATOS (Direto na pasta do nome do modelo)
        with store.open("models", f"{nome}/model.pkl", "wb") as fh:
            joblib.dump(melhor_modelo, fh)
        
        with store.open("models", f"{nome}/threshold.txt", "w") as f:
            f.write(str(best_threshold))
        
        save_feature_importance(melhor_modelo, feature_names, store, nome)
        leaderboard.append({'Modelo': nome, 'AUC': grid.best_score_})

    # Salvar Leaderboard na raiz do bucket
    with store.open("models", "leaderboard.csv", "w") as fh:
        pd.DataFrame(leaderboard).sort_values(by='AUC', ascending=False).to_csv(fh, index=False)
    
    print("Treino concluído. Artefatos organizados no MinIO.")

if __name__ == "__main__":
    train_and_evaluate()