import joblib
import pandas as pd
import yaml
from pathlib import Path
from storage import get_storage

def carregar_modelo_minio(model_name: str = None):
    """Carrega artefatos do MinIO respeitando a estrutura de pastas atual e exibe features."""
    # 1. Configuração e Storage
    PROJECT_ROOT = Path(__file__).parents[1]
    with open(PROJECT_ROOT / "Model" / "config.yml", "r") as f:
        cfg = yaml.safe_load(f)
    store = get_storage(cfg)
    
    # 2. Identificar melhor modelo se necessário
    if model_name is None:
        with store.open("models", "leaderboard.csv", "r") as f:
            model_name = pd.read_csv(f).iloc[0]['Modelo']

    # 3. Carregamento dos artefatos
    # A estrutura observada é: models/models/{nome_do_modelo}/...
    # E: models/evaluation_data/...
    with store.open("models", f"models/{model_name}/model.pkl", "rb") as fh:
        model = joblib.load(fh)
    with store.open("models", f"models/{model_name}/threshold.txt", "r") as f:
        threshold = float(f.read().strip())
        
    # Carregamento dos artefatos de inferência
    with store.open("models", "evaluation_data/medianas.pkl", "rb") as fh:
        medianas = joblib.load(fh)
    with store.open("models", "evaluation_data/feature_names.pkl", "rb") as fh:
        feature_names = joblib.load(fh)
        
    # --- AJUSTE: Acessar o passo do classificador dentro do Pipeline ---
    # Geralmente o nome do passo no Pipeline é 'model' ou 'classifier'.
    # Verificamos se é um Pipeline para extrair as importâncias corretamente.
    try:
        clf = model.named_steps['model'] if hasattr(model, 'named_steps') else model
    except KeyError:
        # Se você não nomeou o passo, tenta pegar o último passo do pipeline
        clf = list(model.named_steps.values())[-1] if hasattr(model, 'named_steps') else model

    # Extraímos as importâncias do objeto real (ex: XGBoost ou RF)
    if hasattr(clf, 'feature_importances_'):
        importancias = pd.DataFrame({
            'feature': feature_names,
            'importance': clf.feature_importances_
        }).sort_values(by='importance', ascending=False)

        print(f"\n--- Top 10 Features Mais Importantes ({model_name}) ---")
        print(importancias.head(10))
        
    return model, threshold, medianas, feature_names

def prever_risco(model, threshold, medianas, feature_names, dados_input: pd.DataFrame):
    """Realiza a inferência com tratamento de dados consistente."""
    # Cria base com medianas e garante ordem das colunas para evitar desalinhamento
    df_modelo = pd.DataFrame([medianas], columns=feature_names).copy()
    
    # Atualiza apenas as colunas que vieram no input, mantendo as outras como medianas
    for col in dados_input.columns:
        if col in df_modelo.columns:
            df_modelo[col] = float(dados_input[col].iloc[0])
            
    # Predição de probabilidade
    probs = model.predict_proba(df_modelo)
    prob_final = float(probs[0, 1])
    
    # Retorna probabilidade e decisão baseada no threshold carregado
    return prob_final, bool(prob_final >= threshold)