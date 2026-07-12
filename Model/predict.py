import joblib
import pandas as pd
import yaml
from pathlib import Path
from storage import get_storage

def carregar_modelo_minio(model_name: str = None):
    """Carrega artefatos do MinIO respeitando a estrutura de pastas atual."""
    # 1. Configuração e Storage
    PROJECT_ROOT = Path(__file__).parents[1]
    with open(PROJECT_ROOT / "Model" / "config.yml", "r") as f:
        cfg = yaml.safe_load(f)
    store = get_storage(cfg)
    
    # 2. Identificar melhor modelo se necessário
    if model_name is None:
        with store.open("models", "leaderboard.csv", "r") as f:
            model_name = pd.read_csv(f).iloc[0]['Modelo']

    # 3. Caminhos baseados nos seus prints:
    # A estrutura observada é: models/models/{nome_do_modelo}/...
    # E: models/evaluation_data/...
    
    # Carregamento do Modelo (caminho: models/models/NOME/model.pkl)
    # Note que adicionei 'models/' antes do nome_modelo para refletir o print
    model_path = f"models/{model_name}/model.pkl"
    threshold_path = f"models/{model_name}/threshold.txt"
    
    with store.open("models", model_path, "rb") as fh:
        model = joblib.load(fh)
    with store.open("models", threshold_path, "r") as f:
        threshold = float(f.read().strip())
        
    # Carregamento dos artefatos de inferência (caminho: models/evaluation_data/...)
    with store.open("models", "evaluation_data/medianas.pkl", "rb") as fh:
        medianas = joblib.load(fh)
    with store.open("models", "evaluation_data/feature_names.pkl", "rb") as fh:
        feature_names = joblib.load(fh)
        
    return model, threshold, medianas, feature_names

def prever_risco(model, threshold, medianas, feature_names, dados_input: pd.DataFrame):
    """Realiza a inferência com tratamento de dados consistente."""
    # Cria base com medianas e garante ordem das colunas
    df_modelo = pd.DataFrame([medianas], columns=feature_names).copy()
    
    # Atualiza apenas as colunas que vieram no input
    for col in dados_input.columns:
        if col in df_modelo.columns:
            df_modelo[col] = float(dados_input[col].iloc[0])
            
    # Predição
    prob = model.predict_proba(df_modelo)[0, 1]
    return float(prob), bool(prob >= threshold)