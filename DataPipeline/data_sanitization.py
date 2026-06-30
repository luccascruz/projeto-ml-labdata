import pandas as pd
import os
from pathlib import Path

# Configuração de caminhos baseada no diretório atual
RAW_DIR = Path("Dados/raw")
TRUSTED_DIR = Path("Dados/trusted")


# Garante que a pasta 'trusted' existe
os.makedirs(TRUSTED_DIR, exist_ok=True)

def sanitize_application(df):
    # Remoção de colunas com mais de 50% de nulos
    percentual_nulos = (df.isnull().sum() / len(df)) * 100
    df = df.drop(columns=percentual_nulos[percentual_nulos > 50].index.tolist())
    
    # 1. Tratar colunas NUMÉRICAS com a MEDIANA
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 2. Tratar colunas de OBJETO/STRING com 'Desconhecido'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Desconhecido')
    
    # Feature Engineering (Idade)
    if 'DAYS_BIRTH' in df.columns:
        df['IDADE_ANOS'] = abs(df['DAYS_BIRTH']) / 365
        df.drop(columns=['DAYS_BIRTH'], inplace=True)
    return df


def sanitize_previous_app(df):
    """Limpeza específica para o previous_application"""
    # 1. Tratar apenas colunas NUMÉRICAS com 0
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # 2. Tratar apenas colunas de CATEGORIA (object) com 'Desconhecido'
    cat_cols = df.select_dtypes(include=['object']).columns
    # Usamos loc para garantir que a atribuição seja feita na coluna correta
    df[cat_cols] = df[cat_cols].fillna('Desconhecido')
    
    return df


def sanitize_bureau(df):
    """Limpeza específica para o bureau"""
    # Preencher nulos de valores monetários com 0
    cols_monetarias = [col for col in df.columns if 'AMT' in col]
    df[cols_monetarias] = df[cols_monetarias].fillna(0)
    return df


def sanitize_previous_app(df):
    """Limpeza específica para o previous_application"""
    
    # 1. Identifica colunas numéricas e preenche com 0
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # 2. Identifica colunas de categoria e preenche com 'Desconhecido'
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Desconhecido')
    
    return df

def run_sanitization():
    # Mapeamento: Arquivo -> Função de Limpeza
    tasks = {
        'application_train.csv': sanitize_application,
        'bureau.csv': sanitize_bureau,
        'previous_application.csv': sanitize_previous_app
    }
    
    for filename, clean_func in tasks.items():
        input_path = os.path.join(RAW_DIR, filename)
        if os.path.exists(input_path):
            print(f"--- Sanitizando: {filename} ---")
            df = pd.read_csv(input_path)
            df_clean = clean_func(df)
            
            output_path = os.path.join(TRUSTED_DIR, f"clean_{filename}")
            df_clean.to_csv(output_path, index=False)
            print(f"Salvo em: {output_path}")
        else:
            print(f"Arquivo não encontrado: {filename}")


if __name__ == "__main__":
    run_sanitization()