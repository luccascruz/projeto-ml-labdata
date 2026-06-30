import pandas as pd
import os

# Caminhos
TRUSTED_DIR = r'C:\Users\MBA\DataEngineering\projeto_ml\projeto_ml\Dados\trusted'
OUTPUT_ABT = os.path.join(TRUSTED_DIR, 'abt_final.csv')

def build_abt():
    print("--- Construindo a ABT Final ---")
    
    # 1. Carregar bases sanitizadas
    app = pd.read_csv(os.path.join(TRUSTED_DIR, 'clean_application_train.csv'))
    bureau = pd.read_csv(os.path.join(TRUSTED_DIR, 'clean_bureau.csv'))
    prev_app = pd.read_csv(os.path.join(TRUSTED_DIR, 'clean_previous_application.csv'))

    # 2. Agregação Inteligente (O segredo do sucesso)
    # Agrupamos bureau pelo ID do cliente, calculando métricas de risco
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': 'sum',
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'DAYS_CREDIT': 'max'
    }).rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_EXT', 'AMT_CREDIT_SUM_DEBT': 'TOTAL_DEBT'})

    # Agrupamos prev_app calculando número de empréstimos anteriores
    prev_agg = prev_app.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count'
    }).rename(columns={'SK_ID_PREV': 'QTD_PREV_APPS'})

    # 3. Merge (Juntando tudo na tabela principal)
    print("Realizando os merges...")
    abt = app.merge(bureau_agg, on='SK_ID_CURR', how='left')
    abt = abt.merge(prev_agg, on='SK_ID_CURR', how='left')

    # 4. Limpeza pós-merge
    # Clientes sem registros em outras tabelas receberão 0 nessas novas colunas
    abt.fillna(0, inplace=True)

    # 5. Salvar ABT Final
    abt.to_csv(OUTPUT_ABT, index=False)
    print(f"ABT Final construída com sucesso! Shape: {abt.shape}")
    print(f"Salva em: {OUTPUT_ABT}")

if __name__ == "__main__":
    build_abt()