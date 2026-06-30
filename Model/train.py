import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

DATA_FILE = os.path.join('Dados', 'trusted', 'abt_final.csv')

def train_model():
    print("--- Iniciando Pipeline com ABT Finalizada ---")
    
    # 1. Carregamento e Blindagem
    df = pd.read_csv(DATA_FILE)
    
    # Filtro essencial: Garantir que só trabalhamos com números (além do Target)
    target = df['TARGET']
    df = df.select_dtypes(include=['number']).drop(columns=['TARGET'], errors='ignore')
    df = df.fillna(0) # Segurança extra para nulos residuais
    
    # Adicionar o Target de volta
    df['TARGET'] = target
    
    # 2. Feature Engineering
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        df['RENDA_POR_FAMILIA'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # 3. Seleção de Variáveis por Importância (Sem correlação manual para não perder sinal)
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Rankeando variáveis...")
    rf_ranker = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_ranker.fit(X_train, y_train)
    
    # Seleção dos 20 melhores para dar mais contexto ao modelo
    importances = pd.Series(rf_ranker.feature_importances_, index=X.columns)
    top_features = importances.nlargest(20).index.tolist()
    
    # Aplicar corte
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    print(f"Features selecionadas para treino: {top_features}")

    # 4. Balanceamento e Treino
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=250, # Aumentei um pouco para acomodar as novas features
        max_depth=15,    # Reduzi levemente para evitar overfitting nas novas features
        class_weight={0: 1, 1: 5}, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # 5. Avaliação com Threshold 0.35
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.35).astype(int)

    print(f"\n--- Relatório Final ---")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'modelo_risco_credito.pkl')
    
    # Visualização de Elite
    pd.Series(model.feature_importances_, index=top_features).nlargest(10).plot(kind='barh')
    plt.title('Top 10 Variáveis após Merge (Bureau + PrevApp)')
    plt.tight_layout()
    plt.savefig('feature_importance_final.png')

if __name__ == "__main__":
    train_model()