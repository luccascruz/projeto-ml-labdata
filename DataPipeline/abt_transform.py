import sys
import pandas as pd
from pathlib import Path
import yaml

# Configuração de caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"
sys.path.insert(0, str(PROJECT_ROOT))
from storage import get_storage

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_abt(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    store = get_storage(cfg)
    kw = store.io_kwargs()

    clean_files = cfg["data"]["clean_files"]
    id_col = cfg["project"]["id_column"]
    target = cfg["project"]["target"]

    print("--- Construindo ABT (Apenas Agregações) ---")

    # 1. Carregar bases sanitizadas
    app = pd.read_parquet(store.path("clean", clean_files["application"]), **kw)
    prev = pd.read_parquet(store.path("clean", clean_files["previous_application"]), **kw)
    bur = pd.read_parquet(store.path("clean", clean_files["bureau"]), **kw)

    # 2. Agregações (Apenas o que precisa ser reduzido para o nível do cliente)
    prev_stats = prev.groupby(id_col).agg({
        'SK_ID_PREV': 'count',
        'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').mean()
    }).rename(columns={'SK_ID_PREV': 'PREV_COUNT', 'NAME_CONTRACT_STATUS': 'PREV_APPROVAL_RATE'})

    bureau_stats = bur.groupby(id_col).agg({
        'SK_ID_BUREAU': 'count',
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_SUM': 'sum'
    }).rename(columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})

    # 3. Merge
    df = app.merge(prev_stats, on=id_col, how='left').merge(bureau_stats, on=id_col, how='left')
    
    # Preenchimento simples para evitar NaNs após o merge (opcional, mas recomendado)
    df = df.fillna(0)

    # 4. Divisão Train/Val
    print("Realizando split...")
    train, val = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target]
    )

    # 5. Escrita
    train.to_csv(store.path("abt", "train.csv"), index=False, **kw)
    val.to_csv(store.path("abt", "val.csv"), index=False, **kw)
    
    print(f"ABT Finalizada. Shape Train: {train.shape}, Val: {val.shape}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    build_abt()