# Projeto final do módulo de AI/ML da LabData FIA: Home Credit

- **Contexto**: O Home Credit, empresa que oferece empréstimos para clientes com pouco ou nenhum histórico de crédito, enfrenta o desafio de equilibrar a concessão de crédito com a minimização da inadimplência.
- A "Dor" do Negócio: A análise manual ou baseada em regras simples de crédito é lenta e falha em capturar padrões complexos de risco, resultando em perda de receita (clientes bons negados) ou aumento de prejuízo (clientes inadimplentes aprovados).
- **Objetivo**: Desenvolver um modelo de Machine Learning preditivo capaz de classificar a probabilidade de inadimplência de um solicitante, permitindo uma tomada de decisão automatizada, mais rápida e mais precisa.
- **Impacto Esperado**: Aumentar a rentabilidade da carteira de crédito e promover a inclusão financeira de forma sustentável para o negócio.


## Como treinar o modelo

### 1. Ambiente virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dados

Baixe a base do Kaggle (<https://www.kaggle.com/competitions/home-credit-default-risk/overview>)
e coloque os CSVs **brutos** na pasta `Dados/`:
`application_train.csv`, `bureau.csv`, `previous_application.csv`.

### 3. Pipeline (do dado bruto ao modelo)

O pipeline lê e grava **no MinIO (S3)**, não em arquivos locais. A forma
recomendada de rodar de ponta a ponta é via Docker, que sobe MinIO + Airflow e
orquestra `raw → clean → abt → train` — veja **[Rodar via Docker](#rodar-via-docker-infra-mlops)**.

Os estágios continuam sendo scripts `.py` chamáveis
(`DataPipeline/data_sanitization.py`, `DataPipeline/abt_transform.py`,
`Model/train.py`); rodá-los diretamente exige as variáveis de ambiente do MinIO
(`MINIO_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) apontando para uma
instância acessível.

Os parâmetros (buckets, colunas, hiperparâmetros, threshold) ficam nos arquivos de
configuração `DataPipeline/config.yml` e `Model/config.yml` — não é preciso editar os scripts.

## Rodar via Docker (infra MLOps)

Via reproduzível (Linux/macOS/Windows) que sobe MinIO + Airflow e roda o pipeline
`raw → clean → abt → train` orquestrado:

```bash
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # cole em AIRFLOW_FERNET_KEY
docker compose up -d --build
```

- MinIO: <http://localhost:9001> · Airflow: <http://localhost:8080>
- O `seeder` carrega `Dados/*.csv` no bucket `raw`; dispare o DAG
  `home_credit_pipeline` no Airflow. Detalhes em [MLOps/README.md](MLOps/README.md).
