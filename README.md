# Projeto final do módulo de AI/ML da LabData FIA: Home Credit Default Risk

- **Contexto**: O Home Credit, empresa que oferece empréstimos para clientes com pouco ou nenhum histórico de crédito, enfrenta o desafio de equilibrar a concessão de crédito com a minimização da inadimplência.
- A "Dor" do Negócio: A análise manual ou baseada em regras simples de crédito é lenta e falha em capturar padrões complexos de risco, resultando em perda de receita (clientes bons negados) ou aumento de prejuízo (clientes inadimplentes aprovados).
- **Objetivo**: Desenvolver um modelo de Machine Learning preditivo capaz de classificar a probabilidade de inadimplência de um solicitante, permitindo uma tomada de decisão automatizada, mais rápida e mais precisa.
- **Impacto Esperado**: Aumentar a rentabilidade da carteira de crédito e promover a inclusão financeira de forma sustentável para o negócio.

### Pré-Requisitos

* python 3.12
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

* Docker
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)


## 🚀 Primeiros Passos
```bash
# Para começar, clone o repositório em sua máquina local:
git clone https://github.com/luccascruz/projeto-ml-labdata.git

# Entre no seu projeto:
cd projeto-ml-labdata
```

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
e coloque os CSVs **brutos** na pasta `Dados/raw/`:
`application_train.csv`, `bureau.csv`, `previous_application.csv`.

O serviço `seeder` do Docker envia esses arquivos para o bucket `raw` do MinIO
automaticamente ao subir o stack.

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
docker compose up -d --build # Para baixar as imagens
docker compose up -d # Para iniciar os conteiners
```

Acesse o MinIO:
- MinIO: <http://localhost:9001>
  - **user**=minioadmin
  - **Password**=minioadmin123

Acesse o Airflow:
- Airflow: <http://localhost:8080>
  - **user**=admin
  - **Password**=admin

Acesse o Streamlit:
- Streamlit <http://localhost:8501>

- O `seeder` carrega `Dados/raw/*.csv` no bucket `raw`; dispare o DAG
  `home-credit-pipeline` no Airflow. Detalhes em [MLOps/README.md](MLOps/README.md).

- **obs**: Se der erro de permission 403, execute este código:
`docker compose exec airflow-scheduler chmod -R 777 /opt/airflow/logs`

## Serviço de predição (Streamlit)

Depois que o DAG `home-credit-pipeline` concluir (o modelo precisa existir no
bucket `models`), acesse <http://localhost:8501>. O serviço:

1. aplica as **regras de negócio** (idade, comprometimento de renda) — parâmetros
   em `Model/config.yml`, bloco `inference`;
2. carrega o **modelo campeão** (topo do `leaderboard.csv`) do MinIO;
3. devolve a probabilidade de inadimplência e a **ação automática**:
   APROVAR / NEGAR / ANÁLISE HUMANA (zona cinzenta em torno do threshold).

Disparar o pipeline / o monitoramento pela linha de comando:

```bash
docker compose exec airflow-scheduler airflow dags trigger home-credit-pipeline
docker compose exec airflow-scheduler airflow dags trigger home-credit-monitoring
```


### COMANDOS NO DOCKER
# 1. Parar tudo e remover containers
docker compose down

# 2. Remover caches de build antigos (limpeza profunda)
docker builder prune -f

# 3. Construir novamente
docker compose build --no-cache

# 4. Verificar os logs
docker compose logs -f streamlit-app

⚠️ **Observação** sobre Conflitos de Dependência (XGBoost)

O erro de serialização (`Input stream corrupted`) do XGBoost com o `joblib`
acontece quando o modelo é **gravado com uma versão e lido com outra** — não é
um defeito da 3.3.0 em si. Por isso a versão é **pinada igual** (`xgboost==3.3.0`)
no `requirements.txt` da raiz e nos requirements das imagens
(`MLOps/docker/*/requirements.txt`): quem treina (Airflow) e quem lê o `.pkl`
(Streamlit) usam a mesma versão.

Se for rodar fora do Docker, instale exatamente o `requirements.txt` da raiz —
não misture versões de XGBoost entre treino e inferência.