# MLOps — Infraestrutura (Airflow + MinIO + Docker)

Frente de deploy/infra do projeto: orquestração do pipeline
`raw → clean → abt → train` no **Airflow**, com **MinIO** como data lake (S3).
Todo o I/O do pipeline é feito nos buckets do MinIO — **não há modo local de
filesystem**. A forma canônica de reproduzir o projeto é subir este stack.

## Arquitetura

```
Dados/*.csv (host)  --seeder-->            MinIO(raw)
MinIO(raw)          --[Airflow: sanitize]--> MinIO(clean, parquet)
MinIO(clean)        --[Airflow: build_abt]-> MinIO(abt/abt.csv)
MinIO(abt)          --[Airflow: train]-----> MinIO(models/<name>/modelo_risco_credito.pkl)
```

Componentes (`docker-compose.yml`, na raiz):

| Serviço             | Papel                                                        |
|---------------------|-------------------------------------------------------------|
| **minio**           | Object storage S3 (API `:9000`, console `:9001`).           |
| **createbuckets**   | Cria os buckets `raw/clean/abt/models` (`minio/mc`).        |
| **seeder**          | Sobe `Dados/*.csv` → bucket `raw` (config-driven, Dockerfile).|
| **postgres**        | Metadados do Airflow.                                        |
| **airflow-init**    | `db migrate` + cria usuário admin.                          |
| **airflow-webserver** | UI do Airflow (`:8080`).                                  |
| **airflow-scheduler** | Executa as tasks (LocalExecutor).                        |

O pipeline acessa o MinIO via `storage.py` (raiz): monta URIs `s3://bucket/arquivo`
e lê endpoint/credenciais de variáveis de ambiente. Os nomes dos buckets vêm do
`config.yml` de cada domínio (`DataPipeline/` e `Model/`).

## Pré-requisitos

- Docker Desktop / Docker Engine + Docker Compose.
- Os 3 CSVs brutos do Kaggle em `Dados/` (`application_train.csv`, `bureau.csv`,
  `previous_application.csv`). O `seeder` os envia para o bucket `raw`.

## Como subir

```bash
cp .env.example .env
# gerar a Fernet key e colar em AIRFLOW_FERNET_KEY no .env:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

docker compose up -d --build
```

- Console MinIO: <http://localhost:9001> (usuário/senha do `.env`)
- Airflow: <http://localhost:8080> (admin do `.env`)

O `seeder` roda automaticamente e popula o bucket `raw`.

## Rodar o pipeline

Na UI do Airflow, disparar o DAG `home_credit_pipeline`, ou via CLI:

```bash
docker compose exec airflow-scheduler airflow dags trigger home_credit_pipeline
```

Ao final, o bucket `models/` contém o `.pkl` e os artefatos de avaliação
(`X_test.csv`, `y_test.csv`, `feature_importance_final.png`).

## Modificação ao vivo (banca)

O repositório é montado em `/opt/project`. Alterar parâmetros em
`DataPipeline/config.yml` ou `Model/config.yml` no host e re-disparar o DAG
reflete a mudança **sem rebuild** da imagem.

## Próximos passos (monitoramento e automação)

- **Monitoramento de dados/modelo:** logar métricas por execução (ROC AUC, KS),
  validar schema/nulos na ingestão e alertar em falha.
- **Data drift:** comparar a distribuição das features novas vs. base de treino
  (PSI/KS) num DAG agendado; disparar re-treino ao ultrapassar um limiar.
- **Ações automatizadas:** conectar a saída do modelo a uma ação de negócio
  (aprovar/negar/encaminhar) via serviço de predição (FastAPI) + fila/evento.
- **Registro de modelos:** versionar o `.pkl` no bucket `models/` por data/commit.
```
