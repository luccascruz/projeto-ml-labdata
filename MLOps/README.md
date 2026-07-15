# MLOps — Infraestrutura (Airflow + MinIO + Docker)

Frente de deploy/infra do projeto: orquestração do pipeline
`raw → clean → abt → train` no **Airflow**, com **MinIO** como data lake (S3).
Todo o I/O do pipeline é feito nos buckets do MinIO — **não há modo local de
filesystem**. A forma canônica de reproduzir o projeto é subir este stack.

## Arquitetura

![desenho_arquitetura](/MLOps/arquitetura_MLOps.png)

```
Dados/*.csv (host)  --seeder-->            MinIO(raw)
MinIO(raw)          --[Airflow: sanitize]--> MinIO(clean, parquet)
MinIO(clean)        --[Airflow: build_abt]-> MinIO(abt/abt.csv)
MinIO(abt)          --[Airflow: train]-----> MinIO(models/models/<name>/model.pkl)
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

## Monitoramento em produção (proposta)

A `TARGET` é muito desbalanceada (~8% de inadimplência), então acurácia engana —
um modelo que nunca prevê inadimplência "acerta" ~92% e é inútil. Por isso o
acompanhamento se apoia em três frentes:

1. **Performance do modelo (quando o rótulo real chega).** O desfecho do
   empréstimo tem defasagem (meses), então isso é acompanhamento *batch*, não
   online: a cada novo lote de empréstimos com resultado conhecido (pagou/não
   pagou), recalcular **ROC AUC** (métrica oficial), **KS** e
   **precisão/recall/F1 da classe inadimplente**, comparando **predito × real**.
   Persistir essas métricas por execução (ex.: tabela `model_metrics` no
   Postgres ou um bucket `monitoring/` no MinIO) para ver tendência ao longo do
   tempo, não só o valor pontual.
2. **Drift de dados e de score (sem esperar o rótulo).** Enquanto o rótulo não
   matura, monitorar **PSI (Population Stability Index)** das features de
   entrada e da distribuição do score do modelo, comparando a janela recente
   contra a base de treino. Um DAG agendado no Airflow roda esse cálculo e
   grava o PSI por feature; PSI > 0.2 numa feature crítica (renda, valor do
   crédito, dívida no bureau) ou no score geral dispara alerta.
3. **Falhas operacionais do pipeline/serviço.** Validar schema e nulos na
   ingestão (task dedicada no DAG, falha o pipeline em vez de propagar dado
   ruim), checar taxa de erro/latência do endpoint de predição (Streamlit/
   FastAPI) e alertar (e-mail/Slack via `EmailOperator`/webhook do Airflow) se
   uma task falhar ou o serviço ficar indisponível.

Além da métrica técnica, o indicador que importa para o negócio é o **KPI de
carteira**: taxa de inadimplência real da carteira aprovada pelo modelo vs. a
taxa histórica sem modelo, e perda esperada evitada — é isso que sustenta
"o modelo é o meio, não o fim" na banca.

## Ações automatizadas (proposta — ML + automação + agentes de IA)

A saída do `predict.py` (probabilidade + decisão pelo threshold) alimenta um
fluxo de decisão, não só um número na tela:

- **Aprovação/negativa automática dentro da faixa de confiança.** Score muito
  abaixo do threshold → aprovação automática; muito acima → negativa
  automática com justificativa gerada; na faixa intermediária (zona cinzenta
  em torno do threshold) → encaminha para análise humana em vez de decidir
  sozinho.
- **Agente de IA para justificativa e próxima ação.** Um agente (LLM) recebe a
  probabilidade, o threshold e as *top features* que mais pesaram na decisão
  (`feature_importances_`, já extraído em `Model/predict.py`) e gera: (a) uma
  explicação em linguagem natural para o analista/cliente (explicabilidade,
  item já cobrado na etapa de grupo) e (b) uma recomendação de próxima ação
  (ex.: "solicitar comprovante de renda adicional" em vez de negar direto).
- **Disparo de re-treino por drift.** Quando o DAG de monitoramento (item
  acima) detecta PSI acima do limiar por N execuções seguidas, ele aciona
  automaticamente o DAG `home_credit_pipeline` para re-treinar com dados mais
  recentes, e um agente resume o que mudou entre o modelo antigo e o novo
  (features, métricas) para revisão humana antes de promover o novo `.pkl`.
- **Registro de modelos.** Versionar o `.pkl` e as métricas de avaliação no
  bucket `models/` por data/execução, para poder comparar e reverter (rollback)
  se o modelo novo performar pior em produção.

**Status:** desenhado nesta entrega; implementação de referência priorizada
para o `predict.py` + Streamlit (serviço de predição) e para o DAG de
sanitização/ABT/treino, que já rodam ponta a ponta no Airflow. O DAG de
monitoramento/drift e o agente de justificativa são os próximos passos de
código a partir deste desenho.

### Melhorias ambiente
- Rodar scripts python em seu próprio container com DockerOperator do airflow
- Otimizar Dockerfiles de cada container - remover libs não utilizadas
- Versões de imagens estáticas