# MLOps — Infraestrutura (Airflow + MinIO + Docker)

Frente de deploy/infra do projeto: orquestração do pipeline
`raw → clean → abt → train` no **Airflow**, com **MinIO** como data lake (S3).
Todo o I/O do pipeline é feito nos buckets do MinIO — **não há modo local de
filesystem**. A forma canônica de reproduzir o projeto é subir este stack.

## Arquitetura

![desenho_arquitetura](arquitetura_MLOps.png)

```
Dados/raw/*.csv (host) --seeder-->             MinIO(raw)
MinIO(raw)          --[DAG: data-sanitization]--> MinIO(clean/*.parquet)
MinIO(clean)        --[DAG: build-abt]---------> MinIO(abt/abt.parquet)
MinIO(abt)          --[DAG: train-model]-------> MinIO(models/models/<nome>/model.pkl)
MinIO(models)       --[Streamlit: predict.py]--> decisão de crédito (:8501)
MinIO(models)       --[DAG: check-monitoring]--> MinIO(monitoring/relatorio_monitoramento.csv)
                                              └-> [drift?] --> re-treino automático
```

Componentes (`docker-compose.yml`, na raiz):

| Serviço             | Papel                                                        |
|---------------------|-------------------------------------------------------------|
| **minio**           | Object storage S3 (API `:9000`, console `:9001`).           |
| **createbuckets**   | Cria os buckets `raw/clean/abt/models/monitoring` (`minio/mc`). |
| **seeder**          | Sobe `Dados/raw/*.csv` → bucket `raw` (config-driven, Dockerfile).|
| **postgres**        | Metadados do Airflow.                                        |
| **airflow-init**    | `db migrate` + cria usuário admin.                          |
| **airflow-webserver** | UI do Airflow (`:8080`).                                  |
| **airflow-scheduler** | Executa as tasks (LocalExecutor).                        |
| **streamlit-app**   | Serviço de predição (`:8501`) — consome o modelo do MinIO.  |

DAGs (`MLOps/dags/`):

| DAG                       | Papel                                                     |
|---------------------------|-----------------------------------------------------------|
| **home-credit-pipeline**  | `raw → clean → abt → train` (disparo manual).             |
| **home-credit-monitoring**| Performance + data drift (`@daily`) → dispara re-treino.  |

O pipeline acessa o MinIO via `storage.py` (raiz): monta URIs `s3://bucket/arquivo`
e lê endpoint/credenciais de variáveis de ambiente. Os nomes dos buckets vêm do
`config.yml` de cada domínio (`DataPipeline/` e `Model/`).

## Nota: SSL corporativo no build

Em redes corporativas com inspeção HTTPS (proxy/antivírus tipo Zscaler,
Kaspersky, ESET), o `pip install` dentro dos containers pode falhar com
`SSL: CERTIFICATE_VERIFY_FAILED` ao acessar o PyPI, porque o container não
confia no certificado usado pela inspeção. Os Dockerfiles (`MLOps/docker/*`)
já usam `--trusted-host pypi.org --trusted-host files.pythonhosted.org` como
contorno. Se ainda assim falhar, tente construir fora da VPN/rede corporativa,
ou instale o certificado raiz da sua rede na imagem base.

## Pré-requisitos

- Docker Desktop / Docker Engine + Docker Compose.
- Os 3 CSVs brutos do Kaggle em `Dados/raw/` (`application_train.csv`,
  `bureau.csv`, `previous_application.csv`). O `seeder` os envia para o bucket
  `raw`. (O `seeder` é idempotente: se os arquivos já estiverem no bucket, ele
  não precisa dos CSVs no host.)

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

Na UI do Airflow, disparar o DAG `home-credit-pipeline`, ou via CLI:

```bash
docker compose exec airflow-scheduler airflow dags trigger home-credit-pipeline
```

Alternativa sem Airflow: `MLOps/pipeline_orchestration.py` é um runner local que
executa a mesma sequência da DAG (`sanitize → build_abt → train`) — útil para
validar o fluxo ponta a ponta. Requer as variáveis de ambiente do MinIO
(`MINIO_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) apontando para
uma instância acessível:

```bash
python MLOps/pipeline_orchestration.py
```

Ao final, o bucket `models/` contém, por modelo treinado
(`XGBOOST`, `RANDOM_FOREST`, `LOGISTIC_REGRESSION`):

- `models/<nome>/model.pkl` e `models/<nome>/threshold.txt`
- `models/<nome>/feature_importance.png` (explicabilidade)
- `leaderboard.csv` — ranking por AUC; o topo é o **modelo campeão** que o
  `predict.py`/Streamlit carrega por padrão.
- `evaluation_data/` — `X_train/X_val/X_test`, `y_*`, `medianas.pkl` e
  `feature_names.pkl` (usados pela avaliação e pelo monitoramento).

## Serviço de predição (Streamlit)

```bash
docker compose up -d streamlit-app
```

Acesse <http://localhost:8501>. O serviço carrega o modelo campeão do bucket
`models` e devolve probabilidade + ação (APROVAR / NEGAR / ANÁLISE HUMANA).
Requer que o DAG `home-credit-pipeline` já tenha rodado.

## Rodar o monitoramento

```bash
docker compose exec airflow-scheduler airflow dags trigger home-credit-monitoring
```

Grava `monitoring/relatorio_monitoramento.csv` no MinIO. Se o PSI estourar o
limiar ou o ROC AUC cair abaixo do mínimo (`Model/config.yml` → `monitoring`),
a task falha e o DAG dispara automaticamente o re-treino.

## Modificação ao vivo (banca)

O repositório é montado em `/opt/project`. Alterar parâmetros em
`DataPipeline/config.yml` ou `Model/config.yml` no host e re-disparar o DAG
reflete a mudança **sem rebuild** da imagem.

## Monitoramento em produção (implementado)

**Onde está:** `Model/monitoring.py` (cálculo) + `MLOps/dags/monitoring_pipeline.py`
(DAG `home-credit-monitoring`, `@daily`) + bloco `monitoring` em `Model/config.yml`
(limiares). Saída em `monitoring/relatorio_monitoramento.csv` no MinIO.

A `TARGET` é muito desbalanceada (~8% de inadimplência), então acurácia engana —
um modelo que nunca prevê inadimplência "acerta" ~92% e é inútil. Por isso o
acompanhamento se apoia em três frentes:

1. **Performance do modelo (quando o rótulo real chega).** ✅ *implementado —
   `metricas_performance()`.* O desfecho do empréstimo tem defasagem (meses),
   então isso é acompanhamento *batch*, não online: a cada novo lote de
   empréstimos com resultado conhecido (pagou/não pagou), recalcula **ROC AUC**
   (métrica oficial), **KS** e **precisão/recall/F1 da classe inadimplente**,
   comparando **predito × real**. As métricas são persistidas por execução no
   bucket `monitoring/` para ver tendência, não só o valor pontual.
   *Ressalva honesta:* nesta entrega a "janela de produção" é o **conjunto de
   teste held-out** — dado que o modelo nunca viu no `fit`. É a simulação
   possível sem tráfego real; o mecanismo de cálculo é idêntico ao que rodaria
   sobre o lote de produção, bastando trocar a fonte da janela.
2. **Drift de dados e de score (sem esperar o rótulo).** ✅ *implementado —
   `calcular_psi()`.* Enquanto o rótulo não matura, monitora **PSI (Population
   Stability Index)** das features de entrada (`monitoring.features_monitoradas`
   no config) e da distribuição do score, comparando a janela recente contra a
   referência de treino. PSI > `psi_alerta` (0.2, padrão de mercado) dispara o
   alerta. Faixas usuais: < 0.1 estável | 0.1–0.2 atenção | > 0.2 drift.
3. **Falhas operacionais do pipeline/serviço.** ✅ *parcial.* O
   `monitoring.py` **falha explicitamente** (exit != 0) quando o drift estoura
   ou a performance cai — a task fica vermelha no Airflow, o que é o gancho de
   alerta e o gatilho do re-treino, em vez de degradar em silêncio. *Próximo
   passo:* task dedicada de validação de schema/nulos na ingestão e alerta por
   e-mail/Slack (`EmailOperator`/webhook), além de latência do endpoint.

Além da métrica técnica, o indicador que importa para o negócio é o **KPI de
carteira**: taxa de inadimplência real da carteira aprovada pelo modelo vs. a
taxa histórica sem modelo, e perda esperada evitada — é isso que sustenta
"o modelo é o meio, não o fim" na banca.

## Ações automatizadas (ML + automação + agentes de IA)

A saída do `predict.py` (probabilidade + decisão pelo threshold) alimenta um
fluxo de decisão, não só um número na tela:

- **Aprovação/negativa automática dentro da faixa de confiança.** ✅
  *implementado — `Model/predict.py::decidir()`, exposto no Streamlit.* Score
  abaixo de `threshold - zona_cinzenta` → **APROVAR** automático; acima de
  `threshold + zona_cinzenta` → **NEGAR** automático; dentro da zona cinzenta
  → **ANÁLISE HUMANA**, em vez de o modelo decidir sozinho justamente onde ele
  é menos confiável. A largura da zona é o parâmetro `inference.zona_cinzenta`
  (`Model/config.yml`).
- **Agente de IA para justificativa e próxima ação.** Um agente (LLM) recebe a
  probabilidade, o threshold e as *top features* que mais pesaram na decisão
  (`feature_importances_`, já extraído em `Model/predict.py`) e gera: (a) uma
  explicação em linguagem natural para o analista/cliente (explicabilidade,
  item já cobrado na etapa de grupo) e (b) uma recomendação de próxima ação
  (ex.: "solicitar comprovante de renda adicional" em vez de negar direto).
- **Disparo de re-treino por drift.** ✅ *implementado —
  `MLOps/dags/monitoring_pipeline.py`.* Quando o `check-monitoring` detecta PSI
  acima do limiar (ou performance degradada), a task falha e o
  `TriggerDagRunOperator` (`trigger_rule="one_failed"`) aciona automaticamente
  o DAG `home-credit-pipeline` para re-treinar com dados mais recentes. No
  caminho feliz nada é disparado.
- **Registro de modelos.** ⏳ *próximo passo.* Hoje o `.pkl` é sobrescrito por
  execução. Versionar por data/execução no bucket `models/` permitiria comparar
  e reverter (rollback) se o modelo novo performar pior — hoje o `leaderboard.csv`
  já dá a comparação **entre modelos**, mas não **entre versões no tempo**.

**Status desta entrega:**

| Item do brief individual              | Status                                        |
|---------------------------------------|-----------------------------------------------|
| (a) Arquitetura ponta a ponta         | ✅ origem → clean → ABT → treino → serviço    |
| (b) Infra `docker-compose` + Airflow  | ✅ MinIO + Postgres + Airflow + Streamlit     |
| (c) Monitoramento (falha/perf/drift)  | ✅ `Model/monitoring.py` + DAG `@daily`       |
| (d) Ações automatizadas               | ✅ decisão automática + re-treino por drift; ⏳ agente LLM de justificativa |

O **agente de IA de justificativa** (abaixo) permanece como desenho: é o único
item de automação não implementado, por decisão de escopo/tempo.

### Melhorias ambiente
- Rodar scripts python em seu próprio container com DockerOperator do airflow
- Otimizar Dockerfiles de cada container - remover libs não utilizadas
- Versões de imagens estáticas