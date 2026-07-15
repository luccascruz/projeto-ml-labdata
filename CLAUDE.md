# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

Projeto Final (Capstone) do MBA em Big Data & Analytics (LABDATA / FIA). Vale como avaliação da disciplina de Inteligência Artificial **e** como etapa do TCC. Simula o ciclo real de um projeto de Machine Learning numa empresa, seguindo o framework **CRISP-DM**. A base escolhida é **Home Credit Default Risk** (Kaggle — <https://www.kaggle.com/competitions/home-credit-default-risk/overview>): prever a probabilidade de um candidato pagar um empréstimo (variável alvo `TARGET`), usando dados alternativos para ampliar a inclusão financeira de pessoas sem histórico de crédito.

Premissa norteadora da avaliação: **"O modelo é apenas o meio, não o fim."** Cada decisão técnica precisa estar justificada pelo problema de negócio e pela ação que a empresa tomará a partir do resultado. A entrega final inclui um pitch executivo de 15 min (Demoday) focado em problema de negócio, insights da EDA e se o modelo ficou bom/ruim **e o porquê** — não apenas a métrica.

Métrica oficial de avaliação do modelo: **ROC AUC**.

## Estado atual

A **primeira parte (etapa de grupo)** já foi implementada — o repositório não é mais um scaffold vazio. Já existem:

- `DataPipeline/data_sanitization.py` — limpeza raw → clean (saídas direto em `/Dados`: `clean_data.csv` e tabelas auxiliares limpas).
- `DataPipeline/abt_transform.py` — construção da ABT (clean → `Dados/abt.csv`).
- `DataPipeline/config.yml` — **config do domínio de dados** (caminhos, nomes de arquivo, target, sanitização, agregações da ABT).
- `Model/config.yml` — **config do domínio de modelo/avaliação** (hiperparâmetros, `test_size`, balanceamento, threshold, métricas). Os dois configs cumprem a exigência do brief (config em `/DataPipeline` **e** em `/Model`).
- `Model/train.py` — treino (RandomForest + SMOTE) lendo `Model/config.yml`, salvando o `.pkl` no bucket `models` do MinIO (via `storage.py`; ver nota de infra abaixo).
- `Model/evaluation.ipynb` e `DataPipeline/exp_analysis.ipynb` — notebooks de avaliação e EDA (ainda iniciais).
- `requirements.txt` e `README.md` — preenchidos.

**Layout de dados (estrutura do brief, slide 10):** artefatos direto em `/Dados` (flat) — `clean_data.csv` e `abt.csv` com os nomes canônicos pedidos. Os brutos mantêm os nomes originais do Kaggle (`application_train.csv`, `bureau.csv`, `previous_application.csv`) porque a base é multi-tabela; tabelas auxiliares limpas (`clean_bureau.csv`, `clean_previous_application.csv`) são intermediárias da ABT.

Os scripts **leem do config** (cada um do config do seu domínio) — **NESTE repositório, a I/O do pipeline (`DataPipeline/data_sanitization.py`, `DataPipeline/abt_transform.py`, `Model/train.py`) é toda no MinIO (S3)** via o helper `storage.py` (raiz): não há mais modo de filesystem local para o pipeline. Os nomes dos buckets vêm do bloco `storage.buckets` de cada config; endpoint e credenciais vêm de variáveis de ambiente (nunca do `config.yml`). Ver `MLOps/README.md` para a arquitetura completa (Airflow + MinIO + Docker).

**Estado da entrega individual (15/07):** os quatro entregáveis do brief individual já existem neste repositório:

- `/Model/predict.py` — carrega modelo/threshold/medianas/feature_names do bucket `models` no MinIO e faz inferência (`carregar_modelo_minio` + `prever_risco`).
- `/MLOps/app/main.py` — serviço de predição via **Streamlit** (não FastAPI; decisão tomada nesta branch), com validação de regra de negócio (idade, comprometimento de renda) antes de chamar o modelo.
- `/MLOps/dags/home_credit_pipeline.py` + `docker-compose.yml` — orquestração `raw → clean → abt → train` no Airflow contra o MinIO, com `seeder` para carregar os CSVs locais no bucket `raw`. Stack completa: MinIO, Postgres (metadados Airflow), Airflow webserver/scheduler, serviço Streamlit.
- `/MLOps/pipeline_orchestration.py` — runner sequencial local (sanitize → build_abt → train) que espelha a DAG, útil para validar o fluxo sem subir o Airflow inteiro.
- `/MLOps/README.md` — arquitetura desenhada (diagrama + tabela de componentes) e proposta de **monitoramento** (performance batch via ROC AUC/KS/precisão-recall-F1 ao maturar o rótulo, drift via PSI, falhas operacionais) e de **ações automatizadas com agente de IA** (decisão automática por faixa de score, agente de explicabilidade/próxima ação, re-treino disparado por drift) — desenho completo, implementação de referência ainda pendente para o DAG de drift e o agente.

**Vazamento treino/teste — resolvido nesta branch:** a dúvida do Lucas em aberto com a professora ("como separar treino/teste evitando vazamento") foi endereçada — o split (`train_test_split` estratificado, treino/validação/teste) e a imputação por mediana saíram do `abt_transform.py` (que agora só grava a ABT crua, uma linha por empréstimo, sem split) e foram para dentro de `Model/train.py::prepare_model_data`. A mediana é calculada **somente no treino** (pós-split) e aplicada em validação/teste — sem vazamento de informação entre splits. `data_sanitization.py` não faz mais nenhuma imputação. `Model/config.yml` define as proporções em `split.test_size`/`split.val_size`.

**Pendências conhecidas** (a tratar se sobrar tempo, não bloqueiam a entrega): a avaliação (`evaluation.ipynb`) ainda precisa ser conferida para rodar sobre o `X_test`/`y_test` held-out salvo pelo `train.py` (`evaluation_data/X_test.pkl`), e não sobre a base inteira; DAG de drift/monitoramento e o agente de justificativa ainda são desenho, não código.

**Monitoramento — qual métrica acompanhar (a definir/implementar):** decidir a métrica de acompanhamento em vez de assumir acurácia por padrão. Como a `TARGET` é muito desbalanceada (~8% de inadimplência), a **acurácia engana**: um modelo que nunca prevê inadimplência já "acerta" ~92% e ainda assim é inútil. Priorizar então **ROC AUC** (métrica oficial) e **KS**, mais **precisão/recall/F1 da classe inadimplente**, comparando **predito × valor real** conforme os rótulos reais (pagou/não pagou) forem chegando — como o desfecho do empréstimo tem defasagem, o monitoramento *online* se apoia também em *drift* do score e das features (ex.: PSI) até os rótulos maturarem. Além da métrica técnica, definir uma **variável/KPI de negócio** para julgar se o modelo é "bom" na ótica da empresa (ex.: taxa de inadimplência da carteira aprovada pelo modelo, perda esperada evitada, ganho por decisão) — coerente com o princípio **"o modelo é o meio, não o fim"**.

O serviço de predição já foi implementado como **Streamlit** em `MLOps/app/main.py` (o placeholder `fastAPIorstreamlit` mencionado em versões anteriores deste arquivo não existe mais — foi renomeado ao implementar).

## Arquitetura: fluxo de dados (a regra central)

O coração do projeto é um pipeline em camadas. Cada estágio lê a saída do anterior; entender qualquer script exige conhecer essa cadeia:

```
raw_data.csv  --(DataPipeline/data_sanitization.py)-->  clean_data.csv
clean_data.csv  --(DataPipeline/abt_transform.py)----->  abt.csv  (ABT = Analytical Base Table)
abt.csv  --(Model/train.py)--------------------------->  modelo treinado
modelo + dados novos  --(Model/predict.py)------------>  predições
```

Os três artefatos de dados (`raw_data.csv`, `clean_data.csv`, `abt.csv`) vivem, na etapa de grupo (sem Docker), em `/Dados`; **nesta branch de infra, o pipeline lê e grava sempre nos buckets do MinIO** (`raw`/`clean`/`abt`/`models` — ver `MLOps/README.md`), não mais em `/Dados`. A base original tem múltiplas tabelas — `application_train.csv` é a principal (uma linha por empréstimo); `bureau.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv` e `installments_payments.csv` são históricas e precisam ser agregadas para uma linha por empréstimo durante a construção da ABT.

Mapa de diretórios:

- **`/Dados`** — artefatos de dados (`raw_data.csv` → `clean_data.csv` → `abt.csv`). Não versionar CSVs pesados; usar `.gitignore`.
- **`/DataPipeline`** — manipulação de dados em **scripts `.py`** (não notebooks):
  - `data_sanitization.py` — limpeza (raw → clean)
  - `abt_transform.py` — construção da ABT (clean → abt)
  - `config.yml` — configuração do pipeline (caminhos, metadados, variável alvo)
  - `exp_analysis.ipynb` — **único notebook permitido aqui**; EDA feita **sobre o dado limpo** (`clean_data.csv`), não sobre o bruto.
- **`/Model`** — `train.py` (treinamento), `evaluation.ipynb` (notebook de avaliação + interpretabilidade) e seu **próprio arquivo de configuração** (o brief exige config em `/DataPipeline` **e** em `/Model`). `predict.py` aqui é entregável da etapa individual.
- **`/MLOps`** — etapa individual (deploy): `pipeline_orchestration.py` (orquestração), `app/` (serviço Streamlit ou API) e `Readme.md` próprio com o desenho da arquitetura. `docker-compose.yml` na raiz.

## Regras inegociáveis do projeto

Estas regras vêm da especificação de entrega e têm precedência sobre conveniência de código:

1. **Proibido hardcoding.** Nenhum caminho, parâmetro, nome de coluna, variável alvo ou metadado pode ficar "chumbado" nos scripts. Tudo deve ser lido de arquivos de configuração (`config.yml` já reservado em `DataPipeline/`; também são aceitos `.json`/`.env`). Ao escrever qualquer script, primeiro defina o parâmetro no config e leia-o.

2. **Pipeline em scripts `.py`, não em notebooks.** Toda manipulação de dados (limpeza e ABT) é código `.py` modular, para ser orquestrável depois. Notebooks existem **apenas** para EDA (`exp_analysis.ipynb`) e avaliação (`evaluation.ipynb`).

3. **EDA é sobre o dado limpo.** A análise exploratória roda em cima de `clean_data.csv`, depois do `data_sanitization.py` — nunca sobre o raw.

4. **Código pensado para MLOps desde já.** A fase seguinte fará deploy com Docker, Airflow (orquestrando do dado bruto até a ABT) e FastAPI/Streamlit para servir o modelo. Por isso o código de pipeline precisa ser modular, parametrizado e executável de forma independente (cada estágio como entrypoint chamável), não acoplado a estado de notebook.

5. **Multiplataforma (Linux, macOS e Windows).** O projeto é desenvolvido no Windows, mas a professora pode rodá-lo em **Linux ou macOS** — então tudo precisa funcionar em qualquer um dos três sistemas. Regras práticas:
   - **Caminhos sempre portáveis:** monte paths com `pathlib.Path` / `os.path.join` (nunca concatene strings com `\` nem fixe separadores ou letras de drive como `C:\`). Caminhos vêm do config e são relativos à raiz do projeto.
   - **Sem dependências de SO no código:** nada de chamadas a `cmd`/PowerShell, executáveis `.exe` ou comandos específicos de Windows dentro dos scripts `.py`.
   - **Quebra de linha e encoding:** salve arquivos como UTF-8 com LF (`\n`); use um `.gitattributes` para evitar que o Git converta finais de linha. Ao ler/escrever arquivos, passe `encoding="utf-8"` explicitamente.
   - **`requirements.txt` sem fixar build de SO:** não inclua wheels nem versões atreladas a uma única plataforma; prefira versões que tenham distribuição para Linux/macOS/Windows.
   - **Docker é a garantia final:** o `docker-compose` (Airflow + serviço de predição) deve subir igual nos três sistemas, usando imagens base Linux — é a forma canônica de reproduzir o projeto na máquina da professora independentemente do SO dela.

## Cronograma, entregáveis e avaliação (oficial — brief LABDATA FIA)

Datas no formato DD/MM (edição 2026).

### Cronograma do trabalho em grupo
- **Dia 1 — Kickoff (22/06):** apresentação do contexto, objetivos e aula-exemplo.
- **Dia 2 — Dados (29/06):** definir o problema, o impacto no negócio e as métricas de sucesso; EDA (padrões, qualidade da base, nulos, inconsistências, comportamento das variáveis); estruturar a **ABT**.
- **Dia 3 — Modelo (06/07):** desenvolver a modelagem e avaliar em cenário de teste. **Traga mais de uma métrica de avaliação do modelo** — além do ROC AUC oficial, reporte métricas complementares (ex.: precisão, recall, F1, KS, matriz de confusão, curva PR) para sustentar a leitura do desempenho à luz do desbalanceamento da `TARGET` e do problema de negócio.
- **Dia 4 — Narrativa de negócio:** análise crítica do desempenho (limitações, erros, vieses, cenários de falha) + storytelling + **Demoday** (pitch de 15 min para banca/empresa).
- **Entrega final do grupo: 13/07.**
- **Entrega final individual: 15/07.**

### Entregáveis do grupo
- **A) Pitch da solução** (Demoday).
- **B) PowerPoint com 5 slides**, nesta ordem exata: (1) problema de negócio; (2) análise exploratória; (3) ABT; (4) modelo — técnica do ciclo de desenvolvimento (modelo de ML, controle de overfitting, hiperparâmetros); (5) avaliação — performance do algoritmo + explicabilidade.
- **C) Código no Git** seguindo a estrutura obrigatória (ver "Arquitetura"). Cada integrante mantém o **seu próprio repositório**. `/DataPipeline` e `/Model` precisam cada um do seu arquivo de configuração. `README.md` deve conter: descrição do projeto + objetivo de negócio + resumo da metodologia + instruções de como treinar o modelo.

### Entregável individual (deploy / arquitetura)
Desenvolvimentos adicionais no mesmo repositório:
- `/Model/predict.py`.
- `/MLOps/`: `Readme.md` (desenho da arquitetura com componentes + próximos passos de monitoramento e automação), `docker-compose`, `pipeline_orchestration.py` e `/app` (resultado servido via Streamlit ou API). Acrescentar ao README as instruções de execução do serviço de predição.
- Propor e demonstrar: (a) arquitetura funcional completa, da origem dos dados ao deploy do modelo como serviço de predição; (b) infraestrutura via `docker-compose` (a orquestração `bruta → clean → abt → Model` roda em Airflow); (c) **monitoramento** de dados e modelo em produção — falhas, perda de performance e mudança de comportamento dos dados (*data drift*); (d) **ações automatizadas** acionadas pelas previsões do modelo, conectando ML + automação + agentes de IA.
- **Esteja preparado para modificar o projeto ao vivo na banca** (a professora sorteia e pede para alterar parâmetros do modelo ou da orquestração na hora).

### Critérios de avaliação
- **Grupo** — *BUSINESS KNOWLEDGE* (alinhamento estratégico; viabilidade operacional) e *RESULTS* (análise e diagnóstico; métricas e governança). RESULTS é a defesa do **"podemos confiar que vai funcionar?"**: apresente métricas de rastreabilidade da confiança do modelo, de resultado de negócio e de conformidade de governança.
- **Individual** — *FUNDAMENTALS* (fundamentação teórica; mecanismos técnicos) e *CODING* (qualidade de código; tratamento de dados; construção de pipeline).
- **Pesos (TCC):** Pitch 20% (nota do grupo) · Entregável técnico 20% · Banca final 60% (notas individuais). O foco maior da avaliação está na entrega final e na defesa da solução.

## Dicas e Diretrizes da Professora para o Projeto Final (MBA)

Compilado das exigências e conselhos da professora para o sucesso nas etapas em grupo e individuais.

### 1. Contexto de Negócio e Storytelling
- **O modelo é o meio, não o fim:** a solução não é o modelo pelo modelo. Defina bem o contexto de negócio — qual a dor ou oportunidade da empresa e como a saída do modelo resolve isso.
- **Visão executiva:** na apresentação do *Demoday*, foco executivo. Explique o "porquê" (o problema), o "que" e os "resultados". Os detalhes técnicos minuciosos (o "como") são avaliados nos repositórios e na etapa individual.
- **Ciclo completo:** mostre o discurso de ponta a ponta. Se o modelo prevê *churn*, o que o negócio fará com isso? (ex: acionar gerente comercial, ligar para o cliente).

### 2. Dados, Modelagem e Avaliação
- **Sustente seus resultados:** não importa se a acurácia foi 99% ou "uma porcaria" — o fundamental é saber explicar o **porquê** o resultado foi bom ou ruim.
- **A EDA justifica o modelo:** é a análise exploratória que sustenta os resultados, cria hipóteses e explica as métricas finais.
- **Resultados negativos são válidos:** se as variáveis não correlacionam ou os dados são insuficientes e o modelo falha, isso é um resultado válido — desde que você prove o motivo pela EDA.

### 3. Qualidade de Código e Repositório (GitHub)
- **Repositório individual:** mesmo com o código do grupo sendo igual, **cada aluno** deve ter o próprio repositório Git com o trabalho do grupo.
- **Sem caminhos/variáveis "chumbados" (hardcoded):** nunca engesse nomes de variáveis ou paths no script. Use arquivos de configuração (`.json`, `.csv`, `.yml` ou módulo `.py`) para passar parâmetros e metadados.
- **Scripts vs. Notebooks:** limpeza, padronização, criação da ABT e deploy usam **scripts** `.py`, não notebooks. Jupyter fica restrito à EDA e à avaliação do modelo.
- **Reprodutibilidade:** o repositório precisa ter `requirements.txt` e `README.md` com instruções claras para a professora treinar e reproduzir o modelo do zero na máquina dela — que pode ser **Linux ou macOS** (ver regra 5, "Multiplataforma"). As instruções do README devem usar comandos que funcionem nesses sistemas (ex.: `python3 -m venv`, ativação `source .venv/bin/activate`) e o caminho via Docker como alternativa garantida.

### 4. Apresentação em Grupo (Demoday)
- **Gestão de tempo:** exatos 15 minutos. Ensaie. Se o tempo acabar e ainda estiverem no primeiro slide, a apresentação é cortada no meio.
- **Divisão de falas:** o grupo define quem apresenta (uma pessoa ou todas) e quem responde dúvidas. A nota desta etapa é coletiva.

### 5. Etapa Individual (Banca e Deploy)
- **Demonstração funcional:** na banca, você executa o projeto de ponta a ponta — orquestração rodando no Docker com Airflow até a resposta gerada via FastAPI ou Streamlit. Não são só slides.
- **Modificações ao vivo:** esteja preparado para alterar o código na hora. A professora sorteia e pede para mudar algum parâmetro do modelo ou da orquestração ao vivo, para ver como o sistema se comporta.
- **Evite "copiar e colar" cego:** o pipeline pode ser semelhante entre membros do grupo, mas cópias sem sentido (ex: deixar o path da máquina do colega no seu código) somadas à incapacidade de explicar o próprio código resultam em reprovação sumária (Sub).

## Divisão de tarefas do grupo (etapa individual)

O grupo dividiu as frentes de trabalho da etapa individual/deploy assim. Cada integrante mantém o **seu próprio repositório** (regra do brief), mas as tarefas abaixo mostram quem está puxando cada parte da solução compartilhada.

**Dúvidas em aberto (a esclarecer com a professora):**
- Onde exatamente a análise exploratória deve ser feita (dado bruto vs. dado limpo / ABT). *(Alex)*

> A dúvida do Lucas sobre separação treino/teste sem vazamento já foi resolvida nesta branch — ver "Vazamento treino/teste — resolvido nesta branch" acima.

**Frentes por integrante:**
- **Alex** — script de análise exploratória sobre os dados brutos **e** sobre a ABT (para orientar o treino) + `Model/predict.py` para uso no teste do FastAPI.
- **Luccas (eu)** — melhorias no código + configurações do repositório + infraestrutura Docker (**MinIO + Airflow**). *(frente deste repositório/branch)*
- **Diego** — desenvolvimento do FastAPI (entrada de dados → saída do resultado do modelo) + montagem do PowerPoint da apresentação (recebe do grupo evidências, métricas, testes e trechos de código).
- **Italo** — script de avaliação e métricas do modelo (apoio às apresentações e interpretações).
- **Victor** — incluir no script o teste de mais dois modelos (Regressão **Logística** e **Boosting**) para comparar e escolher o melhor + documentação técnica de todo o processo (ingestão → transformação → treino) + pipeline de ingestão/transformação no Airflow + MinIO + FastAPI + validar a `%` de drop de colunas (se é a melhor forma para treinar) + desenho da arquitetura.

> Observação: várias frentes se sobrepõem em Airflow/MinIO/FastAPI e documentação — alinhar interfaces (nomes de artefatos, contrato de entrada/saída do modelo, caminhos no config) para os trabalhos encaixarem.

## Idioma

Comunicação, comentários e documentação do projeto em **português (Brasil)**.
