"""DAG de monitoramento do modelo em producao (entregavel individual, itens c e d).

Fluxo:
    check-monitoring  -> avalia performance (ROC AUC/KS/precisao-recall-F1) e
                         data drift (PSI das features e do score), gravando o
                         relatorio no bucket `monitoring` do MinIO.
    trigger-retrain   -> AÇÃO AUTOMATIZADA: se o monitoramento falhar (drift
                         acima do limiar ou performance degradada), dispara o
                         DAG `home-credit-pipeline` para re-treinar com dados
                         mais recentes. So roda no caminho de falha
                         (trigger_rule=one_failed) — no caminho feliz nada
                         acontece, que e o comportamento desejado.

Os limiares vivem em Model/config.yml (bloco `monitoring`) — alterar la muda o
comportamento sem tocar no DAG.
"""
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# PROJECT_HOME vem do docker-compose (x-airflow-common): o DAG nao chumba path.
PROJECT = os.environ.get("PROJECT_HOME", "/opt/project")

default_args = {"owner": "mlops", "retries": 0}

with DAG(
    dag_id="home-credit-monitoring",
    description="Monitora performance + data drift e dispara re-treino automatico",
    start_date=datetime(2026, 1, 1),
    # Agendado: o monitoramento e recorrente, diferente do treino (manual).
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["home-credit", "mlops", "monitoring"],
) as dag:

    check = BashOperator(
        task_id="check-monitoring",
        bash_command=f"python {PROJECT}/Model/monitoring.py",
    )

    retrain = TriggerDagRunOperator(
        task_id="trigger-retrain",
        trigger_dag_id="home-credit-pipeline",
        # So dispara se o check falhar (drift/performance); sucesso => nada.
        trigger_rule="one_failed",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    check >> retrain
