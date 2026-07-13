"""DAG do pipeline Home Credit: raw -> clean -> abt -> train.

Cada task invoca o entrypoint __main__ do script correspondente (BashOperator),
mantendo os scripts desacoplados do Airflow. O backend de storage (S3) e as
credenciais do MinIO chegam via variaveis de ambiente do container (compose).
"""
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# PROJECT_HOME e definido no docker-compose (x-airflow-common), fonte unica do
# ponto de montagem do repo — o DAG nao chumba o caminho.
PROJECT = os.environ.get("PROJECT_HOME", "/opt/project")

default_args = {"owner": "mlops", "retries": 0}

with DAG(
    dag_id="home-credit-pipeline",
    description="raw -> clean -> abt -> train (Home Credit)",
    start_date=datetime(2026, 1, 1),
    schedule=None,          # disparo manual
    catchup=False,
    default_args=default_args,
    tags=["home-credit", "mlops"],
) as dag:
    
    sanitize = BashOperator(
        task_id="data-sanitization",
        bash_command=f"python {PROJECT}/DataPipeline/data_sanitization.py",
    )
    
    build_abt = BashOperator(
        task_id="build-abt",
        bash_command=f"python {PROJECT}/DataPipeline/abt_transform.py",
    )
    
    train = BashOperator(
        task_id="train-model",
        bash_command=f"python {PROJECT}/Model/train.py",
    )

    sanitize >> build_abt >> train
