"""Runner sequencial local do pipeline (espelha o DAG do Airflow).

Executa sanitize -> build_abt -> train em ordem, reutilizando os entrypoints
dos scripts. Como o pipeline e todo S3, exige as variaveis de ambiente do MinIO
(MINIO_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) apontando para uma
instancia acessivel. Util para validar o fluxo sem subir o Airflow.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from DataPipeline.data_sanitization import run_sanitization
from DataPipeline.abt_transform import build_abt
from Model.train import train_and_evaluate


def main() -> None:
    print("== [1/3] sanitize ==")
    run_sanitization()
    print("== [2/3] build_abt ==")
    build_abt()
    print("== [3/3] train ==")
    train_and_evaluate()
    print("== pipeline concluido ==")


if __name__ == "__main__":
    main()
