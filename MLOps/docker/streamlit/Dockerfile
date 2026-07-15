FROM python:3.12-slim

WORKDIR /app

# Build context é a raiz do projeto (definido em docker-compose.yml), então os
# caminhos abaixo são relativos à raiz — Docker não permite COPY de fora do
# build context (../../../ não funciona).
COPY MLOps/docker/streamlit/requirements.txt .
# --trusted-host: contorna falha de SSL em redes corporativas com inspecao
# HTTPS (proxy/antivirus). Ver MLOps/README.md.
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "MLOps/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]