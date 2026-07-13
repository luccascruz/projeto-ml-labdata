FROM python:3.12-slim

WORKDIR /app

# Agora o requirements.txt está no mesmo nível do contexto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Para copiar o restante do código da raiz, precisamos sair da pasta MLOps/docker/streamlit/
# O Docker consegue subir níveis com ../
COPY ../../../ . 

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "MLOps/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]