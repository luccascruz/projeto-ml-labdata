# Projeto final do módulo de AI/ML da LabData FIA: Home Credit

- **Contexto**: O Home Credit, empresa que oferece empréstimos para clientes com pouco ou nenhum histórico de crédito, enfrenta o desafio de equilibrar a concessão de crédito com a minimização da inadimplência.
- A "Dor" do Negócio: A análise manual ou baseada em regras simples de crédito é lenta e falha em capturar padrões complexos de risco, resultando em perda de receita (clientes bons negados) ou aumento de prejuízo (clientes inadimplentes aprovados).
- **Objetivo**: Desenvolver um modelo de Machine Learning preditivo capaz de classificar a probabilidade de inadimplência de um solicitante, permitindo uma tomada de decisão automatizada, mais rápida e mais precisa.
- **Impacto Esperado**: Aumentar a rentabilidade da carteira de crédito e promover a inclusão financeira de forma sustentável para o negócio.


## Etapas

 - Crie um ambiente virtual:
python -m venv venv

 - Ative o ambiente virtual:
 - Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
 - .\venv\Scripts\activate

 - Instale todas as bibliotecas necessárias para trabalhar com ML
pip install -r requirements.txt

 - Entre no site do Kaggle e baixe os dados necessários, seguindo o link abaixo:
https://www.kaggle.com/competitions/home-credit-default-risk/overview

 - Executar o script data_sanization que está dentro da pasta DataPipeline
python .\DataPipeline\data_sanization.py

 - Executar:
python .\DataPipeline\abt_transform.py

 - Executar o script para treinar o modelo:
python .\Model\train.py 
