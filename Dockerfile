# Use uma imagem base do Python
FROM python:3.9-slim

# Defina o diretório de trabalho como /app
WORKDIR /app

# Copie o arquivo de requisitos para o contêiner
COPY requirements.txt .

# Instale as dependências do Python a partir do arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie os arquivos necessários para o contêiner
COPY . /app

# Comando para iniciar o seu aplicativo Flask na porta 5001 e somente localmente
CMD ["python", "app.py", "--host=127.0.0.1", "--port=5001"]