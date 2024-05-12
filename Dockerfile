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

# Comando para iniciar o seu aplicativo Flask
CMD ["python", "app.py"]
