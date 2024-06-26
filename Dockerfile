# Use a imagem base do Python
FROM python:3.9-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos do diretório atual para o diretório de trabalho no contêiner
COPY . .

# Copie o arquivo de requisitos para o contêiner
COPY requirements.txt .

# Instala as dependências do aplicativo Flask a partir do arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define variáveis de ambiente para Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Expõe a porta do aplicativo Flask
EXPOSE 5001

# Comando para executar o aplicativo Flask quando o contêiner for iniciado
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]