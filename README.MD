# Inadimplência Prediction Service

## Descrição

A aplicação de previsão de inadimplência é construída usando Flask e é projetada para avaliar e prever a probabilidade de inadimplência de clientes com base em dados financeiros e pessoais. A aplicação é empacotada em um contêiner Docker, facilitando a distribuição e implantação em diversos ambientes, mantendo a consistência entre desenvolvimento e produção.

## Pré-requisitos

Antes de iniciar, certifique-se de que o Docker está instalado em sua máquina. Se você ainda não tem Docker instalado, você pode baixá-lo e instalá-lo seguindo as instruções no [site oficial do Docker](https://docs.docker.com/get-docker/).

## Instalação

Siga os passos abaixo para configurar a aplicação de previsão de inadimplência em seu ambiente local.

### Clonar o Repositório

Primeiro, clone o repositório onde o código da aplicação está hospedado:

```bash
git clone https://github.com/fspoliveira/Inadimplencia
cd Inadimplencia
```

## Como Usar

Siga as instruções abaixo para colocar a aplicação Model Manager em funcionamento usando Docker.

### Construção da Imagem Docker

Abra o terminal e navegue até o diretório onde está o Dockerfile. Execute o comando abaixo para construir a imagem Docker do projeto:

```bash
docker build -t inadimplencia .
```

Depois execute

```bash
docker run -d -p 5001:5001 inadimplencia
```