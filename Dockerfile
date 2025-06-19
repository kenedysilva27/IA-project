# Use a imagem oficial do Python como base
# Recomendo Python 3.9-slim-buster ou 3.10-slim-buster para compatibilidade e tamanho
# Python 3.12 pode ter algumas mudanças que podem afetar bibliotecas antigas,
# mas se suas dependências funcionam bem, pode manter.
FROM python:3.11-slim-buster

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências Python
RUN pip install -r requirements.txt

# Copie todo o código da aplicação para o diretório de trabalho
# Isso inclui app.py, a pasta templates/ e o arquivo pacientes.csv
COPY . .

# Exponha a porta que a aplicação Flask usa (porta padrão do Flask)
EXPOSE 5000

# Defina as variáveis de ambiente para o Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production 

# Defina o comando para executar a aplicação Flask
CMD ["flask", "run"]
