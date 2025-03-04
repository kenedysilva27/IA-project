FROM python:3.12

WORKDIR /src

# Copia primeiro o requirements.txt (para cache)
COPY requirements.txt ./

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto (app.py, chat_dataframe.csv, etc.)
COPY . /src

# Expõe a porta do Streamlit
EXPOSE 8501

# Executa o Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
