import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

@st.cache_data(show_spinner=False)
def load_documents():
    df = pd.read_csv("C:/Users/55839/Desktop/IA/chat_dataframe.csv")
    loader = DataFrameLoader(df, page_content_column="descricao")
    docs = loader.load()
    for doc, (_, row) in zip(docs, df.iterrows()):
        doc.metadata["correto"] = row["correto"]
        doc.metadata["insumo_1"] = row["insumo_1"]
    return docs

docs = load_documents()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

def custom_prompt(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    knowledge = "\n".join([doc.page_content for doc in results])
    prompt = f"""Use o contexto abaixo para responder à pergunta:

Contexto:
{knowledge}

Pergunta: {query}
Resposta:"""
    return prompt

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

st.title("Interaja com o Chat")
st.write("Digite sua pergunta e veja a resposta gerada com base no contexto da base de conhecimento.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

for speaker, text in st.session_state.conversation:
    role = "user" if speaker == "Usuário" else "assistant"
    with st.chat_message(role):
        st.write(text)

user_input = st.chat_input("Digite sua pergunta...")

if user_input:
    st.session_state.conversation.append(("Usuário", user_input))
    prompt = custom_prompt(user_input)
    messages = [
        SystemMessage(content="Você é um assistente que responde perguntas baseadas no contexto fornecido."),
        HumanMessage(content=prompt)
    ]
    response = chat.invoke(messages)
    st.session_state.conversation.append(("Assistente", response.content))
