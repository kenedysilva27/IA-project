from flask import Flask, render_template, request, jsonify
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Langchain/OpenAI modules not found. OpenAI embeddings will not be available.")
    LANGCHAIN_AVAILABLE = False

load_dotenv()

app = Flask(__name__)

CSV_FILE_PATH = "pacientes.csv"

METADATA_COLUMNS = [
    "Plano_Saude",
    "Score_Comorbidade_Charlson",
    "Telemonitoramento",
    "Resultado_Exames_Importantes",
    "Procedimentos_Realizados"
]

global_vectorstore = None
global_docs = []
global_df = None

class Document:
    """Simple class to represent a document"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class CustomVectorStore:
    """Custom vector search system - WITHOUT VectorDB"""

    def __init__(self, documents, use_openai=True):
        self.documents = documents
        self.use_openai = use_openai and LANGCHAIN_AVAILABLE
        self.embeddings_model = None
        self.doc_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        if self.use_openai:
            try:
                self.embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
                self._create_openai_embeddings()
            except Exception as e:
                print(f"⚠️ Could not use OpenAI embeddings: {e}")
                print("🔄 Falling back to TF-IDF...")
                self.use_openai = False
                self._create_tfidf_embeddings()
        else:
            self._create_tfidf_embeddings()

    def _create_openai_embeddings(self):
        """Creates embeddings using OpenAI"""
        try:
            texts = [doc.page_content for doc in self.documents]
            self.doc_embeddings = self.embeddings_model.embed_documents(texts)
            print("✅ OpenAI embeddings created successfully!")
        except Exception as e:
            print(f"❌ Error creating OpenAI embeddings: {e}")
            self.use_openai = False
            self._create_tfidf_embeddings()

    def _create_tfidf_embeddings(self):
        """Creates embeddings using TF-IDF as fallback"""
        try:
            texts = [doc.page_content for doc in self.documents]
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print("✅ TF-IDF embeddings created successfully!")
        except Exception as e:
            print(f"❌ Error creating TF-IDF embeddings: {e}")

    def similarity_search(self, query, k=3):
        """Performs similarity search"""
        try:
            if self.use_openai and self.embeddings_model and self.doc_embeddings is not None:
                return self._openai_similarity_search(query, k)
            else:
                return self._tfidf_similarity_search(query, k)
        except Exception as e:
            print(f"❌ Error in search: {e}")
            return self.documents[:k]

    def _openai_similarity_search(self, query, k):
        """Search using OpenAI embeddings"""
        try:
            query_embedding = self.embeddings_model.embed_query(query)

            similarities = []
            for i, doc_emb in enumerate(self.doc_embeddings):
                similarity = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(doc_emb).reshape(1, -1))[0][0]
                similarities.append((similarity, i))

            similarities.sort(reverse=True)
            top_docs = []
            for _, idx in similarities[:k]:
                top_docs.append(self.documents[idx])

            return top_docs
        except Exception as e:
            print(f"❌ Error in OpenAI search: {e}")
            return self._tfidf_similarity_search(query, k)

    def _tfidf_similarity_search(self, query, k):
        """Search using TF-IDF"""
        try:
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                return self.documents[:k]

            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            top_indices = similarities.argsort()[-k:][::-1]
            top_docs = [self.documents[i] for i in top_indices]

            return top_docs
        except Exception as e:
            print(f"❌ Error in TF-IDF search: {e}")
            return self.documents[:k]

def filter_metadata_manually(metadata_dict):
    """Function to manually filter metadata"""
    filtered = {}
    for key, value in metadata_dict.items():
        if pd.isna(value):
            filtered[key] = "N/A"
        elif isinstance(value, (str, int, float, bool)):
            filtered[key] = str(value)
        elif hasattr(value, '__iter__') and not isinstance(value, str):
            filtered[key] = str(value)
        else:
            filtered[key] = str(value)
    return filtered

def load_documents_from_dataframe():
    """Loads documents from the CSV file"""
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"🚫 Error: The file '{CSV_FILE_PATH}' was not found.")
        return [], pd.DataFrame()

    df['page_content'] = df.apply(lambda row: (
        f"Paciente ID: {row['ID_Paciente']}. Idade: {row['Idade']}, Sexo: {row['Sexo']}. "
        f"Diagnóstico Principal: {row['Diagnostico_Principal']}. "
        f"Comorbidades: {row.get('Comorbidades', 'Nenhuma') if pd.notna(row.get('Comorbidades')) else 'Nenhuma'}. "
        f"Duração Internação: {row['Duracao_Internacao_Dias']} dias. "
        f"Internações Últ. Ano: {row['Qtd_Internacoes_Ult_Ano']}. "
        f"Procedimentos Realizados: {row.get('Procedimentos_Realizados', 'Nenhum') if pd.notna(row.get('Procedimentos_Realizados')) else 'Nenhum'}. "
        f"Necessitou Homecare: {'Sim' if row['Necessidade_Homecare'] else 'Não'}. "
        f"Medicação Pós-Alta: {row.get('Medicacao_Pos_Alta', 'Nenhuma') if pd.notna(row.get('Medicacao_Pos_Alta')) else 'Nenhuma'}. "
        f"Readmissão em 30 dias: {'Sim' if row['Readmissao_30_Dias'] else 'Não'}."
    ), axis=1)

    docs = []
    for i, row in df.iterrows():
        temp_metadata = {}

        for col in METADATA_COLUMNS:
            if col in df.columns:
                value = row[col]
                if pd.isna(value):
                    temp_metadata[col] = "N/A"
                else:
                    temp_metadata[col] = str(value)

        filtered_metadata = filter_metadata_manually(temp_metadata)
        doc = Document(page_content=row['page_content'], metadata=filtered_metadata)
        docs.append(doc)

    return docs, df

def initialize_vectorstore():
    """Initializes the custom search system"""

    print("🧹 Initializing custom system (without VectorDB)!")

    docs, df = load_documents_from_dataframe()

    if df.empty:
        print("🚫 No data loaded. Exiting initialization.")
        return None, [], None

    try:
        print("🔄 Creating custom search system...")
        vectorstore = CustomVectorStore(docs, use_openai=True)
        print("✅ Custom search system created successfully!")

    except Exception as e:
        print(f"❌ Error creating search system: {e}")
        print("🔄 Trying TF-IDF mode...")

        try:
            vectorstore = CustomVectorStore(docs, use_openai=False)
            print("✅ TF-IDF search system created successfully!")
        except Exception as e2:
            print(f"❌ Complete failure: {e2}")
            return None, [], None

    return vectorstore, docs, df

def custom_prompt(vectorstore, query: str) -> str:
    """Creates a custom prompt with context from similar cases"""
    try:
        results = vectorstore.similarity_search(query, k=3)
        knowledge = ""

        for i, doc in enumerate(results):
            knowledge += f"--- Caso Similar {i+1} ---\n"
            knowledge += f"Detalhes do Paciente: {doc.page_content}\n"

            if doc.metadata:
                knowledge += f"Plano de Saúde: {doc.metadata.get('Plano_Saude', 'N/A')}\n"
                knowledge += f"Score Charlson: {doc.metadata.get('Score_Comorbidade_Charlson', 'N/A')}\n"
                knowledge += f"Procedimentos: {doc.metadata.get('Procedimentos_Realizados', 'N/A')}\n"
            knowledge += "\n"

        prompt = f"""Você é um assistente especializado em dados de saúde e históricos de pacientes.
Use o contexto dos casos de pacientes similares abaixo para responder à pergunta.
Se a pergunta não puder ser respondida com o contexto fornecido, diga que não tem informações suficientes.

Contexto de Casos de Pacientes Similares:
{knowledge}

Pergunta: {query}

Resposta Detalhada:"""
        return prompt

    except Exception as e:
        print(f"Error fetching similar cases: {e}")
        return f"Pergunta: {query}\n\nResposta: Não foi possível buscar casos similares devido a um erro técnico."

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages and returns AI responses."""
    if not global_vectorstore or global_df.empty:
        return jsonify({"response": "Sistema não inicializado. Por favor, recarregue a página ou verifique os logs."}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "Mensagem vazia."}), 400

    try:
        enhanced_prompt = custom_prompt(global_vectorstore, user_message)

        if not LANGCHAIN_AVAILABLE:
            return jsonify({"response": "Langchain/OpenAI modules not available. Cannot generate AI response."}), 500

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )

        messages = [
            SystemMessage(content="Você é um assistente especializado em análise de dados médicos e de saúde."),
            HumanMessage(content=enhanced_prompt)
        ]

        response = llm(messages)
        response_content = response.content

        return jsonify({"response": response_content})

    except Exception as e:
        print(f"Error processing chat message: {e}")
        return jsonify({"response": f"❌ Erro ao processar sua pergunta: {str(e)}"}), 500

@app.route('/stats')
def get_stats():
    """Returns statistics about the loaded data."""
    if not global_df.empty:
        total_patients = len(global_df)
        readmission_rate = (global_df['Readmissao_30_Dias'].sum() / total_patients * 100) if total_patients > 0 else 0
        avg_stay = global_df['Duracao_Internacao_Dias'].mean() if total_patients > 0 else 0
        homecare_rate = (global_df['Necessidade_Homecare'].sum() / total_patients * 100) if total_patients > 0 else 0
        
        stats = {
            "total_patients": total_patients,
            "readmission_rate": f"{readmission_rate:.1f}%",
            "avg_stay": f"{avg_stay:.1f}",
            "homecare_rate": f"{homecare_rate:.1f}%",
            "loaded_documents": len(global_docs),
            "data_fields": len(global_df.columns)
        }
        return jsonify(stats)
    return jsonify({
        "total_patients": 0, "readmission_rate": "0.0%", "avg_stay": "0.0",
        "homecare_rate": "0.0%", "loaded_documents": 0, "data_fields": 0
    }), 200

with app.app_context():
    global_vectorstore, global_docs, global_df = initialize_vectorstore()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

