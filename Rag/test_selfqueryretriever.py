# test_selfqueryretriever.py
"""
Script de prueba sencillo para SelfQueryRetriever (LangChain)
Este script muestra cómo usar SelfQueryRetriever para convertir una consulta en lenguaje natural
en una búsqueda semántica con filtros de metadatos.
Requisitos:
  pip install langchain chromadb sentence-transformers openai
"""
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever

# 1. Configuración de embeddings y almacenamiento
persist_directory = "./chroma_db"  # carpeta con tu ChromaDB persistente
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# 2. Inicializar LLM para parsing de consulta a filtros
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

# 3. Definir información de metadatos de tu colección
#    (campo, descripción)
metadata_field_info = [
    ("category", "categoría del documento"),
    ("date", "fecha de publicación en formato YYYY-MM-DD"),
    ("author", "autor del documento")
]

# 4. Crear SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    embeddings=embeddings,
    vectorstore=vectorstore,
    document_content_description="artículos médicos",
    metadata_field_info=metadata_field_info
)

# 5. Hacer una consulta de prueba
def main():
    query = "¿Cómo puedo prevenir el cáncer de mama?"
    docs = retriever.get_relevant_documents(query)
    if docs:
        print("--- Top resultado ---")
        print(docs[0].page_content)
    else:
        print("No se encontraron documentos relevantes.")

if __name__ == "__main__":
    main()
