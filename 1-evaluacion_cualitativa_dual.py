# evaluacion_cualitativa_dual.py
"""
Compara respuestas generadas desde dos colecciones ChromaDB:
- Una indexada con modelo base (BAAI/bge-m3): documentos_osakidetza
- Otra indexada con modelo fine-tuneado: documentos_finetuneado
Este script evalúa de forma cualitativa si el modelo especializado mejora la relevancia de los resultados.
"""


import os
import sys
import logging
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from rich import print
from datetime import datetime

top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(top)

from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import format_result_snippets
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
cfg = cargar_configuracion("../config.yaml")
paths = cfg["paths"]

# Funciones

def cargar_coleccion(path_db, nombre):
    client = PersistentClient(path=path_db)
    return client.get_collection(nombre)

modelo_base = SentenceTransformer(cfg["model"]["name_embedding"])
modelo_finetune = SentenceTransformer(os.path.join(paths["model_path"], cfg["model"]["name_finetuning"]))

col_base = cargar_coleccion(paths["chroma_db_path"], cfg["collection"]["name"])
col_finetuned = cargar_coleccion(paths["chroma_db_path"], cfg["collection"]["name_finetuneado"])

def comparar_respuestas(query: str, k: int = 3) -> str:
    emb_base = modelo_base.encode(query, normalize_embeddings=True).tolist()
    emb_fine = modelo_finetune.encode(query, normalize_embeddings=True).tolist()

    output = f"\n[Query] {query}\n"

    if col_base.count() == 0:
        output += "[BASE vacía]\n"
    else:
        r_base = col_base.query(query_embeddings=[emb_base], n_results=k)
        output += "--- Resultados BASE ---\n"
        output += format_result_snippets(r_base) + "\n"

    if col_finetuned.count() == 0:
        output += "[FINE-TUNEADA vacía]\n"
    else:
        r_fine = col_finetuned.query(query_embeddings=[emb_fine], n_results=k)
        output += "--- Resultados FINE-TUNEADA ---\n"
        output += format_result_snippets(r_fine) + "\n"

    return output

# Main
if __name__ == "__main__":
    queries = [
        "tratamiento del cáncer de mama",
        "síntomas del ictus",
        "cuidados para la diabetes",
        "cómo prevenir el asma",
        "efectos de la epilepsia",
        "detección del cáncer de pulmón",
        "manejo de la depresión"
    ]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("resultados", f"comparativa_cualitativa_{now}.txt")
    os.makedirs("resultados", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in queries:
            result_text = comparar_respuestas(q)
            print(result_text)
            f.write(result_text + "\n")
            f.write("="*80 + "\n")

    logger.info(f"Resultados guardados en {output_path}")
