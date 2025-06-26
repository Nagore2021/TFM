import os
import re
import json
import logging
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, pipeline
from typing import Optional, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuración básica
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# ==============================================
# FUNCIONES AUXILIARES
# ==============================================

def safe_model_name(model_name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', model_name)

# def get_embeddings(model, tokenizer, texts: List[str], batch_size: int = 16) -> torch.Tensor:
#     if not texts:
#         return torch.tensor([])

#     if hasattr(model, 'encode') and tokenizer is None:
#         return model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

#     embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         mask = inputs.attention_mask.unsqueeze(-1)
#         batch_embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
#         embeddings.append(batch_embeddings.cpu())
#     return torch.cat(embeddings)

def get_embeddings(model, texts, batch_size=16):
    if not texts:
        return torch.tensor([])
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

def evaluar_modelo(model_name: str, data: Dict[str, Any], ks: List[int]) -> List[Dict[str, Any]]:
    try:
        logger.info(f"\n{'='*50}\nEvaluando modelo: {model_name}\n{'='*50}")

        ## Detecta si es modelo local (fine-tuneado) o de HuggingFace
        if model_name.startswith("models/") or os.path.isdir(model_name):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.join(project_root, model_name.replace("models/", "models" + os.sep))
            model = SentenceTransformer(model_path, device=device)  # Modelo local
            tokenizer = None
            logger.info(f"Cargado modelo fine-tuneado local desde: {model_path}")
        else:
            model = AutoModel.from_pretrained(model_name).to(device)  # Modelo remoto
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()


        # Extrae corpus (documentos) y queries (consultas)
        corpus_texts = list(data['corpus'].values())
        corpus_ids = list(data['corpus'].keys())

        
        traductor = pipeline("translation", model="Helsinki-NLP/opus-mt-eu-es")
        queries, relevant_documents = [], []
        for q in data['queries']:
            text = q.get('text_es') or traductor(q.get('text_eu',''))[0]['translation_text']
            queries.append(text)
            relevant_documents.append(set(str(d) for d in q.get('relevant_docs', [])))

        # Genera embeddings para corpus y queries
        corpus_embeddings = get_embeddings(model, tokenizer, corpus_texts)
        query_embeddings = get_embeddings(model, tokenizer, queries)

        resultados_por_k = []

        for k in ks:
            per_query = []
            for i, qe in enumerate(query_embeddings):
                  # 1. Calcula similitud coseno entre query y todos los documentos
                sims_query = torch.cosine_similarity(qe.unsqueeze(0), corpus_embeddings, dim=1)

                 # 2. Obtiene los top-K documentos más similares
                topk_idx = torch.topk(sims_query, k).indices.tolist()

                 # 3. Calcula métricas de recuperación
                relevant = relevant_documents[i]
                tp = sum(1 for idx in topk_idx if corpus_ids[idx] in relevant)


                # Precision@K = documentos relevantes encontrados / K
                prec = tp / k

                 # Recall@K = documentos relevantes encontrados / total relevantes
                rec = tp / len(relevant) if relevant else 0

                 # F1@K = media armónica de precisión y recall
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                per_query.append({'precision': prec, 'recall': rec, 'f1': f1})

            avg_precision = float(np.mean([pq['precision'] for pq in per_query]))
            avg_recall = float(np.mean([pq['recall'] for pq in per_query]))
            avg_f1 = float(np.mean([pq['f1'] for pq in per_query]))

            rr = []
            for i, qe in enumerate(query_embeddings):
                sims_query = torch.cosine_similarity(qe.unsqueeze(0), corpus_embeddings, dim=1)
                ordered = torch.argsort(sims_query, descending=True).tolist()
                # Para cada query, encuentra la posición del primer documento relevante
                rank = next((pos for pos, idx in enumerate(ordered) if corpus_ids[idx] in relevant_documents[i]), None)
                rr.append(1.0 / (rank + 1) if rank is not None else 0.0)
            mrr = float(np.mean(rr))

            resultados_por_k.append({
                'model': model_name,
                'k': k,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'mrr': mrr
            })

        return resultados_por_k

    except Exception as e:
        logger.error(f"Error evaluando modelo {model_name}: {str(e)}")
        return []

def main():
    try:
        with open('dataset_test.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        MODELOS = [
    
            "BAAI/bge-m3",
            "jinaai/jina-embeddings-v2-base-es",
            "sentence-transformers/all-MiniLM-L6-v2",
            "Qwen/Qwen3-Embedding-8B",
            "models/bge_m3_epochs/epoch4_MRR0.9717"
        ]

        K_VALUES = [1, 3, 5, 10]
        resultados = []

        for model_name in MODELOS:
            res_modelo = evaluar_modelo(model_name, data, ks=K_VALUES)
            resultados.extend(res_modelo)

        if resultados:
            print("=== RESUMEN COMPARATIVO ===")
            header_fmt = "{:<45} {:<5} {:<12} {:<12} {:<12} {:<12}"
            print(header_fmt.format("Modelo", "K", "Precision", "Recall", "F1", "MRR"))
            row_fmt = "{:<45} {:<5} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}"
            for r in resultados:
                print(row_fmt.format(r['model'], r['k'], r['precision'], r['recall'], r['f1'], r['mrr']))

    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()
