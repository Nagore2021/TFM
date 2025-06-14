
"""
Script Experimento: TF–IDF + Bi-Encoder + Cross-Encoder

Este script implementa y evalúa cuatro estrategias distintas para la recuperación de información:

1. TF-IDF para recuperación inicial de top N documentos (pool).
2. Reordenación semántica del pool usando Bi-Encoder (similaridad coseno).
3. Reranking final con Cross-Encoder (más preciso).
4. Evaluación con métricas: Precision, Recall, F1, MRR, nDCG.

Ahora incluye información detallada de chunks para consistencia con otros experimentos.
"""

import os, json, logging, math
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import sys
import torch
from transformers import pipeline as hf_pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import translate_eu_to_es, limpiar_texto_excel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Carga los datos desde un archivo JSON.

    Args:
        path (str): Ruta al archivo JSON.

    Returns:
        tuple: Una tupla que contiene las consultas y el corpus.
    """
    data = json.load(open(path, encoding='utf-8'))
    return data['queries'], data['corpus']

def load_chunk_metadata(corpus: Dict[str, str]) -> Dict[str, Dict]:
    """
    Simula metadatos de chunks para mantener consistencia.
    En un sistema real, estos metadatos vendrían de la base de datos de chunks.
    
    Args:
        corpus: Diccionario con doc_id -> texto completo
        
    Returns:
        Dict con metadatos simulados por documento
    """
    metadata = {}
    for doc_id, text in corpus.items():
        # Simulamos que cada documento tiene un chunk principal (chunk0)
        chunk_id = f"{doc_id}_chunk0"
        metadata[doc_id] = {
            'chunk_id': chunk_id,
            'chunk_position': 0,
            'document_id': doc_id,
            'chunk_text': text[:500] + "..." if len(text) > 500 else text  # Limitamos para Excel
        }
    return metadata

def evaluate_with_metadata(retrieved: List[str], gold: List[str], k: int, 
                          metadata: Dict[str, Dict], method: str, 
                          query_text: str) -> Dict[str, Any]:
    """
    Evalúa los documentos recuperados incluyendo información de chunks.

    Args:
        retrieved (List[str]): Lista de IDs de documentos recuperados.
        gold (List[str]): Lista de IDs de documentos relevantes.
        k (int): Número de documentos a considerar.
        metadata (Dict): Metadatos de chunks por documento.
        method (str): Método de recuperación utilizado.
        query_text (str): Texto de la consulta.

    Returns:
        Dict[str, Any]: Diccionario con las métricas de evaluación y metadatos.
    """
    # Obtener únicos manteniendo orden
    seen, uniq = set(), []
    for d in retrieved:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
        if len(uniq) >= k:
            break
    
    # Calcular métricas
    tp = [d for d in uniq if d in gold]
    fp = [d for d in uniq if d not in gold]
    fn = [d for d in gold if d not in uniq]
    
    precision = len(tp) / k if k else 0
    recall = len(tp) / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MRR (Mean Reciprocal Rank)
    rr = 0
    for i, d in enumerate(uniq, 1):
        if d in gold:
            rr = 1 / i
            break
    
    # nDCG (Normalized Discounted Cumulative Gain)
    rels = [1 if d in gold else 0 for d in uniq]
    dcg = sum(rels[i] / math.log2(i + 2) for i in range(len(rels)))
    ideal = [1] * min(len(gold), k)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # Información del mejor chunk (primer documento recuperado)
    chunk_position = ""
    chunk_id = ""
    chunk_text = ""
    
    if uniq:
        best_doc = uniq[0]
        if best_doc in metadata:
            meta = metadata[best_doc]
            chunk_position = meta.get('chunk_position', '')
            chunk_id = meta.get('chunk_id', '')
            chunk_text = limpiar_texto_excel(meta.get('chunk_text', ''))
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'mrr': round(rr, 4),
        'ndcg': round(ndcg, 4),
        'fp': ";".join(fp),
        'fn': ";".join(fn),
        'topk': ",".join(uniq),
        'method': method,
        'query': query_text,
        'k': k,
        'chunk_position': chunk_position,
        'chunk_id': chunk_id,
        'chunk_text': chunk_text
    }

def plot_curves(df: pd.DataFrame, out_dir: str):
    """
    Genera gráficos de las métricas de evaluación para diferentes valores de k.

    Args:
        df (pd.DataFrame): DataFrame que contiene los resultados de las evaluaciones.
        out_dir (str): Directorio donde se guardarán los gráficos.
    """
    os.makedirs(out_dir, exist_ok=True)
    ks = sorted(df['k'].unique())
    
    for metric, label in [('precision', 'Precision'), ('recall', 'Recall'), ('mrr', 'MRR'), ('ndcg', 'nDCG')]:
        plt.figure(figsize=(10, 6))
        for method in df['method'].unique():
            vals = df[df.method == method].groupby('k')[metric].mean().loc[ks]
            plt.plot(ks, vals, marker='o', label=method.upper(), linewidth=2)
        
        plt.title(f'{label}@k', fontsize=14)
        plt.xlabel('k', fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/{metric}_at_k.png", bbox_inches='tight', dpi=300)
        plt.close()

def main():
    """Función principal del experimento."""
    # Configuración
    data_path = '../evaluacion/dataset_test.json'
    out_dir = 'resultados/experimento_tfidf_biencoder_xenc'
    pool_size = 10
    top_ks = [1, 3, 5, 10]

    # Cargar datos
    queries, corpus = load_data(data_path)
    doc_ids = list(corpus.keys())
    docs = list(corpus.values())
    
    # Cargar metadatos simulados
    metadata = load_chunk_metadata(corpus)
    
    logger.info(f"Cargados {len(docs)} documentos y {len(queries)} consultas")

    # Cargar configuración y modelo de traducción
    cfg = cargar_configuracion('../config.yaml')
    translator = hf_pipeline('translation', model=cfg['model']['translation_model'], device='cpu')

    # Vectorización TF-IDF
    logger.info("Iniciando vectorización TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=15000,
        stop_words=None,  # Mantenemos las stop words para español médico
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(docs)
    logger.info(f"TF–IDF vectorizado: dimensión {X.shape}")

    # Cargar modelos de embeddings y reranking
    logger.info("Cargando modelos de embeddings...")
    biencoder = SentenceTransformer(cfg['model']['name_embedding'])
    
    # Determinar dispositivo
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {DEVICE}")
    
    xenc = CrossEncoder(cfg['model']['name_cross_encoder'], device=DEVICE)
    
    # Mover modelos a GPU si está disponible
    if hasattr(biencoder, 'to'):
        biencoder.to(DEVICE)

    records = []
    
    # Procesamiento de consultas
    for q in tqdm(queries, desc="Procesando consultas"):
        # Traducir consulta si es necesario
        text = q.get('text_es')
        if not text:
            text = translator([q['text_eu']])[0]['translation_text']
        
        gold = q.get('relevant_docs', [])
        if not gold:
            logger.warning(f"Consulta sin documentos relevantes: {text[:50]}...")
            continue

        # === 1. RECUPERACIÓN INICIAL CON TF-IDF ===
        qv = vectorizer.transform([text])
        sims = (X @ qv.T).toarray().ravel()
        idx = sims.argsort()[::-1]
        ret_tfidf = [doc_ids[i] for i in idx]
        
        # === 2. REORDENACIÓN SEMÁNTICA CON BI-ENCODER ===
        pool = ret_tfidf[:pool_size]
        
        # Generar embeddings
        q_emb = biencoder.encode(text, convert_to_tensor=True)
        pool_texts = [corpus[doc_id] for doc_id in pool]
        pool_embs = biencoder.encode(pool_texts, convert_to_tensor=True)
        
        # Calcular similitud coseno
        if hasattr(q_emb, 'cpu'):  # Si estamos usando GPU
            sem_scores = np.array([
                float(torch.cosine_similarity(q_emb.unsqueeze(0), emb.unsqueeze(0)))
                for emb in pool_embs
            ])
        else:
            sem_scores = [float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))) 
                         for emb in pool_embs]
        
        # Reordenar pool por similitud semántica
        pool_sem = [doc for _, doc in sorted(zip(sem_scores, pool), key=lambda x: x[0], reverse=True)]

        # === 3. RERANKING FINAL CON CROSS-ENCODER ===
        pairs = [(text, corpus[doc_id]) for doc_id in pool_sem]
        scores = xenc.predict(pairs, batch_size=8)  # Batch más pequeño para evitar memoria
        reranked = [doc for _, doc in sorted(zip(scores, pool_sem), key=lambda x: x[0], reverse=True)]

        # === 4. EVALUACIÓN PARA CADA K ===
        for k in top_ks:
            # TF-IDF
            res_tf = evaluate_with_metadata(ret_tfidf, gold, k, metadata, 'tfidf', text)
            records.append(res_tf)
            
            # Bi-Encoder
            res_sem = evaluate_with_metadata(pool_sem, gold, k, metadata, 'biencoder', text)
            records.append(res_sem)
            
            # Cross-Encoder
            res_xenc = evaluate_with_metadata(reranked, gold, k, metadata, 'cross_encoder', text)
            records.append(res_xenc)

    # === 5. GUARDAR RESULTADOS ===
    logger.info("Guardando resultados...")
    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    
    # Archivo detallado (con todas las columnas)
    df.to_excel(f"{out_dir}/detalle_tfidf_biencoder_xenc.xlsx", index=False)
    logger.info(f"Archivo detallado guardado: {out_dir}/detalle_tfidf_biencoder_xenc.xlsx")

    # Archivo resumen (métricas promediadas)
    summary = (
        df.groupby(['method', 'k'])[['precision', 'recall', 'mrr', 'f1', 'ndcg']]
        .mean().round(4).reset_index()
    )
    summary.to_excel(f"{out_dir}/summary_tfidf_biencoder_xenc.xlsx", index=False)
    logger.info("Resumen de resultados:")
    print(summary.to_markdown(index=False))

    # === 6. GENERAR GRÁFICOS ===
    plot_curves(df, out_dir)
    logger.info(f"Gráficos guardados en {out_dir}")

    # === 7. ESTADÍSTICAS FINALES ===
    logger.info(f"\n=== ESTADÍSTICAS FINALES ===")
    logger.info(f"Total de consultas procesadas: {len(queries)}")
    logger.info(f"Total de evaluaciones: {len(records)}")
    logger.info(f"Métodos evaluados: {df['method'].unique()}")
    logger.info(f"Valores de k evaluados: {sorted(df['k'].unique())}")
    
    # Mejor rendimiento por método
    best_results = df.groupby('method')[['precision', 'recall', 'f1', 'mrr', 'ndcg']].mean()
    logger.info(f"\nMejor rendimiento promedio por método:")
    print(best_results.round(4).to_string())

if __name__ == '__main__':
    main()