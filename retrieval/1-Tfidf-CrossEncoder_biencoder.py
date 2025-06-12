#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Experimento: TF–IDF + Bi-Encoder + Cross-Encoder

Este script implementa y evalúa cuatro estrategias distintas para la recuperación de información:

1. TF-IDF para recuperación inicial de top N documentos (pool).
2. Reordenación semántica del pool usando Bi-Encoder (similaridad coseno).
3. Reranking final con Cross-Encoder (más preciso).
4. Evaluación con métricas: Precision, Recall, F1, MRR, nDCG.

implementa y evalúa cuatro estrategias distintas:

Baseline: Este enfoque utiliza una búsqueda basada en k-Nearest Neighbors (kNN) con similitud coseno. Es un método clásico que compara la similitud entre vectores de características para recuperar los documentos más relevantes.

MetadataFilter: Esta estrategia filtra los resultados basándose en metadatos asociados a los documentos, como la categoría y subcategoría. Esto permite afinar los resultados para que sean más relevantes en contextos específicos.

SemanticFilter: Utiliza un filtro semántico que evalúa la similitud coseno entre los embeddings de los documentos y la consulta. Los embeddings son representaciones vectoriales del texto que capturan su significado semántico, permitiendo una comparación más profunda y contextual.

Compression: Esta técnica emplea un modelo de lenguaje para extraer fragmentos relevantes de los documentos recuperados. Esto ayuda a reducir la cantidad de información irrelevante y a centrarse en los segmentos más pertinentes para la consulta.

1. TF–IDF para recuperación inicial de top N documentos (pool).
2. Reordenación semántica del pool usando Bi-Encoder (cosine similarity).
3. Reranking final con Cross-Encoder (más preciso).
4. Evaluación con métricas: Precision, Recall, F1, MRR, nDCG.


Recuperación Inicial con TF-IDF:

Propósito: Obtener un conjunto inicial de documentos que son potencialmente relevantes para la consulta.
Método: Utiliza TF-IDF para calcular la similitud entre la consulta y los documentos en el corpus.
Resultado: Un conjunto de documentos candidatos (pool).
Reordenación Semántica con Bi-Encoder:

Propósito: Mejorar la relevancia semántica de los documentos en el pool inicial.
Método: Utiliza un Bi-Encoder para generar embeddings de la consulta y los documentos, y luego calcula la similitud coseno entre estos embeddings para reordenar los documentos.
Resultado: Un conjunto de documentos reordenados según su similitud semántica con la consulta.
Reranking Final con Cross-Encoder:

Propósito: Refinar aún más el conjunto de documentos para seleccionar los más relevantes.
Método: Utiliza un Cross-Encoder para evaluar la relevancia de cada documento en el pool en relación con la consulta.
Resultado: Un conjunto final de documentos que son los más relevantes para la consulta.
"""

import os, json, logging, math
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import sys
from transformers import pipeline as hf_pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import translate_eu_to_es

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path: str):
    """
    Carga los datos desde un archivo JSON.

    Args:
        path (str): Ruta al archivo JSON.

    Returns:
        tuple: Una tupla que contiene las consultas y el corpus.
    """
    data = json.load(open(path, encoding='utf-8'))
    return data['queries'], data['corpus']

def evaluate(retrieved: List[str], gold: List[str], k: int) -> Dict[str, Any]:

    """
    Evalúa los documentos recuperados en comparación con los documentos relevantes.

    Args:
        retrieved (List[str]): Lista de identificadores de documentos recuperados.
        gold (List[str]): Lista de identificadores de documentos relevantes.
        k (int): Número de documentos a considerar en la evaluación.

    Returns:
        Dict[str, Any]: Diccionario con las métricas de evaluación calculadas.
    """
    seen, uniq = set(), []
    for d in retrieved:
        if d not in seen:
            seen.add(d); uniq.append(d)
        if len(uniq) >= k: break
    tp = [d for d in uniq if d in gold]
    fp = [d for d in uniq if d not in gold]
    fn = [d for d in gold if d not in uniq]
    precision = len(tp)/k if k else 0
    recall    = len(tp)/len(gold) if gold else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
    rr=0
    for i,d in enumerate(uniq,1):
        if d in gold:
            rr = 1/i
            break
    rels = [1 if d in gold else 0 for d in uniq]
    dcg  = sum(rels[i]/math.log2(i+2) for i in range(len(rels)))
    ideal = [1]*min(len(gold),k)
    idcg = sum(ideal[i]/math.log2(i+2) for i in range(len(ideal)))
    ndcg = dcg/idcg if idcg>0 else 0
    return {
        'precision':round(precision,4),
        'recall':   round(recall,4),
        'f1':       round(f1,4),
        'mrr':      round(rr,4),
        'ndcg':     round(ndcg,4),
        'fp':       ";".join(fp),
        'fn':       ";".join(fn),
        'topk':     ",".join(uniq)
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
    for metric,label in [('precision','Precision'),('recall','Recall'),('mrr','MRR')]:
        plt.figure()
        for method in df['method'].unique():
            vals = df[df.method==method].groupby('k')[metric].mean().loc[ks]
            plt.plot(ks, vals, marker='o', label=method.upper())
        plt.title(f'{label}@k')
        plt.xlabel('k'); plt.ylabel(label)
        plt.legend()
        plt.savefig(f"{out_dir}/{metric}_at_k.png", bbox_inches='tight')
        plt.close()

def main():
    data_path = '../evaluacion/dataset_test.json'
    out_dir   = 'resultados/experimento_tfidf_biencoder_xenc'
    pool_size = 10
    top_ks    = [1,3,5,10]

    # Cargar datos
    queries, corpus = load_data(data_path)
    doc_ids = list(corpus.keys())
    docs    = list(corpus.values())
    logger.info(f"Cargados {len(docs)} docs y {len(queries)} queries")

     # Cargar configuración y modelo de traducción
    cfg = cargar_configuracion('../config.yaml')
    translator = hf_pipeline('translation', model=cfg['model']['translation_model'], device='cpu')

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
    X = vectorizer.fit_transform(docs)
    logger.info(f"TF–IDF vectorizado: dimensión {X.shape[1]}")

     # Cargar modelos de embeddings y reranking
    biencoder = SentenceTransformer(cfg['model']['name_embedding'])
    xenc = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')

    records = []
    for q in tqdm(queries, desc="Queries"):
         # Traducir consulta si es necesario
        text = q.get('text_es') or translator([q['text_eu']])[0]['translation_text']
        gold = q.get('relevant_docs',[])

          # Recuperación inicial con TF-IDF
        qv = vectorizer.transform([text])
        sims = (X @ qv.T).toarray().ravel()
        idx  = sims.argsort()[::-1]
        ret_tfidf = [doc_ids[i] for i in idx]
        pool = ret_tfidf[:pool_size]

        # Reordenación semántica con Bi-Encoder
        #Reordenación Semántica con Bi-Encoder
        q_emb = biencoder.encode(text)
        pool_embs = biencoder.encode([corpus[doc_id] for doc_id in pool])

        #Calcular la similitud coseno entre el embedding de la consulta y los embeddings de los documentos en el pool. 
        sem_scores = [float(np.dot(q_emb, emb)) for emb in pool_embs]

        #Reordenar los documentos en el pool basado en las puntuaciones de similitud
        pool_sem = [doc for _, doc in sorted(zip(sem_scores, pool), key=lambda x: x[0], reverse=True)]

        # Reranking final con Cross-Encoder
        pairs = [(text, corpus[doc_id]) for doc_id in pool_sem]
        scores = xenc.predict(pairs, batch_size=16)
        reranked = [doc for _,doc in sorted(zip(scores, pool_sem), key=lambda x: x[0], reverse=True)]

        # Evaluación
        for k in top_ks:
            res_tf   = evaluate(ret_tfidf, gold, k)
            res_tf.update({'method':'tfidf','query':text,'k':k})
            res_sem  = evaluate(pool_sem, gold, k)
            res_sem.update({'method':'biencoder','query':text,'k':k})
            res_xenc = evaluate(reranked, gold, k)
            res_xenc.update({'method':'xenc','query':text,'k':k})
            records += [res_tf, res_sem, res_xenc]

    # Guardar resultados
    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(f"{out_dir}/detalle_tfidf_biencoder_xenc.xlsx", index=False)

    # Resumen de resultados
    summary = (
        df.groupby(['method','k'])[['precision','recall','mrr','f1','ndcg']]
          .mean().round(4).reset_index()
    )
    summary.to_excel(f"{out_dir}/summary_tfidf_biencoder_xenc.xlsx", index=False)
    logger.info("Resumen:\n" + summary.to_markdown())


     # Generar gráficos
    plot_curves(df, out_dir)
    logger.info(f"Gráficas guardadas en {out_dir}")

if __name__=='__main__':
    main()
