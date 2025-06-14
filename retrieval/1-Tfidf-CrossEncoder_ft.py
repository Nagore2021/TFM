#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 1b: TF–IDF + Cross-Encoder sobre colección FINE-TUNEADA

1. TF–IDF para recuperar top N documentos (pool size).
2. Reranking de ese pool con Cross-Encoder.
3. Evaluación con métricas: Precision, Recall, F1, MRR, nDCG.
4. Gráficas de Precision@k, Recall@k y MRR@k.

Precision: Proporción de documentos relevantes recuperados.
Recall: Proporción de documentos relevantes recuperados en relación con el total de documentos relevantes.
F1: Media armónica de Precision y Recall.
MRR (Mean Reciprocal Rank): Promedio del inverso del rango de los documentos relevantes.
nDCG (Normalized Discounted Cumulative Gain): Métrica que considera la relevancia y la posición de los documentos recuperados.
"""

import os, json, logging, math
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import sys
from transformers import pipeline as hf_pipeline

top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(top)

from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import translate_eu_to_es 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path: str):
    data = json.load(open(path, encoding='utf-8'))
    return data['queries'], data['corpus']

def evaluate(retrieved: List[str], gold: List[str], k: int) -> Dict[str, Any]:
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
   Se generan gráficas de Precision@k, Recall@k y MRR@k para visualizar el rendimiento del sistema.
    """
    os.makedirs(out_dir, exist_ok=True)
    ks = sorted(df['k'].unique())
    for metric,label in [('precision','Precision'),('recall','Recall'),('mrr','MRR')]:
        plt.figure()
        for method in ['tfidf','xenc']:
            vals = df[df.method==method].groupby('k')[metric].mean().loc[ks]
            plt.plot(ks, vals, marker='o', label=method.upper())
        plt.title(f'{label}@k')
        plt.xlabel('k'); plt.ylabel(label)
        plt.legend()
        plt.savefig(f"{out_dir}/{metric}_at_k.png", bbox_inches='tight')
        plt.close()

def main():
    data_path = '../evaluacion/dataset_test.json'
    out_dir   = 'resultados/experimento_tfidf_biencoder_xenc_ft'
    pool_size = 10
    top_ks    = [1,3,5,10]

    queries, corpus = load_data(data_path)
    doc_ids = list(corpus.keys())
    docs    = list(corpus.values())
    logger.info(f"Cargados {len(docs)} docs y {len(queries)} queries")

    cfg = cargar_configuracion('../config.yaml')

      #Preparar traductor Euskera→Español

    translator = hf_pipeline('translation',
                              model=cfg['model']['translation_model'],
                              device='cpu')

   

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
    X = vectorizer.fit_transform(docs)
    logger.info(f"TF–IDF vectorizado: dimensión {X.shape[1]}")

    xenc = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')

    records = []
    for q in tqdm(queries, desc="Queries"):

        text = q.get('text_es')
        if not text and q.get('text_eu'):
            text = translator([q['text_eu']])[0]['translation_text']
       
        gold = q.get('relevant_docs',[])

        qv = vectorizer.transform([text])
        sims = (X @ qv.T).toarray().ravel()
        idx  = sims.argsort()[::-1]
        ret_tfidf = [doc_ids[i] for i in idx]
        pool = ret_tfidf[:pool_size]

        pairs = [(text, corpus[doc_id]) for doc_id in pool]
        scores = xenc.predict(pairs, batch_size=16)
        reranked = [doc for _,doc in sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)]

        for k in top_ks:
            res_tf   = evaluate(ret_tfidf, gold, k)
            res_tf.update({'method':'tfidf','query':text,'k':k})
            res_xenc = evaluate(reranked, gold, k)
            res_xenc.update({'method':'xenc','query':text,'k':k})
            records += [res_tf, res_xenc]

    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)

   
    df.to_excel(f"{out_dir}/detalle_1bft.xlsx", index=False)
    summary = (
        df.groupby(['method','k'])
          [['precision','recall','mrr','f1','ndcg']]
          .mean().round(4).reset_index()
    )
   # summary.to_json(f"{out_dir}/summary_1bft.json", orient='records')
    summary.to_excel(f"{out_dir}/summary_1bft.xslx", index=False)
   
    logger.info("Resumen:\n" + summary.to_markdown())

    plot_curves(df, out_dir)
    logger.info(f"Gráficas guardadas en {out_dir}")

if __name__=='__main__':
    main()
