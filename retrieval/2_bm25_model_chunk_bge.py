"""
Script Experimento 2: BM25 + Bi-Encoder + Cross-Encoder a Nivel de Chunks

Pipeline híbrido para evaluación de chunk retrieval usando ChromaDB:

METODOLOGÍA A NIVEL DE CHUNKS:
1. BM25 para recuperación inicial de chunks relevantes
2. Reordenación semántica usando Bi-Encoder (similaridad coseno)  
3. Reranking final con Cross-Encoder (máxima precisión)
4. Evaluación con métricas: Precision, Recall, F1, MRR, nDCG

OBJETIVO: 
- Encontrar el mejor CHUNK (no documento) para contexto RAG
- Evaluación sobre chunks reales de ChromaDB (300 caracteres)
- Pipeline progresivo optimizado para chunk retrieval
"""

import os, json, logging, math
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import sys
import torch
import spacy
import yaml
import chromadb
from collections import defaultdict
from transformers import pipeline as hf_pipeline

# Cargar modelo spaCy para tokenización en español
nlp = spacy.load("es_core_news_sm")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.chroma_utils import normalize_doc_id, normalizar_doc, limpiar_texto_excel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import translate_eu_to_es, limpiar_texto_excel

# Cargar modelo spaCy para tokenización en español
nlp = spacy.load("es_core_news_sm")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Carga queries para evaluación desde dataset_test.json
    
    Args:
        path (str): Ruta al archivo JSON con queries y ground truth
        
    Returns:
        tuple: (queries, corpus) - solo queries se usan, corpus se mantiene por compatibilidad
    """
    data = json.load(open(path, encoding='utf-8'))
    return data['queries'], data.get('corpus', {})

def load_chunk_metadata(corpus: Dict[str, str]) -> Dict[str, Dict]:
    """
    Simula metadatos de chunks para mantener consistencia con otros experimentos.
    En este experimento trabajamos con documentos completos.
    
    Args:
        corpus: Diccionario con doc_id -> texto completo
        
    Returns:
        Dict con metadatos simulados por documento
    """
    metadata = {}
    for doc_id, text in corpus.items():
        chunk_id = f"{doc_id}_chunk0"  # Simulamos documento completo como chunk único
        metadata[doc_id] = {
            'chunk_id': chunk_id,
            'chunk_position': "1/1",  # Documento completo
            'document_id': doc_id,
            'chunk_text': text[:500] + "..." if len(text) > 500 else text
        }
    return metadata

def evaluate_with_metadata(retrieved: List[str], gold: List[str], k: int, 
                          metadata: Dict[str, Dict], method: str, 
                          query_text: str) -> Dict[str, Any]:
    """
    Evalúa los documentos recuperados incluyendo información de chunks.
   
    Función que calcula métricas clave: precisión, recall, F1, MRR y nDCG para top-k documentos recuperados.
    
    Args:
        retrieved (List[str]): Lista de IDs de documentos recuperados en orden de relevancia
        gold (List[str]): Lista de IDs de documentos relevantes (ground truth)
        k (int): Número de documentos a considerar en el top-k
        
    Returns:
        Dict con métricas: precision, recall, f1, mrr, ndcg, fp, fn, topk
    
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

def load_queries(json_path, translator):
    """
    Traduce queries desde euskera si es necesario y normaliza su texto con spaCy.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    queries, relevant = [], {}
    for q in data['queries']:
        text = q.get("text_es") or translator([q["text_eu"]])[0]['translation_text']
        if text:
            norm = normalizar_doc(text)
            queries.append(norm)
            relevant[norm] = [normalize_doc_id(d) for d in q['relevant_docs']]
    return queries, relevant


class BM25PipelineEvaluator:
    def __init__(self, config_path, mode):
        """
        Evaluador de pipeline BM25 + Bi-Encoder + Cross-Encoder a nivel de chunks.
        """
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.mode = mode
        self.records = []
        self.doc_to_best_chunk = {}
        self.client = chromadb.PersistentClient(path=self.cfg['paths']['chroma_db_path'])
        self._load_models()

    def _load_models(self):
        """
        Carga modelos de recuperación semántica y traducción.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {device}")
        
        model_name = self.cfg['model'][f'name_{self.mode}']
        self.biencoder = SentenceTransformer(model_name, device=device)
        self.cross_encoder = CrossEncoder(self.cfg['model']['name_cross_encoder'], device=device)
        self.translator = hf_pipeline('translation', model=self.cfg['model']['translation_model'], 
                                    device=0 if device=="cuda" else -1)

    def load_collection(self):
        """
        Carga la colección de chunks de ChromaDB.
        """
        collection_name = self.cfg['collection']['name'] if self.mode == 'embedding' else self.cfg['collection']['name_finetuneado']
        logger.info(f"Cargando colección: {collection_name}")
        
        self.collection = self.client.get_collection(collection_name)
        data = self.collection.get(include=['documents', 'metadatas'])
        
        # Acceso rápido a contenido de chunks
        self.docs_raw = dict(zip([m['chunk_id'] for m in data['metadatas'] if 'chunk_id' in m], 
                               data['documents']))
        
        # Normalizar documentos para BM25
        self.docs_norm = [normalizar_doc(doc) for doc in data['documents']]
        
        # IDs de documentos
        self.doc_ids = [normalize_doc_id(m['document_id']) for m in data['metadatas']]
        
        # Metadatos completos
        self.metadatas = {
            m['chunk_id']: m for m in data['metadatas'] 
            if m.get('chunk_id') and m.get('document_id')
        }
        
        # Mapeo para BM25
        self.chunk_ids = [m['chunk_id'] for m in data['metadatas'] if 'chunk_id' in m]
        
        # Mapeo documento -> chunk0 (fallback)
        self.doc_id_to_chunk0 = {}
        for m in data['metadatas']:
            if 'chunk_id' in m and 'document_id' in m:
                doc_id = normalize_doc_id(m['document_id'])
                chunk_id = m['chunk_id']
                if doc_id not in self.doc_id_to_chunk0 or "_chunk0" in chunk_id:
                    self.doc_id_to_chunk0[doc_id] = chunk_id
        
        # Inicializar BM25 sobre chunks normalizados
        self.bm25 = BM25Okapi([[t.text.lower() for t in nlp(d)] for d in self.docs_norm])
        
        logger.info(f"Colección cargada: {len(data['documents'])} chunks")

    def search_bm25_initial(self, query, pool_size=10):
        """
        Etapa 1: Búsqueda inicial con BM25 sobre chunks.
        """
        # IMPORTANTE: Usar misma normalización que en indexación
        query_normalized = normalizar_doc(query)
        tokens = [t.text.lower() for t in nlp(query_normalized)]
        scores = self.bm25.get_scores(tokens)
        
        # Agrupar chunks por documento y encontrar mejor chunk
        temp = defaultdict(list)
        for i, score in enumerate(scores):
            if i < len(self.chunk_ids):
                chunk_id = self.chunk_ids[i]
                if chunk_id in self.metadatas:
                    doc_id = normalize_doc_id(self.metadatas[chunk_id]['document_id'])
                    temp[doc_id].append((score, chunk_id))
        
        # Seleccionar mejor chunk por documento
        doc_rankings = []
        for doc_id, scores_chunks in temp.items():
            if scores_chunks:
                top_score, best_chunk = max(scores_chunks, key=lambda x: x[0])
                doc_rankings.append((top_score, doc_id, best_chunk))
                self.doc_to_best_chunk[doc_id] = best_chunk
        
        # Ordenar y tomar top pool_size documentos
        doc_rankings.sort(reverse=True)
        return [doc_id for _, doc_id, _ in doc_rankings[:pool_size]]

    def search_biencoder_rerank(self, query, bm25_pool):
        """
        Etapa 2: Reordenación semántica del pool BM25 con Bi-Encoder.
        """
        if not bm25_pool:
            return []
        
        # Obtener textos de los mejores chunks seleccionados por BM25
        pool_texts = []
        for doc_id in bm25_pool:
            chunk_id = self.doc_to_best_chunk.get(doc_id, f"{doc_id}_chunk0")
            text = self.docs_raw.get(chunk_id, '')
            pool_texts.append(text)
        
        # Calcular embeddings
        q_emb = self.biencoder.encode(query, convert_to_tensor=True)
        pool_embs = self.biencoder.encode(pool_texts, convert_to_tensor=True)
        
        # Calcular similitudes
        similarities = []
        for pool_emb in pool_embs:
            if hasattr(q_emb, 'cpu'):  # GPU
                sim = float(torch.cosine_similarity(q_emb.unsqueeze(0), pool_emb.unsqueeze(0)))
            else:  # CPU
                sim = float(np.dot(q_emb, pool_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(pool_emb)))
            similarities.append(sim)
        
        # Reordenar por similitud semántica
        ranked_pairs = sorted(zip(similarities, bm25_pool), key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in ranked_pairs]

    def search_crossencoder_final(self, query, biencoder_pool):
        """
        Etapa 3: Reranking final con Cross-Encoder.
        """
        if not biencoder_pool:
            return []
        
        # Preparar pares query-chunk para Cross-Encoder
        pairs = []
        for doc_id in biencoder_pool:
            chunk_id = self.doc_to_best_chunk.get(doc_id, f"{doc_id}_chunk0")
            text = self.docs_raw.get(chunk_id, '')
            pairs.append([query, text])
        
        # Predecir scores
        scores = self.cross_encoder.predict(pairs)
        
        # Reordenar por scores del Cross-Encoder
        ranked_pairs = sorted(zip(scores, biencoder_pool), key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in ranked_pairs]

    def evaluate_pipeline(self, queries, relevant_docs, max_queries=50):
        """
        Evalúa el pipeline completo: BM25 → Bi-Encoder → Cross-Encoder
         Parámetro opcional `max_queries` para limitar el número de queries evaluadas.
        """
        for q in tqdm(queries, desc="Evaluando pipeline BM25"):
            gold = relevant_docs[q]
            
            # Limpiar mapeo de chunks
            self.doc_to_best_chunk.clear()
            
            # Etapa 1: BM25 inicial
            bm25_pool = self.search_bm25_initial(q, pool_size=10)
            
            # Etapa 2: Reordenación con Bi-Encoder  
            biencoder_reranked = self.search_biencoder_rerank(q, bm25_pool)
            
            # Etapa 3: Reranking final con Cross-Encoder
            final_ranking = self.search_crossencoder_final(q, biencoder_reranked)
            
            # Evaluación de cada etapa del pipeline
            methods_rankings = {
                'bm25': bm25_pool,
                'biencoder': biencoder_reranked,
                'cross_encoder': final_ranking
            }
            
            for method_name, ranking in methods_rankings.items():
                for k in [1, 3, 5, 10]:
                    metrics = evaluate_with_metadata(
                        retrieved=ranking,
                        gold=gold, 
                        k=k,
                        metadata=self.metadatas,  
                        method=method_name,       
                        query_text=q             
                    )   
                   
                    self.records.append(metrics)
                    
                   

    def save_results(self, output_dir):
        """
        Guarda resultados del pipeline BM25.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        
        # Archivo detallado
        detail_file = f"{output_dir}/bm25_pipeline_chunks_detailed.xlsx"
        df.to_excel(detail_file, index=False)
        
        # Resumen
        summary = df.groupby(['method','k'])[['precision','recall','f1','mrr','ndcg']].mean().round(4).reset_index()
        summary_file = f"{output_dir}/bm25_pipeline_chunks_summary.xlsx"
        summary.to_excel(summary_file, index=False)
        
        logger.info("\n=== PIPELINE BM25 + BI-ENCODER + CROSS-ENCODER ===")
        print(summary.to_markdown(index=False))
        
        return df

def plot_curves(df: pd.DataFrame, out_dir: str):
    """
    Genera gráficos comparativos de las métricas para diferentes valores de k.
    """
    os.makedirs(out_dir, exist_ok=True)
    ks = sorted(df['k'].unique())
    
    # Colores consistentes para cada método
    colors = {
        'bm25': '#1f77b4',
        'biencoder': '#ff7f0e', 
        'cross_encoder': '#2ca02c'
    }
    
    for metric, label in [('precision', 'Precision'), ('recall', 'Recall'), 
                         ('mrr', 'MRR'), ('ndcg', 'nDCG')]:
        plt.figure(figsize=(12, 8))
        
        for method in sorted(df['method'].unique()):
            method_data = df[df.method == method]
            vals = method_data.groupby('k')[metric].mean().reindex(ks)
            
            plt.plot(ks, vals, marker='o', label=method.upper(), 
                    linewidth=2.5, markersize=8, color=colors.get(method, 'gray'))
        
        plt.title(f'{label}@k - Pipeline BM25 + Bi-Encoder + Cross-Encoder', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('k (Top-k documentos)', fontsize=14)
        plt.ylabel(label, fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(ks)
        
        # Mejorar formato del eje Y
        if metric in ['precision', 'recall', 'ndcg']:
            plt.ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{metric}_at_k_bm25_pipeline.png", bbox_inches='tight', dpi=300)
        plt.close()

def main():
    """Función principal del experimento BM25 + Bi-Encoder + Cross-Encoder a nivel de chunks."""
    
    # Configuración
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline BM25 + Bi-Encoder + Cross-Encoder para chunk retrieval")
    parser.add_argument('--config', type=str, default="../config.yaml", help="Ruta al archivo de configuración")
    parser.add_argument('--json_gold', type=str, default="../evaluacion/dataset_test.json", help="Ruta al dataset de evaluación")
    parser.add_argument('--mode', type=str, default="embedding", choices=["embedding", "finetuneado"], help="Modo: embedding o finetuneado")
    parser.add_argument('--output_dir', type=str, default="resultados/experiment_bm25_pipeline_chunks", help="Directorio de salida")
    args = parser.parse_args()

    logger.info("=== EXPERIMENTO 2: BM25 PIPELINE A NIVEL DE CHUNKS ===")
    logger.info("Metodología: BM25 → Bi-Encoder → Cross-Encoder sobre chunks reales")
    
    # Inicializar evaluador
    evaluator = BM25PipelineEvaluator(args.config, args.mode)
    
    # Cargar datos
    queries, rel_docs = load_queries(args.json_gold, evaluator.translator)
    logger.info(f"Cargadas {len(queries)} queries para evaluación")
    
    # Cargar colección de chunks
    evaluator.load_collection()
    
    # Ejecutar evaluación del pipeline
    evaluator.evaluate_pipeline(queries, rel_docs)
    
    # Guardar resultados
    df = evaluator.save_results(args.output_dir)

     # VERIFICACIÓN DE DATOS
    if df.empty:
        logger.error("No se generaron resultados - verificar configuración")
        return
    
    logger.info(f"Generados {len(df)} registros de evaluación")
    
    summary = df.groupby(['method','k'])[['precision','recall','f1','mrr','ndcg']].mean().round(4).reset_index()

    # Análisis comparativo
    logger.info("\n=== ANÁLISIS COMPARATIVO POR ETAPA DEL PIPELINE ===")
    for metric in ['precision', 'recall', 'mrr', 'f1', 'ndcg']:
        k1_results = df[df['k'] == 1].groupby('method')[metric].mean()
        if not k1_results.empty:
            best_method = k1_results.idxmax()
            best_value = k1_results.max()
            logger.info(f"Mejor {metric.upper()}@1: {best_method.upper()} = {best_value:.4f}")
    
    logger.info(f"\nResultados guardados en: {args.output_dir}")
    
    logger.info("RESUMEN DE RESULTADOS - PIPELINE BM25:")
    logger.info("="*60)
    try:
        print(summary.to_markdown(index=False))
    except AttributeError:
        print("\n=== RESUMEN DE RESULTADOS ===")
        print(summary.to_string(index=False))

    # === 6. ANÁLISIS COMPARATIVO ===
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS COMPARATIVO POR MÉTODO:")
    logger.info("="*60)
    
    for metric in ['precision', 'recall', 'mrr', 'f1', 'ndcg']:
        best_at_k1 = summary[(summary['k'] == 1)].loc[summary[metric].idxmax(), ['method', metric]]
        logger.info(f"Mejor {metric.upper()}@1: {best_at_k1['method'].upper()} = {best_at_k1[metric]:.4f}")

 

    # === 7. ESTADÍSTICAS FINALES ===
    logger.info("\n" + "="*60)
    logger.info("ESTADÍSTICAS FINALES:")
    logger.info("="*60)
    logger.info(f"Total consultas procesadas: {len(queries)}")
    logger.info(f"Total evaluaciones generadas: {len(df)}")
    logger.info(f"Métodos evaluados: {df['method'].unique().tolist()}")
    logger.info(f"Valores de k: {sorted(df['k'].unique().tolist())}")
    
    # Comparación de rendimiento promedio
    avg_performance = df.groupby('method')[['precision', 'recall', 'f1', 'mrr', 'ndcg']].mean()
    logger.info("\nRendimiento promedio por método:")
    print(avg_performance.round(4).to_string())
    
    logger.info(f"\nResultados guardados en: {args.output_dir}")
    logger.info("="*60)
    logger.info("PIPELINE BM25 CHUNKS COMPLETADO!")
    logger.info("="*60)

if __name__ == '__main__':
    main()