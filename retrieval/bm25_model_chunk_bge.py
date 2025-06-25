"""
Script Experimento: BM25 + Bi-Encoder + Cross-Encoder - a nivel de chunk

Pipeline híbrido para evaluación de chunk retrieval usando ChromaDB con 4 estrategias:

METODOLOGÍA DUAL COMPLETA (4 estrategias - NIVEL CHUNKS):
1. BM25 Independiente (todos los chunks → ranking de chunks)
2. Bi-Encoder Independiente (todos los chunks → ranking de chunks) 
3. Cross-Encoder Independiente (todos los chunks → ranking de chunks)


DIFERENCIAS CON EXPERIMENTO ORIGINAL:
- Mantiene evaluación a nivel de chunks (como original)
- Añade métodos independientes para comparación justa
- Implementa pool balanceado con chunks (no documentos)
- Comparable con metodología TF-IDF pero usando chunks reales
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
from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import translate_eu_to_es, limpiar_texto_excel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_queries_with_chunk_mapping(json_path, translator, metadatas):
    """
    Carga queries y mapea document_ids del ground truth a chunk_ids reales de ChromaDB.
    EXPANDE cada document_id a TODOS sus chunk_ids para evaluación a nivel de chunk.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries, relevant = [], {}
    doc_to_chunks = defaultdict(list)
    
    # Crear mapeo: document_id → [lista de chunk_ids]
    for chunk_id, metadata in metadatas.items():
        if 'document_id' in metadata:
            doc_id = normalize_doc_id(metadata['document_id'])
            doc_to_chunks[doc_id].append(chunk_id)
    
    logger.info(f"Mapeo creado: {len(doc_to_chunks)} documentos → {sum(len(chunks) for chunks in doc_to_chunks.values())} chunks")
    
    # DEBUG: Mostrar algunos mapeos de ejemplo
    logger.info("DEBUG - Ejemplos de mapeo document_id → chunks:")
    for i, (doc_id, chunk_list) in enumerate(list(doc_to_chunks.items())[:5]):
        logger.info(f"  {doc_id} → {len(chunk_list)} chunks: {chunk_list[:3]}...")
    
    for q in data['queries']:
        text = q.get("text_es") or translate_eu_to_es(q["text_eu"], translator)
        
        if text:
            # Convertir document_ids del ground truth a TODOS los chunk_ids
            relevant_chunks = []
            found_docs = []
            missing_docs = []
            
            for doc_id in q['relevant_docs']:
                normalized_doc_id = normalize_doc_id(doc_id)
                chunks_for_doc = doc_to_chunks.get(normalized_doc_id, [])
                
                if chunks_for_doc:
                    # EXPANDIR: Agregar TODOS los chunks del documento
                    relevant_chunks.extend(chunks_for_doc)
                    found_docs.append(f"{doc_id}({len(chunks_for_doc)}chunks)")
                    logger.debug(f"  {doc_id} → expandido a {len(chunks_for_doc)} chunks")
                else:
                    # Intentar variaciones del document_id
                    found_variation = False
                    possible_variations = [
                        doc_id, f"web_{doc_id}", f"pdf_{doc_id}",
                        doc_id.replace("_es", "").replace("_eu", "")
                    ]
                    
                    for variation in possible_variations:
                        normalized_variation = normalize_doc_id(variation)
                        if normalized_variation in doc_to_chunks:
                            chunks_for_variation = doc_to_chunks[normalized_variation]
                            relevant_chunks.extend(chunks_for_variation)
                            found_docs.append(f"{doc_id}→{variation}({len(chunks_for_variation)}chunks)")
                            logger.debug(f"  {doc_id} → {variation} → {len(chunks_for_variation)} chunks")
                            found_variation = True
                            break
                    
                    if not found_variation:
                        missing_docs.append(doc_id)
            
            if relevant_chunks:
                queries.append(text)
                relevant[text] = relevant_chunks
                
                # DEBUG: Información detallada por query
                logger.info(f"Query expandida: {text[:80]}...")
                logger.info(f"  Documentos procesados: {found_docs}")
                if missing_docs:
                    logger.warning(f"  Documentos NO encontrados: {missing_docs}")
                logger.info(f"  Total chunks relevantes: {len(relevant_chunks)}")
                logger.info(f"  Primeros 5 chunks: {relevant_chunks[:5]}")
                logger.info("")
            else:
                logger.warning(f"Query SIN chunks relevantes: {text[:50]}... (docs originales: {q['relevant_docs']})")
    
    logger.info(f"Resultado final: {len(queries)} queries válidas cargadas")
    if relevant:
        avg_chunks = sum(len(rel) for rel in relevant.values()) / len(relevant)
        logger.info(f"Promedio de chunks relevantes por query: {avg_chunks:.1f}")
    
    return queries, relevant

# def evaluate_with_metadata(retrieved_chunks: List[str], gold_chunks: List[str], k: int, 
#                           metadata: Dict[str, Dict], method: str, 
#                           query_text: str) -> Dict[str, Any]:

def evaluate_with_metadata(retrieved_chunks: List[str], gold_chunks: List[str], k: int, 
                          metadata: Dict[str, Dict], method: str, 
                          query_text: str, docs_raw: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Evalúa chunks recuperados directamente (sin conversión document→chunk).
    """
    # Tomar top-k chunks únicos
    seen, uniq_chunks = set(), []
    for chunk_id in retrieved_chunks:
        if chunk_id not in seen:
            seen.add(chunk_id)
            uniq_chunks.append(chunk_id)
        if len(uniq_chunks) >= k:
            break
    
    # Métricas
    tp = [c for c in uniq_chunks if c in gold_chunks]
    fp = [c for c in uniq_chunks if c not in gold_chunks]
    fn = [c for c in gold_chunks if c not in uniq_chunks]
    
    precision = len(tp) / k if k else 0
    recall = len(tp) / len(gold_chunks) if gold_chunks else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MRR
    rr = 0
    for i, chunk_id in enumerate(uniq_chunks, 1):
        if chunk_id in gold_chunks:
            rr = 1 / i
            break
    
    # nDCG
    rels = [1 if c in gold_chunks else 0 for c in uniq_chunks]
    dcg = sum(rels[i] / math.log2(i + 2) for i in range(len(rels)))
    ideal = [1] * min(len(gold_chunks), k)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # Información del mejor chunk
    chunk_position = ""
    chunk_id = ""
    chunk_text = ""
    
    if uniq_chunks:
        best_chunk = uniq_chunks[0]
        if best_chunk in metadata:
            meta = metadata[best_chunk]
            chunk_position = meta.get('chunk_position', '')
            chunk_id = meta.get('chunk_id', best_chunk)
             # CORRECCIÓN: Obtener el texto del chunk desde docs_raw
            if docs_raw and best_chunk in docs_raw:
                chunk_text = limpiar_texto_excel(docs_raw[best_chunk])
            else:
                # Fallback: intentar desde metadatos (si existe)
                chunk_text = limpiar_texto_excel(meta.get('chunk_text', ''))
            logging.debug(f"Mejor chunk: {chunk_id} - Posición: {chunk_position} - Texto: {chunk_text[:50]}...")
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'mrr': round(rr, 4),
        'ndcg': round(ndcg, 4),
        'fp': ";".join(fp),
        'fn': ";".join(fn),
        'topk': ",".join(uniq_chunks),
        'method': method,
        'query': query_text,
        'k': k,
        'chunk_position': chunk_position,
        'chunk_id': chunk_id,
        'chunk_text': chunk_text
    }

class BM25DualChunkEvaluator:
    def __init__(self, config_path, mode):
        """
        Evaluador dual BM25 con 3 estrategias a nivel de chunks.
        Implementa selección del mejor chunk por documento para mayor realismo RAG.
        """
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.mode = mode
        self.records = []
        self.doc_to_best_chunk = {}  # Mapeo documento → mejor chunk
        self.client = chromadb.PersistentClient(path=self.cfg['paths']['chroma_db_path'])
        self._load_models()

    def _load_models(self):
        """
        Carga modelos de recuperación semántica y traducción.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
            
        logger.info(f"Usando dispositivo: {device}")
        
        
        model_name = self.cfg['model'][f'name_{self.mode}']         
        logger.info(f"Nombre del modelo: {self.mode}")    



       # 2) Si el nombre contiene "finetuning", construimos ruta local
        if "finetuning" in self.mode:
            base_dir     = self.cfg['paths']['model_path']
            model_name = os.path.join(base_dir,  model_name)
            logger.info(f"Biencoder fine-tuneado desde carpeta local: {model_name}")
            
        else:
        # nombre HF o ruta absoluta
            model_name = self.cfg['model'][f'name_{self.mode}']  
            logger.info(f"Biencoder desde HuggingFace o ruta directa: {model_name}")

            # 3) Cargamos el biencoder
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
        
        # CORRECCIÓN: Usar chunk_id de metadatos para el mapeo, NO los IDs de ChromaDB
        self.chunk_ids = []
        self.docs_raw = {}  # chunk_id → texto del documento
        self.docs_norm = []  # Lista de documentos normalizados para BM25
        
        # Crear mapeos correctos
        for i, (chroma_id, document, metadata) in enumerate(zip(data['ids'], data['documents'], data['metadatas'])):
            if 'chunk_id' in metadata and 'document_id' in metadata:
                chunk_id = metadata['chunk_id']
                self.chunk_ids.append(chunk_id)
                self.docs_raw[chunk_id] = document  # Mapeo chunk_id → texto
                self.docs_norm.append(normalizar_doc(document))  # Para BM25
        
        # Metadatos completos por chunk_id
        self.metadatas = {
            m['chunk_id']: m for m in data['metadatas'] 
            if m.get('chunk_id') and m.get('document_id')
        }
        
        # Verificar que el mapeo sea consistente
        if len(self.chunk_ids) != len(self.docs_raw) or len(self.chunk_ids) != len(self.docs_norm):
            logger.error(f"Inconsistencia en mapeos: chunk_ids={len(self.chunk_ids)}, docs_raw={len(self.docs_raw)}, docs_norm={len(self.docs_norm)}")
        
        # Inicializar BM25 sobre chunks normalizados
        self.bm25 = BM25Okapi([[t.text.lower() for t in nlp(d)] for d in self.docs_norm])
        
        logger.info(f"Colección cargada: {len(data['documents'])} chunks totales")
        logger.info(f"Chunk IDs procesados: {len(self.chunk_ids)}")
        logger.info(f"Metadatos mapeados: {len(self.metadatas)}")
        logger.info(f"Docs raw mapeados: {len(self.docs_raw)}")
        logger.info(f"Docs normalizados: {len(self.docs_norm)}")
    


    # =============== ESTRATEGIAS INDEPENDIENTES ===============
    
    def calculate_bm25_rankings(self, query):
        """
        ESTRATEGIA 1: BM25 independiente sobre todos los chunks.
        Agrupa por documento y selecciona el mejor chunk por documento.
        Devuelve ranking de documentos ordenados por el mejor chunk de cada uno.
        """
        logger.info(f"BM25: Procesando query '{query[:50]}...'")
        
        query_normalized = normalizar_doc(query)
        tokens = [t.text.lower() for t in nlp(query_normalized)]
        # DEBUG: Verificar tokenización
        logger.info(f"BM25 DEBUG - Query original: {query[:100]}")
        logger.info(f"BM25 DEBUG - Query normalizada: {query_normalized[:100]}")
        logger.info(f"BM25 DEBUG - Tokens extraídos: {tokens}")

        if not tokens:
            logger.warning("BM25: No se encontraron tokens válidos en la query")
            return []

        scores = self.bm25.get_scores(tokens)
        
         # DEBUG: Verificar scores
        logger.info(f"BM25 DEBUG - Scores obtenidos: {len(scores)} chunks")
        logger.info(f"BM25 DEBUG - Score máximo: {max(scores) if scores.size > 0 else 'No scores'}")
        logger.info(f"BM25 DEBUG - Score mínimo: {min(scores) if scores.size > 0 else 'No scores'}")

        # Agrupar chunks por documento y encontrar mejor chunk
        doc_chunk_scores = defaultdict(list)
        for i, score in enumerate(scores):
            if i < len(self.chunk_ids):
                chunk_id = self.chunk_ids[i]
                if chunk_id in self.metadatas:
                    doc_id = normalize_doc_id(self.metadatas[chunk_id]['document_id'])
                    doc_chunk_scores[doc_id].append((score, chunk_id))
        
        # Seleccionar mejor chunk por documento
        doc_rankings = []
        self.doc_to_best_chunk = {}  # Reset mapeo
        
        for doc_id, scores_chunks in doc_chunk_scores.items():
            if scores_chunks:
                top_score, best_chunk = max(scores_chunks, key=lambda x: x[0])
                doc_rankings.append((top_score, doc_id, best_chunk))
                self.doc_to_best_chunk[doc_id] = best_chunk
        
        # Ordenar por score del mejor chunk
        doc_rankings.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"BM25: Top 3 docs: {[f'{doc[:15]}:{score:.4f}' for score, doc, _ in doc_rankings[:3]]}")
        logger.info(f"BM25 DEBUG - Total documentos rankeados: {len(doc_rankings)}")
        
        
        
        # Devolver chunk_ids del mejor chunk por documento
        return [best_chunk for _, _, best_chunk in doc_rankings]

    def calculate_biencoder_rankings(self, query, batch_size=32):
        """
        ESTRATEGIA 2: Bi-Encoder independiente sobre todos los chunks.
        Agrupa por documento y selecciona el chunk con mayor similitud por documento.
        Devuelve ranking de documentos ordenados por el mejor chunk de cada uno.
        Usar query Y chunks ORIGINALES (sin normalizar) para Bi-Encoder.
        """
        logger.debug(f"Bi-Encoder: Procesando query '{query[:50]}...'")
        
        q_emb = self.biencoder.encode(query, convert_to_tensor=True)
        
        # Procesar chunks en batches y agrupar por documento
        doc_chunk_similarities = defaultdict(list)
        
        for i in range(0, len(self.chunk_ids), batch_size):
            batch_chunk_ids = self.chunk_ids[i:i+batch_size]
            batch_texts = [self.docs_raw.get(chunk_id, '') for chunk_id in batch_chunk_ids]
            
            if batch_texts:
                batch_embs = self.biencoder.encode(batch_texts, convert_to_tensor=True)
                
                for j, chunk_emb in enumerate(batch_embs):
                    if hasattr(q_emb, 'cpu'):  # GPU
                        sim = float(torch.cosine_similarity(q_emb.unsqueeze(0), chunk_emb.unsqueeze(0)))
                    else:  # CPU
                        sim = float(np.dot(q_emb, chunk_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(chunk_emb)))
                    
                    chunk_id = batch_chunk_ids[j]
                    if chunk_id in self.metadatas:
                        doc_id = normalize_doc_id(self.metadatas[chunk_id]['document_id'])
                        doc_chunk_similarities[doc_id].append((sim, chunk_id))
        
        # Seleccionar mejor chunk por documento
        doc_rankings = []
        if not hasattr(self, 'doc_to_best_chunk'):
            self.doc_to_best_chunk = {}
        
        for doc_id, sims_chunks in doc_chunk_similarities.items():
            if sims_chunks:
                top_sim, best_chunk = max(sims_chunks, key=lambda x: x[0])
                doc_rankings.append((top_sim, doc_id, best_chunk))
                self.doc_to_best_chunk[doc_id] = best_chunk
        
        # Ordenar por similitud del mejor chunk
        doc_rankings.sort(key=lambda x: x[0], reverse=True)
        
        logger.debug(f"Bi-Encoder: Top 3 docs: {[f'{doc[:15]}:{sim:.4f}' for sim, doc, _ in doc_rankings[:3]]}")
        
        # Devolver chunk_ids del mejor chunk por documento
        return [best_chunk for _, _, best_chunk in doc_rankings]

    def calculate_crossencoder_rankings(self, query, pool_chunks, batch_size=8):
        """
        Cross-Encoder sobre pool de chunks (puede ser todos o subset).
        Devuelve chunk_ids ordenados por score Cross-Encoder.
        """
        if not pool_chunks:
            logger.debug("Cross-Encoder: Pool vacío, devolviendo lista vacía")
            return []
        
        logger.debug(f"Cross-Encoder: Evaluando {len(pool_chunks)} chunks candidatos")
        
        # Preparar pares query-chunk
        pairs = []
        valid_chunks = []
        
        for chunk_id in pool_chunks:
            chunk_text = self.docs_raw.get(chunk_id, '')
            if chunk_text:
                pairs.append([query, chunk_text])
                valid_chunks.append(chunk_id)
        
        if not pairs:
            return []
        
        # Predecir scores en batches
        scores = self.cross_encoder.predict(pairs, batch_size=batch_size)
        
        # Ordenar por scores Cross-Encoder
        ranked_results = sorted(zip(scores, valid_chunks), key=lambda x: x[0], reverse=True)
        
        logger.debug(f"Cross-Encoder: Top 3 chunks: {[f'{cid[:20]}:{score:.4f}' for score, cid in ranked_results[:3]]}")
        return [chunk_id for _, chunk_id in ranked_results]

    def create_balanced_chunk_pool(self, bm25_chunks, biencoder_chunks, pool_size=10):
        """
        Pool Balanceado - Combina mejores chunks de BM25 y Bi-Encoder.
        Ahora funciona con chunk_ids que representan el mejor chunk por documento.
        """
        logger.debug(f"Creando pool balanceado de {pool_size} chunks...")
        
        half_size = pool_size // 2
        bm25_pool = bm25_chunks[:half_size]
        biencoder_pool = biencoder_chunks[:half_size]
        
        seen = set()
        balanced_pool = []
        max_iterations = max(len(bm25_pool), len(biencoder_pool))
        
        # Alternar entre métodos para balance
        for i in range(max_iterations):
            if i < len(bm25_pool) and bm25_pool[i] not in seen:
                balanced_pool.append(bm25_pool[i])
                seen.add(bm25_pool[i])
                
            if i < len(biencoder_pool) and biencoder_pool[i] not in seen:
                balanced_pool.append(biencoder_pool[i])
                seen.add(biencoder_pool[i])
                
            if len(balanced_pool) >= pool_size:
                break
        
        logger.debug(f"Pool balanceado creado: {len(balanced_pool)} chunks únicos (mejores por documento)")
        return balanced_pool

    def calculate_hybrid_pipeline(self, query, pool_size=10, batch_size=8):
        """
        ESTRATEGIA 4: Pipeline Híbrido con Pool Balanceado.
        1. BM25 + Bi-Encoder sobre todos los chunks
        2. Pool balanceado de chunks (no documentos)
        3. Cross-Encoder final sobre pool balanceado
        """
        logger.debug(f"Pipeline Híbrido: Iniciando con pool de {pool_size} chunks")
        
        # Obtener rankings completos
        bm25_chunks = self.calculate_bm25_rankings(query)
        biencoder_chunks = self.calculate_biencoder_rankings(query)
        
        # Crear pool balanceado de chunks
        balanced_pool = self.create_balanced_chunk_pool(bm25_chunks, biencoder_chunks, pool_size)
        
        if not balanced_pool:
            logger.warning("Pipeline Híbrido: Pool balanceado vacío")
            return []
        
        # Cross-Encoder final sobre pool balanceado
        logger.debug("Pipeline: Cross-Encoder final sobre pool balanceado...")
        final_ranking = self.calculate_crossencoder_rankings(query, balanced_pool, batch_size)
        
        logger.debug(f"Pipeline completado: {len(final_ranking)} chunks ordenados")
        return final_ranking

    # =============== EVALUACIÓN PRINCIPAL ===============
    
    def evaluate_all_strategies(self, queries, relevant_docs, max_queries=None):
        """
        Evalúa las 3 estrategias de chunk retrieval.
        """
        if max_queries:
            queries = queries[:max_queries]
            logger.info(f"Limitando evaluación a {max_queries} queries")
        
        for q in tqdm(queries, desc="Evaluando estrategias BM25 chunks"):
            gold_chunks = relevant_docs[q]
            
            logger.info(f"Procesando query: {q[:60]}...")
            logger.debug(f"Gold chunks: {len(gold_chunks)} esperados")
            
            # ESTRATEGIA 1: BM25 Independiente
            bm25_ranking = self.calculate_bm25_rankings(q)

            # DEBUG: Verificar resultados BM25
            logger.info(f"DEBUG - BM25 ranking devuelto: {len(bm25_ranking)} chunks")
            logger.info(f"DEBUG - Gold chunks esperados: {len(gold_chunks)}")
            logger.info(f"DEBUG - Primeros 5 BM25 chunks: {bm25_ranking[:5]}")
            logger.info(f"DEBUG - Primeros 5 gold chunks: {gold_chunks[:5]}")

            # Verificar si hay intersección
            intersection = set(bm25_ranking[:10]) & set(gold_chunks)
            logger.info(f"DEBUG - Intersección top-10 BM25 vs gold: {len(intersection)} chunks coinciden")
            if intersection:
                logger.info(f"DEBUG - Chunks que coinciden: {list(intersection)[:3]}")
            
            # ESTRATEGIA 2: Bi-Encoder Independiente
            biencoder_ranking = self.calculate_biencoder_rankings(q)
            
            # ESTRATEGIA 3: Cross-Encoder Independiente con Pool Balanceado
            bm25_pool = bm25_ranking[:10]  # Top 10 chunks BM25
            biencoder_pool = biencoder_ranking[:10]  # Top 10 chunks Bi-Encoder
            crossencoder_balanced_pool = self.create_balanced_chunk_pool(bm25_pool, biencoder_pool, pool_size=10)
            crossencoder_ranking = self.calculate_crossencoder_rankings(q, crossencoder_balanced_pool)

            

            # Evaluar todas las estrategias para cada K
            strategies = {
                'bm25': bm25_ranking,
                'biencoder': biencoder_ranking,
                'cross_encoder': crossencoder_ranking
            }
            
            for strategy_name, ranking in strategies.items():
                for k in [1, 3, 5, 10]:
                    metrics = evaluate_with_metadata(
                        retrieved_chunks=ranking,  # chunk_ids directos
                        gold_chunks=gold_chunks,
                        k=k,
                        metadata=self.metadatas,
                        method=strategy_name,
                        query_text=q,
                        docs_raw=self.docs_raw 
                    )
                    
                    self.records.append(metrics)

    def save_results(self, output_dir):
        """
        Guarda resultados de las 3 estrategias.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        
        # Archivo detallado
        detail_file = f"{output_dir}/bm25_chunks_3strategies_detalle_ft.xlsx"
        df.to_excel(detail_file, index=False)
        
        # Resumen por estrategia
        summary = df.groupby(['method','k'])[['precision','recall','f1','mrr','ndcg']].mean().round(4).reset_index()
        summary_file = f"{output_dir}/bm25_chunks_3estrategias_resumen_ft.xlsx"
        summary.to_excel(summary_file, index=False)
        
        logger.info("\n=== BM25 CHUNKS - 3ESTRATEGIAS - RESULTADOS ===")
        print(summary.to_markdown(index=False))
        
        # Análisis comparativo @1
        logger.info("\n=== ANÁLISIS COMPARATIVO @1 ===")
        k1_results = summary[summary['k'] == 1].sort_values('mrr', ascending=False)
        for _, row in k1_results.iterrows():
            logger.info(f"{row['method'].upper()}: Precision@1={row['precision']:.4f}, MRR={row['mrr']:.4f}")
        
        return df


def plot_curves(df: pd.DataFrame, out_dir: str):
    """
    Genera gráficos comparativos de las 3 estrategias.
    """
    os.makedirs(out_dir, exist_ok=True)
    ks = sorted(df['k'].unique())
    
    # Definir colores y estilos para cada estrategia
    strategy_styles = {
        'bm25': {'color': '#1f77b4', 'marker': 'o', 'label': 'BM25'},
        'biencoder': {'color': '#ff7f0e', 'marker': 's', 'label': 'Bi-Encoder'},
        'cross_encoder': {'color': '#2ca02c', 'marker': '^', 'label': 'Cross-Encoder'}
        # 'hybrid_balanced': {'color': '#d62728', 'marker': 'D', 'label': 'Híbrido Balanceado'}
    }
    
    # Calcular métricas promedio por método y K
    metrics_summary = df.groupby(['method', 'k'])[['precision', 'recall', 'f1', 'mrr', 'ndcg']].mean().reset_index()
    
    # Lista de métricas a graficar
    metrics = ['precision', 'recall', 'f1', 'mrr', 'ndcg']
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Graficar cada estrategia
        for strategy in metrics_summary['method'].unique():
            strategy_data = metrics_summary[metrics_summary['method'] == strategy]
            
            if strategy in strategy_styles:
                style = strategy_styles[strategy]
                ax.plot(strategy_data['k'], strategy_data[metric], 
                       color=style['color'], marker=style['marker'], 
                       label=style['label'], linewidth=2, markersize=8)
        
        # Configurar el gráfico
        ax.set_xlabel('K (Top-K Chunks)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs K - BM25 Chunks (3 Estrategias)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks([1, 3, 5, 10])
        
        # Establecer límites del eje Y
        if metric in ['precision', 'recall', 'f1', 'mrr', 'ndcg']:
            ax.set_ylim(0, 1.0)
    
    # Ocultar el último subplot si no se usa
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)
    
    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig(f"{out_dir}/bm25_chunks_3strategies_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos guardados en {out_dir}/")

def main():
    """Función principal del experimento BM25 chunks con 3 estrategias."""
    import argparse
    parser = argparse.ArgumentParser(description="BM25 Chunks: 3 estrategias de recuperación")
    parser.add_argument('--config', type=str, default="../config.yaml", help="Ruta al archivo de configuración")
    parser.add_argument('--json_gold', type=str, default="../evaluacion/dataset_test.json", help="Ruta al dataset de evaluación")
    parser.add_argument('--mode', type=str, default="finetuning", choices=["embedding", "finetuneado"], help="Modo: embedding o finetuneado")
    parser.add_argument('--output_dir', type=str, default="resultados/experiment_bm25_chunks_3strategies_fy", help="Directorio de salida")
    parser.add_argument('--max_queries', type=int, default=322, help="Máximo número de queries")
    args = parser.parse_args()

    logger.info("=== Experimento BM25 CHUNKS - 3 ESTRATEGIAS ===")
    logger.info("Estrategias: BM25, Bi-Encoder, Cross-Encoder (Pool Balanceado)")
    logger.info("Selección del mejor chunk por documento ")
    logger.info("Cross-Encoder Individual: Pool balanceado BM25+Bi-Encoder (10+10 mejores chunks)")
    
    
    # Inicializar evaluador
    evaluator = BM25DualChunkEvaluator(args.config, args.mode)
    
    # Cargar colección de chunks
    evaluator.load_collection()
    
    # Cargar datos con mapeo document→chunks
    queries, rel_docs = load_queries_with_chunk_mapping(
        args.json_gold, 
        evaluator.translator, 
        evaluator.metadatas
    )
    logger.info(f"Cargadas {len(queries)} queries para evaluación")
    
    # Ejecutar evaluación de las 3 estrategias
    evaluator.evaluate_all_strategies(queries, rel_docs, args.max_queries)
    
    # Guardar resultados
    df = evaluator.save_results(args.output_dir)
    
    # Análisis final
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS FINAL - BM25 CHUNKS 3 ESTRATEGIAS")
    logger.info("="*60)
    
    if not df.empty:
        logger.info(f"Total consultas procesadas: {len(queries)}")
        logger.info(f"Total evaluaciones generadas: {len(df)}")
        logger.info(f"Estrategias evaluadas: {df['method'].unique().tolist()}")
        
        # Mejor estrategia por métrica @1
        k1_data = df[df['k'] == 1]
        for metric in ['precision', 'mrr', 'f1']:
            if not k1_data.empty and metric in k1_data.columns:
                best_row = k1_data.loc[k1_data[metric].idxmax()]
                logger.info(f"Mejor {metric.upper()}@1: {best_row['method']} = {best_row[metric]:.4f}")
        
        # Mejora respecto a BM25 baseline
        bm25_k1 = k1_data[k1_data['method'] == 'bm25']
        if not bm25_k1.empty:
            baseline = bm25_k1.iloc[0]
            logger.info("\nMEJORAS RESPECTO A BM25 BASELINE:")
            
            for metric in ['precision', 'mrr', 'f1']:
                best_row = k1_data.loc[k1_data[metric].idxmax()]
                # if best_row['method'] != 'bm25':
                #     mejora = ((best_row[metric] - baseline[metric]) / baseline[metric]) * 100
                #     logger.info(f"{metric.upper()}: {mejora:.1f}% mejora ({best_row['method']})")
    

                if best_row['method'] != 'bm25':
            # CORRECCIÓN: Manejar división por cero
                    if baseline[metric] == 0:
                        if best_row[metric] > 0:
                            logger.info(f"{metric.upper()}: ∞% mejora ({best_row['method']}) - BM25 baseline era 0.0")
                        else:
                            logger.info(f"{metric.upper()}: Sin mejora - ambos métodos obtuvieron 0.0")
                    else:
                        mejora = ((best_row[metric] - baseline[metric]) / baseline[metric]) * 100
                        logger.info(f"{metric.upper()}: {mejora:.1f}% mejora ({best_row['method']})")
        logger.info(f"\nResultados guardados en: {args.output_dir}")
    logger.info("="*60)
    logger.info("BM25 CHUNKS 3 ESTRATEGIAS COMPLETADO!")
    logger.info("="*60)


if __name__ == '__main__':
    main()