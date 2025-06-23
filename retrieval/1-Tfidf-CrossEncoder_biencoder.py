"""
EXPERIMENTO 1: TF-IDF + BI-ENCODER + CROSS-ENCODER

PARA PRINCIPIANTES: Este script implementa y compara 4 métodos de búsqueda de documentos



MÉTODOS EVALUADOS:
1. TF-IDF: Busca palabras exactas (método tradicifonal)
2. Bi-Encoder: Entiende el significado de las palabras (IA moderna) 
3. Cross-Encoder: Evalúa muy detalladamente cada par pregunta-documento (IA avanzada)


QUÉ MIDE:
- Precision@K: ¿De los K documentos que devuelve, cuántos son realmente útiles?
- Recall@K: ¿De todos los documentos útiles, cuántos encontró?
- MRR: ¿En qué posición aparece el primer documento útil?
- nDCG: ¿Qué tan bien ordenados están los resultados?
"""

# IMPORTACIONES: Las "herramientas" que necesitamos
import os, json, logging, math
from typing import List, Dict, Any, Tuple
from tqdm import tqdm  # Para mostrar barras de progreso bonitas
import pandas as pd  # Para manejar tablas de datos como Excel
import matplotlib.pyplot as plt  # Para hacer gráficos
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF tradicional
from sentence_transformers import SentenceTransformer, CrossEncoder  # IA moderna
import numpy as np
import sys
import torch  # Para usar GPU si está disponible
from transformers import pipeline as hf_pipeline  # Para traducir euskera→español
from collections import defaultdict

# Fijar semillas para reproducibilidad
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Agregar ruta para importar funciones propias del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embeddings.load_model import cargar_configuracion
from retrieval.chroma_utils import translate_eu_to_es, limpiar_texto_excel

# Configurar logging para ver qué está pasando durante la ejecución
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




def load_data(path: str) -> Tuple[List[Dict], Dict[str, str]]:
    """Cargar datos del experimento"""
    logger.info(f"Cargando datos desde: {path}")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    queries = data['queries']
    corpus = data['corpus']
    logger.info(f"Cargados {len(queries)} preguntas y {len(corpus)} documentos")
    return queries, corpus

def load_chunk_metadata(corpus: Dict[str, str]) -> Dict[str, Dict]:
    """Crear metadatos simulados para mantener consistencia"""
    logger.info("Creando metadatos simulados para documentos...")
    metadata = {}
    for doc_id, text in corpus.items():
        chunk_id = f"{doc_id}_chunk0"
        metadata[doc_id] = {
            'chunk_id': chunk_id,
            'chunk_position': "1/1",
            'document_id': doc_id,
            'chunk_text': text[:500] + "..." if len(text) > 500 else text
        }
    logger.info(f"Metadatos creados para {len(metadata)} documentos")
    return metadata

def evaluate_with_metadata(retrieved: List[str], gold: List[str], k: int, 
                          metadata: Dict[str, Dict], method: str, 
                          query_text: str) -> Dict[str, Any]:
    """Evaluar qué tan buenos son los documentos recuperados"""
    # Limpiar y obtener documentos únicos manteniendo el orden
    seen, uniq = set(), []
    for d in retrieved:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
        if len(uniq) >= k:
            break
    
    # Clasificar resultados
    tp = [d for d in uniq if d in gold]
    fp = [d for d in uniq if d not in gold]
    fn = [d for d in gold if d not in uniq]
    
    # Calcular métricas básicas
    precision = len(tp) / k if k else 0
    recall = len(tp) / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calcular MRR
    rr = 0
    for i, d in enumerate(uniq, 1):
        if d in gold:
            rr = 1 / i
            break
    
    # Calcular nDCG
    rels = [1 if d in gold else 0 for d in uniq]
    dcg = sum(rels[i] / math.log2(i + 2) for i in range(len(rels)))
    ideal = [1] * min(len(gold), k)
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # Extraer información del mejor resultado
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


def calculate_tfidf_rankings(vectorizer, X, doc_ids, query_text):
    """TF-IDF - Búsqueda  por palabras clave"""
    logger.debug(f"TF-IDF: Procesando query '{query_text[:50]}...'")
    qv = vectorizer.transform([query_text])
    sims = (X @ qv.T).toarray().ravel()
    idx = sims.argsort()[::-1]
    return [doc_ids[i] for i in idx]

# def calculate_biencoder_rankings(biencoder, corpus, doc_ids, query_text, device):
#     """Bi-Encoder """
#     logger.debug(f"Bi-Encoder: Procesando query '{query_text[:50]}...'")
#     q_emb = biencoder.encode(query_text, convert_to_tensor=True, device=device)
#     docs_texts = [corpus[doc_id] for doc_id in doc_ids]
#     doc_embs = biencoder.encode(docs_texts, convert_to_tensor=True, device=device)
    
#     similarities = []
#     if hasattr(q_emb, 'cpu'):  # GPU
#         for doc_emb in doc_embs:
#             sim = float(torch.cosine_similarity(q_emb.unsqueeze(0), doc_emb.unsqueeze(0)))
#             similarities.append(sim)
#     else:  # CPU
#         for doc_emb in doc_embs:
#             sim = float(np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)))
#             similarities.append(sim)
    
#     ranked_pairs = sorted(zip(similarities, doc_ids), key=lambda x: x[0], reverse=True)
#     return [doc_id for _, doc_id in ranked_pairs]


def calculate_biencoder_rankings(biencoder, corpus, doc_ids, query_text, device, batch_size=16):
    """Bi-Encoder con procesamiento por batches para evitar OOM"""
    logger.debug(f"Bi-Encoder: Procesando query '{query_text[:50]}...'")
    q_emb = biencoder.encode(query_text, convert_to_tensor=True, device=device)
    
    docs_texts = [corpus[doc_id] for doc_id in doc_ids]
    
    # Procesar documentos en batches
    all_similarities = []
    for i in range(0, len(docs_texts), batch_size):
        batch_texts = docs_texts[i:i+batch_size]
        batch_embs = biencoder.encode(batch_texts, convert_to_tensor=True, device=device)
        
        # Calcular similitudes para este batch
        batch_similarities = []
        for doc_emb in batch_embs:
            sim = float(torch.cosine_similarity(q_emb.unsqueeze(0), doc_emb.unsqueeze(0)))
            batch_similarities.append(sim)
        
        all_similarities.extend(batch_similarities)
        
        # Limpiar memoria
        del batch_embs
        torch.cuda.empty_cache()
    
    # Ranking final
    ranked_pairs = sorted(zip(all_similarities, doc_ids), key=lambda x: x[0], reverse=True)
    return [doc_id for _, doc_id in ranked_pairs]

def calculate_crossencoder_rankings(cross_encoder, corpus, pool_docs, query_text, batch_size=8):
    """Cross-Encoder -"""
    if not pool_docs:
        logger.debug("Cross-Encoder: Pool vacío, devolviendo lista vacía")
        return []
    
    logger.debug(f"Cross-Encoder: Evaluando {len(pool_docs)} documentos candidatos")
    pairs = []
    for doc_id in pool_docs:
        document_text = corpus[doc_id]
        pair = (query_text, document_text)
        pairs.append(pair)
    
    scores = cross_encoder.predict(pairs, batch_size=batch_size)
    ranked_pairs = sorted(zip(scores, pool_docs), key=lambda x: x[0], reverse=True)
    return [doc_id for _, doc_id in ranked_pairs]

def create_balanced_pool(tfidf_ranking, biencoder_ranking, pool_size=50):
    """Pool Balanceado - Eliminando sesgos metodológicos"""
    logger.debug(f"Creando pool balanceado de {pool_size} documentos...")
    half_size = pool_size // 2
    tfidf_pool = tfidf_ranking[:half_size]
    biencoder_pool = biencoder_ranking[:half_size]
    
    seen = set()
    balanced_pool = []
    max_iterations = max(len(tfidf_pool), len(biencoder_pool))
    
    for i in range(max_iterations):
        if i < len(tfidf_pool) and tfidf_pool[i] not in seen:
            balanced_pool.append(tfidf_pool[i])
            seen.add(tfidf_pool[i])
            
        if i < len(biencoder_pool) and biencoder_pool[i] not in seen:
            balanced_pool.append(biencoder_pool[i])
            seen.add(biencoder_pool[i])
            
        if len(balanced_pool) >= pool_size:
            break
    
    logger.debug(f"Pool balanceado creado: {len(balanced_pool)} documentos únicos")
    return balanced_pool

def calculate_hybrid_pipeline(tfidf_ranking, biencoder, cross_encoder, corpus, 
                            query_text, device, pool_size=10, batch_size=8):
    """Pipeline Híbrido """
    logger.debug(f"Pipeline Híbrido: Iniciando con pool de {pool_size} documentos")
    tfidf_pool = tfidf_ranking[:pool_size]
    
    if not tfidf_pool:
        logger.warning("Pipeline Híbrido: Pool TF-IDF vacío")
        return []
    
    # Reordenación con Bi-Encoder
    logger.debug("Pipeline: Reordenando con Bi-Encoder...")
    q_emb = biencoder.encode(query_text, convert_to_tensor=True, device=device)
    pool_texts = [corpus[doc_id] for doc_id in tfidf_pool]
    pool_embs = biencoder.encode(pool_texts, convert_to_tensor=True, device=device)
    
    if hasattr(q_emb, 'cpu'):  # GPU
        sem_scores = [
            float(torch.cosine_similarity(q_emb.unsqueeze(0), emb.unsqueeze(0)))
            for emb in pool_embs
        ]
    else:  # CPU
        sem_scores = [
            float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
            for emb in pool_embs
        ]
    
    biencoder_reranked = [doc for _, doc in sorted(zip(sem_scores, tfidf_pool), 
                                                  key=lambda x: x[0], reverse=True)]
    
    # Reranking final con Cross-Encoder
    logger.debug("Pipeline: Reranking final con Cross-Encoder...")
    final_ranking = calculate_crossencoder_rankings(cross_encoder, corpus, 
                                                   biencoder_reranked, query_text, batch_size)
    
    logger.debug(f"Pipeline completado: {len(final_ranking)} documentos ordenados")
    return final_ranking

# VISUALIZACIÓN: Crear gráficos para entender los resultados
def plot_curves(df: pd.DataFrame, out_dir: str):
    """
    FUNCIÓN: Crear gráficos comparativos de rendimiento
    
    QUÉ HACE:
    - Crea gráficos que muestran cómo se comporta cada método
    - Una línea por método, puntos por diferentes valores de K
    - Permite ver visualmente qué método es mejor y cuándo
    
    PARÁMETROS:
    - df: DataFrame con todos los resultados del experimento
    - out_dir: Directorio donde guardar los gráficos
    """
    logger.info("Generando gráficos comparativos...")
    
    # Calcular métricas promedio por método y K
    metrics_summary = df.groupby(['method', 'k'])[['precision', 'recall', 'f1', 'mrr', 'ndcg']].mean().reset_index()
    
    # Definir colores y estilos para cada método
    method_styles = {
        'tfidf': {'color': '#1f77b4', 'marker': 'o', 'label': 'TF-IDF'},
        'biencoder': {'color': '#ff7f0e', 'marker': 's', 'label': 'Bi-Encoder'},
        'cross_encoder': {'color': '#2ca02c', 'marker': '^', 'label': 'Cross-Encoder'}
        # 'hybrid_pipeline': {'color': '#d62728', 'marker': 'D', 'label': 'Pipeline Híbrido'}
    }
    
    # Lista de métricas a graficar
    metrics = ['precision', 'recall', 'f1', 'mrr', 'ndcg']
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Graficar cada método
        for method in metrics_summary['method'].unique():
            method_data = metrics_summary[metrics_summary['method'] == method]
            
            if method in method_styles:
                style = method_styles[method]
                ax.plot(method_data['k'], method_data[metric], 
                       color=style['color'], marker=style['marker'], 
                       label=style['label'], linewidth=2, markersize=8)
        
        # Configurar el gráfico
        ax.set_xlabel('K (Top-K Documents)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs K', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks([1, 3, 5, 10])
        
        # Establecer límites del eje Y para mejor visualización
        if metric in ['precision', 'recall', 'f1', 'mrr', 'ndcg']:
            ax.set_ylim(0, 1.0)
    
    # Ocultar el último subplot si no se usa
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)
    
    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/comparison_curves.pdf", bbox_inches='tight')
    plt.close()
    
    # Crear gráfico de barras para comparación directa en K=5
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    k5_data = metrics_summary[metrics_summary['k'] == 5]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, method in enumerate(k5_data['method'].unique()):
        method_values = k5_data[k5_data['method'] == method]
        values = [method_values[metric].iloc[0] if len(method_values) > 0 else 0 for metric in metrics]
        
        if method in method_styles:
            style = method_styles[method]
            ax.bar(x + i * width, values, width, 
                  color=style['color'], label=style['label'], alpha=0.8)
    
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparación de Métodos en K=5', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_bars_k5.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/comparison_bars_k5.pdf", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos guardados en {out_dir}/")

# FUNCIÓN PRINCIPAL: Ejecutar todo el experimento
def main():
    """
    FUNCIÓN PRINCIPAL: Orquesta todo el experimento
    
    QUÉ HACE:
    1. Carga datos y modelos
    2. Procesa cada pregunta con los 4 métodos
    3. Evalúa resultados y genera métricas
    4. Guarda resultados en Excel
    5. Crea gráficos comparativos
    
    CONFIGURACIÓN:
    - dataset_path: Ruta al archivo de test
    - out_dir: Directorio para guardar resultados
    - pool_size: Tamaño del pool para Cross-Encoder
    - top_ks: Valores de K para evaluar (1, 3, 5, 10)
    """
    logger.info("INICIANDO EXPERIMENTO TF-IDF + BI-ENCODER + CROSS-ENCODER")
    
    # CONFIGURACIÓN
    dataset_path = '../evaluacion/dataset_test.json'
    out_dir = '../resultados/experimento_tfidf_biencoder_xenc'
    pool_size = 5  # Documentos a pasar a Cross-Encoder.  Reducido de 10 a 5 documentos
    top_ks = [1, 3, 5, 10]  # Valores de K para evaluar
    batch_size = 4  # Reducido de 8 a 4 para usar menos memoria, Tamaño de lote para Cross-Encoder
    num_queries = 100  # Número de preguntas para la prueba
    biencoder_batch_size = 16  # Tamaño de lote para Bi-Encoder, ajustado para evitar OOM
    # PASO 1: CARGAR DATOS
    logger.info("Cargando datos del experimento...")
    queries, corpus = load_data(dataset_path)
    metadata = load_chunk_metadata(corpus)
    
    # PASO 2: CARGAR MODELOS IA
    logger.info("Cargando modelos de IA...")
    
    # Configurar dispositivo (GPU si está disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar modelos
    
    config = cargar_configuracion('../config.yaml')
    biencoder = SentenceTransformer(config['model']['name_embedding'])
    cross_encoder = CrossEncoder(config['model']['name_cross_encoder'], device=device)
    logger.info("Modelos cargados exitosamente")
    
    # PASO 3: PREPARAR TF-IDF
    logger.info("Preparando vectorizador TF-IDF...")
    doc_ids = list(corpus.keys())
    docs = [corpus[doc_id] for doc_id in doc_ids]
    
    # Configurar y entrenar TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams y bigrams
        max_features=15000,  # Máximo 15k características
        stop_words=None,     # No remover stop words (importantes en medicina)
        lowercase=True,      # Convertir a minúsculas
        strip_accents='unicode',  # Normalizar acentos
        min_df=2,         ## Términos en al menos 2 documentos
        max_df=0.95       # Términos en máximo 95% documentos
    )
    
    X = vectorizer.fit_transform(docs)
    logger.info(f"TF-IDF entrenado: {X.shape[0]} documentos, {X.shape[1]} características")
    
    # PASO 4: Procesar cada pregunta
    logger.info("Procesando preguntas del dataset...")
    records = []
    
    # Configurar traductor si es necesario
    translator = None
    try:
        translator = hf_pipeline('translation', model=config['model']['translation_model'], device='cpu')
        logger.info("Traductor euskera→español cargado")
    except:
        logger.warning("No se pudo cargar el traductor")
    
    queries_subset = queries[:num_queries]  # Limitar a 30 preguntas para prueba rápida
    # Procesar cada pregunta
    for query_data in tqdm(queries_subset, desc="Procesando queries"):
        query_text = query_data['text_es']
        gold_docs = query_data['relevant_docs']
        
      
        if not query_text:
            query_text = translator([query_data['text_eu']])[0]['translation_text']
            logger.info(f"Query traducida: {query_text[:50]}...")
           
        
        logger.info(f"Procesando: '{query_text[:60]}...'")
        
        # MÉTODO 1: TF-IDF RANKING
        tfidf_ranking = calculate_tfidf_rankings(vectorizer, X, doc_ids, query_text)
        
        # MÉTODO 2: BI-ENCODER RANKING
        biencoder_ranking = calculate_biencoder_rankings(biencoder, corpus, doc_ids, query_text, device, batch_size=biencoder_batch_size)
        
        # MÉTODO 3: CROSS-ENCODER CON POOL BALANCEADO
        balanced_pool = create_balanced_pool(tfidf_ranking, biencoder_ranking, pool_size * 5)
        crossencoder_ranking = calculate_crossencoder_rankings(cross_encoder, corpus, balanced_pool, query_text, batch_size)
        
        
        
        # EVALUACIÓN PARA CADA K
        for k in top_ks:
            # Evaluar TF-IDF
            res_tfidf = evaluate_with_metadata(tfidf_ranking, gold_docs, k, metadata, 'tfidf', query_text)
            records.append(res_tfidf)
            
            # Evaluar Bi-Encoder
            res_biencoder = evaluate_with_metadata(biencoder_ranking, gold_docs, k, metadata, 'biencoder', query_text)
            records.append(res_biencoder)
            
            # Evaluar Cross-Encoder
            res_crossencoder = evaluate_with_metadata(crossencoder_ranking, gold_docs, k, metadata, 'cross_encoder', query_text)
            records.append(res_crossencoder)
            
            # # # Evaluar Pipeline Híbrido
            # # res_hybrid = evaluate_with_metadata(hybrid_ranking, gold_docs, k, metadata, 'hybrid_pipeline', query_text)
            # records.append(res_hybrid)
    
    # PASO 5: GUARDAR RESULTADOS
    logger.info("Guardando resultados...")
    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    
    # Archivo detallado (todas las columnas por query)
    detail_file = f"{out_dir}/detalle_tfidf_biencoder_xenc.xlsx"
    df.to_excel(detail_file, index=False)
    logger.info(f"Archivo detallado guardado: {detail_file}")
    
    # Archivo resumen (métricas promediadas)
    summary = (
        df.groupby(['method', 'k'])[['precision', 'recall', 'mrr', 'f1', 'ndcg']]
        .mean().round(4).reset_index()
    )
    summary_file = f"{out_dir}/resumen_tfidf_biencoder_xenc.xlsx"
    summary.to_excel(summary_file, index=False)
    logger.info(f"Archivo resumen guardado: {summary_file}")
    
    # Mostrar resumen en consola
    logger.info("RESUMEN DE RESULTADOS:")
    print("\n" + "="*80)
    print("RESUMEN DE MÉTRICAS POR MÉTODO Y K")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    
    # PASO 6: GENERAR GRÁFICOS
    logger.info("Generando gráficos comparativos...")
    plot_curves(df, out_dir)
    
    # PASO 7: ANÁLISIS FINAL
    logger.info("ANÁLISIS FINAL DE RESULTADOS:")
    
    # Encontrar el mejor método por métrica en K=5
    k5_data = summary[summary['k'] == 5]
    
    for metric in ['precision', 'recall', 'f1', 'mrr', 'ndcg']:
        best_row = k5_data.loc[k5_data[metric].idxmax()]
        logger.info(f"Mejor {metric.upper()}@5: {best_row['method']} ({best_row[metric]:.4f})")
    
    # Calcular mejora del mejor método vs TF-IDF baseline
    tfidf_k5 = k5_data[k5_data['method'] == 'tfidf']
    if not tfidf_k5.empty:
        tfidf_baseline = tfidf_k5.iloc[0]
        logger.info("\nMEJORAS RESPECTO A TF-IDF BASELINE:")
        
        for metric in ['precision', 'recall', 'f1', 'mrr', 'ndcg']:
            best_row = k5_data.loc[k5_data[metric].idxmax()]
            if best_row['method'] != 'tfidf':
                mejora = ((best_row[metric] - tfidf_baseline[metric]) / tfidf_baseline[metric]) * 100
                logger.info(f"{metric.upper()}: {mejora:.1f}% mejora ({best_row['method']})")
    
    logger.info("EXPERIMENTO COMPLETADO EXITOSAMENTE!")

if __name__ == "__main__":
    main()