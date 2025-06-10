import os
import json
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from generate_dataset import load_examples
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import logging

# ======== CONFIGURACIÓN BGE-M3 ========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(project_root, "finetuning", "dataset_finetune.json")
model_save_base = os.path.join(project_root, "models", "bge_m3_epochs")

# Crear directorio si no existe
os.makedirs(model_save_base, exist_ok=True)


model_name = "BAAI/bge-m3"

#  CONFIGURACIÓN CONSERVADORA PARA BGE-M3
batch_size = 16           # Más pequeño para modelo grande
lr = 2e-5            # MUY conservador para BGE-M3
max_epochs = 10          # Pocas épocas
patience = 3            # Early stopping estricto
warmup_steps = 200      # Warmup largo

print(f"🔧 Configuración BGE-M3:")
print(f"   Modelo: {model_name}")
print(f"   Dataset: {dataset_path}")
print(f"   Salida: {model_save_base}")
print(f"   Batch size: {batch_size}")
print(f"   Learning rate: {lr}")
print(f"   Max epochs: {max_epochs}")
print(f"   Patience: {patience}")

# ======== VERIFICACIÓN DE ARCHIVOS ========
if not os.path.exists(dataset_path):
    print(f" Error: No se encuentra el dataset en {dataset_path}")
    exit(1)

# ======== CARGA Y DIVISIÓN DE DATOS ========
print(f"\n Cargando datos...")
try:
    all_examples = load_examples(dataset_path)
    print(f" Cargados {len(all_examples)} ejemplos")
except Exception as e:
    print(f" Error cargando datos: {e}")
    exit(1)

random.shuffle(all_examples)
train_data, val_data = train_test_split(all_examples, test_size=0.2, random_state=42)

print(f" División de datos:")
print(f"   Entrenamiento: {len(train_data)} ejemplos")
print(f"   Validación: {len(val_data)} ejemplos")

# ======== ANÁLISIS DEL DATASET ========
print(f"\n Análisis del dataset:")
query_counts = defaultdict(int)
query_to_responses = defaultdict(set)

for ex in all_examples:
    query_text = ex.texts[0]
    response_text = ex.texts[1]
    query_counts[query_text] += 1
    query_to_responses[query_text].add(response_text)

unique_queries = len(query_counts)
total_pairs = len(all_examples)
avg_responses_per_query = total_pairs / unique_queries if unique_queries > 0 else 0
max_responses = max(query_counts.values()) if query_counts else 0

print(f"   Total pares query-response: {total_pairs}")
print(f"   Queries únicas: {unique_queries}")
print(f"   Promedio respuestas por query: {avg_responses_per_query:.2f}")
print(f"   Máximo respuestas por query: {max_responses}")

# ======== DEFINICIÓN DEL MODELO BGE-M3 ========
print(f"\n🔧 Configurando BGE-M3...")
try:
    #  CONFIGURACIÓN ESPECÍFICA PARA BGE-M3
    word_embedding_model = models.Transformer(
        model_name, 
        max_seq_length=512  # BGE-M3 soporta 8192, pero 512 es más estable
    )
    
    #  BGE-M3 usa CLS pooling (mejor para retrieval)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), 
        pooling_mode_cls_token=True,      # CLS pooling para BGE
        pooling_mode_mean_tokens=False    # Desactivar mean pooling
    )
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    print(f" BGE-M3 configurado correctamente")
    print(f" Dimensión embeddings: {model.get_sentence_embedding_dimension()}")
    
except Exception as e:
    print(f" Error configurando BGE-M3: {e}")
    exit(1)

# ======== CONFIGURAR ENTRENAMIENTO ========
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

def evaluate_model_improved(model: SentenceTransformer, val_data: list) -> dict:
    """
    Evaluación mejorada para BGE-M3
    """
    try:
        k = 3
        
        # Agrupar por pregunta
        query_to_relevant = defaultdict(set)
        all_responses = set()
        
        for ex in val_data:
            query_text = ex.texts[0]
            relevant_text = ex.texts[1]
            query_to_relevant[query_text].add(relevant_text)
            all_responses.add(relevant_text)
        
        unique_queries = list(query_to_relevant.keys())
        precision_scores, recall_scores, f1_scores, mrr_scores = [], [], [], []
        
        for query_text in unique_queries:
            gold_docs = list(query_to_relevant[query_text])
            
            if len(gold_docs) == 0:
                continue
            
            # Crear corpus con documentos negativos
            all_other_responses = list(all_responses - set(gold_docs))
            
            if len(all_other_responses) < 20:
                train_responses = set()
                for ex in train_data[:100]:
                    train_responses.add(ex.texts[1])
                all_other_responses.extend(list(train_responses - set(gold_docs)))
            
            if len(all_other_responses) > 30:
                negative_docs = random.sample(all_other_responses, 30)
            else:
                negative_docs = all_other_responses
            
            corpus_docs = gold_docs + negative_docs
            random.shuffle(corpus_docs)
            
            effective_k = min(k, len(corpus_docs))
            
            try:
                # BGE-M3 con normalización
                query_emb = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
                corpus_embs = model.encode(corpus_docs, normalize_embeddings=True, show_progress_bar=False)
                
                # Similitud coseno
                similarities = np.dot(corpus_embs, query_emb)
                ranked_indices = np.argsort(similarities)[::-1]
                
                # Top-k documentos únicos
                top_k_docs = []
                seen = set()
                for i in ranked_indices[:effective_k]:
                    doc = corpus_docs[i]
                    if doc not in seen:
                        seen.add(doc)
                        top_k_docs.append(doc)
                    if len(top_k_docs) >= effective_k:
                        break
                
                # Calcular métricas
                tp = sum(1 for doc in top_k_docs if doc in gold_docs)
                
                precision = tp / len(top_k_docs) if len(top_k_docs) > 0 else 0
                recall = tp / len(gold_docs) if len(gold_docs) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # MRR
                mrr = 0.0
                for i, doc in enumerate([corpus_docs[idx] for idx in ranked_indices]):
                    if doc in gold_docs:
                        mrr = 1.0 / (i + 1)
                        break
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                mrr_scores.append(mrr)
                
            except Exception as e:
                logger.warning(f"Error en evaluación de query: {e}")
                continue
        
        if len(precision_scores) == 0:
            return {"Precision@3": 0.0, "Recall@3": 0.0, "F1@3": 0.0, "MRR": 0.0, "num_queries_evaluated": 0}
        
        return {
            "Precision@3": round(np.mean(precision_scores), 4),
            "Recall@3": round(np.mean(recall_scores), 4),
            "F1@3": round(np.mean(f1_scores), 4),
            "MRR": round(np.mean(mrr_scores), 4),
            "num_queries_evaluated": len(precision_scores)
        }
        
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        return {"Precision@3": 0.0, "Recall@3": 0.0, "F1@3": 0.0, "MRR": 0.0, "num_queries_evaluated": 0}

# ======== ENTRENAMIENTO BGE-M3 ========


best_mrr = 0.0
best_epoch = 0
epochs_without_improvement = 0

train_losses = []
val_mrrs = []
val_f1s = []
epochs_list = []

for epoch in range(1, max_epochs + 1):
    print(f"\n Epoch {epoch}/{max_epochs}")
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 1 else 0,
            show_progress_bar=True,
             use_amp=False,
            optimizer_params={
                "lr": lr,                # Solo learning rate
                "weight_decay": 0.01     # Solo weight decay
            }
        )

        # Evaluación
        print(f" Evaluando época {epoch}...")
        results = evaluate_model_improved(model, val_data)
        current_mrr = results["MRR"]
        current_f1 = results["F1@3"]
        
        print(f" Resultados época {epoch}: {results}")

        # Tracking
        epochs_list.append(epoch)
        train_losses.append(0.1)  # Placeholder
        val_mrrs.append(current_mrr)
        val_f1s.append(current_f1)

        # Early stopping
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Guardar mejor modelo
            best_path = os.path.join(model_save_base, f"epoch{epoch}_MRR{best_mrr:.4f}")
            try:
                model.save(best_path)
                print(f"Mejor modelo BGE-M3 guardado: MRR {best_mrr:.4f} en {best_path}")
            except Exception as e:
                print(f" Error guardando modelo: {e}")
        else:
            epochs_without_improvement += 1
            print(f" Sin mejora. Epochs sin mejora: {epochs_without_improvement}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f" Early stopping activado. Mejor MRR: {best_mrr:.4f} en época {best_epoch}")
            break
            
    except Exception as e:
        print(f" Error en época {epoch}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n Fin del fine-tuning BGE-M3")
print(f" Mejor MRR alcanzado: {best_mrr:.4f} en época {best_epoch}")

# ======== EVALUACIÓN FINAL ========
print(f"\n Evaluación final BGE-M3...")
try:
    best_model_path = os.path.join(model_save_base, f"epoch{best_epoch}_MRR{best_mrr:.4f}")
    
    if os.path.exists(best_model_path):
        print(f" Cargando mejor modelo: epoch{best_epoch}_MRR{best_mrr:.4f}")
        final_model = SentenceTransformer(best_model_path)
        final_results = evaluate_model_improved(final_model, val_data)
    else:
        print(f"Usando modelo de la última época")
        final_results = evaluate_model_improved(model, val_data)
    
    print(f" Resultados finales BGE-M3: {final_results}")
    
except Exception as e:
    print(f" Error en evaluación final: {e}")


# ======== RESUMEN FINAL ========
print(f"\n RESUMEN FINE-TUNING BGE-M3:")
print(f"    Modelos guardados en: {model_save_base}")
print(f"    Mejor MRR: {best_mrr:.4f} (Época {best_epoch})")
print(f"    Queries evaluadas: {final_results.get('num_queries_evaluated', 'N/A')}")
print(f"    Modelo recomendado: epoch{best_epoch}_MRR{best_mrr:.4f}")

# ======== COMPARACIÓN CON BASE ========
print(f"\n Para comparar con BGE-M3 base:")
print(f"   1. Evalúa BGE-M3 original en dataset_test.json")
print(f"   2. Evalúa este modelo fine-tuned en dataset_test.json") 
print(f"   3. Compara MRR@3, Precision@3, Recall@3")
print(f"   4. Documenta si fine-tuning mejora o empeora")

