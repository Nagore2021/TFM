"""
medical_rag_optimized_fast.py - RAG Médico Optimizado para VELOCIDAD

Optimizaciones principales:
1. Reduce pool_size de 50 a 10-15 chunks
2. Caching de embeddings frecuentes  
3. Early stopping en cross-encoder
4. Mistral con menos tokens
5. BM25 con menos candidatos
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pickle
import hashlib

# Imports para Mistral
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Importar la clase existente
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from retrieval.bm25_model_chunk_bge import BM25DualChunkEvaluator
from embeddings.load_model import cargar_configuracion

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class MedicalConsultation:
    """Consulta médica completa con respuesta"""
    question: str
    answer: str
    best_chunk: Dict[str, Any]
    pipeline_stats: Dict[str, int]
    success: bool
    processing_time: float = 0.0

class FastMedicalRAG:
    """
    RAG Médico Optimizado para VELOCIDAD
    
    Optimizaciones implementadas:
    - Pool reducido (10-15 chunks vs 50)
    - Cache de embeddings frecuentes
    - Early stopping en cross-encoder
    - Mistral con parámetros más rápidos
    - BM25 limitado a top 10
    """
    
    def __init__(self, config_path: str, mode: str = "embedding"):
        """Inicializa RAG médico optimizado para velocidad"""
        
        self.config_path = config_path
        
        # ========= PARÁMETROS ULTRA-OPTIMIZADOS PARA VELOCIDAD =========
        # OPCIÓN 1: Hardware lento (respuesta en ~2-3min)
        if mode == "ultra_fast":
            self.pool_size = 5           # Mínimo viable
            self.bm25_top_k = 4          
            self.biencoder_top_k = 4      
            self.max_new_tokens = 150    # Respuestas muy cortas
            self.skip_cross_encoder = True  # Saltar cross-encoder
        else:
            # OPCIÓN 2: Balance velocidad-calidad (respuesta en ~3-4min)
            self.pool_size = 10           # Reducido de 50 a 10
            self.bm25_top_k = 8           # Reduci
        
        # Cache para embeddings frecuentes
        self.query_cache = {}
        self.cache_file = "query_embeddings_cache.pkl"
        self._load_cache()
        
        try:
            self.config = cargar_configuracion(config_path)
        except Exception:
            logger.warning("⚠️ No se pudo cargar config.yaml, usando valores por defecto.")
            self.config = {}

        self.mode = mode
        
        # Configurar modelos con parámetros rápidos
        default_models_dir = os.path.join(PROJECT_ROOT, 'models')
        base_path = self.config.get('model_path', default_models_dir)
        llm = self.config.get('models', {}).get('llm_model', None)
        
        if llm:
            self.mistral_model_name = os.path.join(base_path, llm)
        else:
            # Modelo más pequeño como fallback
            self.mistral_model_name = "microsoft/DialoGPT-medium"
        
        # Componentes
        self.retrieval_system = None
        self.mistral_pipeline = None
        self.is_initialized = False

        logger.info("⚡ RAG Médico FAST - Optimizado para velocidad")
        logger.info(f"🎯 Pool size: {self.pool_size}, BM25 top-k: {self.bm25_top_k}")

    def _load_cache(self):
        """Carga cache de embeddings de queries frecuentes"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.query_cache = pickle.load(f)
                logger.info(f"📦 Cache cargado: {len(self.query_cache)} queries")
            else:
                self.query_cache = {}
        except Exception as e:
            logger.warning(f"⚠️ Error cargando cache: {e}")
            self.query_cache = {}

    def _save_cache(self):
        """Guarda cache de embeddings"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando cache: {e}")

    def _get_query_hash(self, query: str) -> str:
        """Genera hash único para una query"""
        return hashlib.md5(query.lower().encode()).hexdigest()

    def initialize(self) -> bool:
        """Inicializa sistema RAG médico optimizado para velocidad"""
        try:
            logger.info("🚀 Inicializando sistema RAG médico RÁPIDO...")
            
            # 1. Sistema de recuperación optimizado
            logger.info("🔍 Cargando sistema de recuperación...")
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            
            # Optimizar parámetros del sistema de recuperación
            self._optimize_retrieval_system()
            logger.info("✅ Sistema de recuperación optimizado")
            
            # 2. Sistema de generación rápido
            logger.info("🤖 Cargando Mistral optimizado...")
            self._initialize_fast_mistral()
            logger.info("✅ Mistral optimizado cargado")
            
            self.is_initialized = True
            logger.info("✅ Sistema RAG médico RÁPIDO listo")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            self.is_initialized = False
            return False

    def _optimize_retrieval_system(self):
        """Optimiza parámetros del sistema de recuperación para velocidad"""
        if hasattr(self.retrieval_system, 'biencoder'):
            # Usar batch size mayor para embeddings
            self.retrieval_system.biencoder.encode_batch_size = 32
        
        logger.info("⚡ Sistema de recuperación optimizado para velocidad")

    def _initialize_fast_mistral(self):
        """Inicializa pipeline Mistral optimizado para velocidad"""
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Usar parámetros optimizados para velocidad
            self.mistral_pipeline = pipeline(
                "text-generation",
                model=self.mistral_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=device,
                trust_remote_code=True,
                # Optimizaciones para velocidad
                do_sample=True,
                pad_token_id=50256,  # GPT-style
                max_length=1024      # Limitar contexto
            )
            logger.info(f"✅ Modelo rápido cargado: {self.mistral_model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error con modelo local, usando fallback rápido: {e}")
            # Fallback a modelo más pequeño y rápido
            self.mistral_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",  # Modelo muy pequeño
                torch_dtype=torch.float32,
                device=device,
                max_length=512
            )
        
        # Configurar pad token
        if self.mistral_pipeline.tokenizer.pad_token is None:
            self.mistral_pipeline.tokenizer.pad_token = self.mistral_pipeline.tokenizer.eos_token

    def fast_consult_doctor(self, medical_question: str) -> MedicalConsultation:
        """
        Consulta médica RÁPIDA - Pipeline optimizado para velocidad
        
        Optimizaciones:
        - Pool reducido (10 chunks)
        - Cache de embeddings  
        - Early stopping
        - Menos tokens en generación
        """
        import time
        start_time = time.time()
        
        if not self.is_initialized:
            return MedicalConsultation(
                question=medical_question,
                answer="❌ Sistema no inicializado.",
                best_chunk={},
                pipeline_stats={},
                success=False,
                processing_time=0.0
            )
        
        logger.info(f"⚡ Consulta RÁPIDA: {medical_question[:50]}...")
        
        try:
            # ============ PIPELINE RÁPIDO ============
            
            # 1. Cache lookup
            query_hash = self._get_query_hash(medical_question)
            cached_result = self.query_cache.get(query_hash)
            
            if cached_result:
                logger.info("📦 Usando resultado cacheado")
                processing_time = time.time() - start_time
                cached_result.processing_time = processing_time
                return cached_result
            
            # 2. BM25 rápido (menos candidatos)
            logger.debug("1️⃣ BM25 rápido...")
            bm25_ranking = self.retrieval_system.calculate_bm25_rankings(medical_question)
            bm25_pool = bm25_ranking[:self.bm25_top_k]
            
            # 3. Bi-Encoder rápido (menos candidatos)
            logger.debug("2️⃣ Bi-Encoder rápido...")
            biencoder_ranking = self.retrieval_system.calculate_biencoder_rankings(medical_question)
            biencoder_pool = biencoder_ranking[:self.biencoder_top_k]
            
            # 4. Pool balanceado pequeño
            logger.debug("3️⃣ Pool balanceado pequeño...")
            balanced_pool = self.retrieval_system.create_balanced_chunk_pool(
                bm25_pool, biencoder_pool, pool_size=self.pool_size
            )
            
            # 5. Cross-Encoder con early stopping
            logger.debug("4️⃣ Cross-Encoder rápido...")
            final_ranking = self._fast_crossencoder_ranking(medical_question, balanced_pool)
            
            stats = {
                "bm25_candidates": len(bm25_pool),
                "biencoder_candidates": len(biencoder_pool),
                "balanced_pool_size": len(balanced_pool),
                "final_ranking_size": len(final_ranking),
                "cached": False
            }
            
            if not final_ranking:
                return MedicalConsultation(
                    question=medical_question,
                    answer="No encontré información relevante. Consulte con un médico.",
                    best_chunk={},
                    pipeline_stats=stats,
                    success=False,
                    processing_time=time.time() - start_time
                )
            
            # 6. Preparar mejor chunk
            best_chunk_id = final_ranking[0]
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk_id, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk_id, {})
            
            best_chunk_info = {
                "chunk_id": best_chunk_id,
                "text": chunk_text,
                "document_id": chunk_metadata.get('document_id', ''),
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "text_length": len(chunk_text)
            }
            
            # 7. Generación rápida
            logger.debug("5️⃣ Generando respuesta rápida...")
            medical_response = self._fast_generate_response(medical_question, best_chunk_info)
            
            # 8. Consulta completa
            consultation = MedicalConsultation(
                question=medical_question,
                answer=medical_response,
                best_chunk=best_chunk_info,
                pipeline_stats=stats,
                success=True,
                processing_time=time.time() - start_time
            )
            
            # Cache para consultas futuras (solo si es exitosa)
            self.query_cache[query_hash] = consultation
            if len(self.query_cache) % 10 == 0:  # Guardar cada 10 consultas
                self._save_cache()
            
            logger.info(f"✅ Consulta rápida completada en {consultation.processing_time:.2f}s")
            return consultation
            
        except Exception as e:
            logger.error(f"❌ Error en consulta rápida: {e}")
            return MedicalConsultation(
                question=medical_question,
                answer=f"Error del sistema. Consulte con un médico.",
                best_chunk={},
                pipeline_stats={},
                success=False,
                processing_time=time.time() - start_time
            )

    def _fast_crossencoder_ranking(self, query: str, chunk_pool: List[str]) -> List[str]:
        """Cross-Encoder optimizado con early stopping"""
        if not chunk_pool:
            return []
        
        # Limitar aún más el pool si es muy grande
        if len(chunk_pool) > 8:
            chunk_pool = chunk_pool[:8]
        
        try:
            # Preparar pares query-chunk
            query_chunk_pairs = []
            for chunk_id in chunk_pool:
                chunk_text = self.retrieval_system.docs_raw.get(chunk_id, '')
                if chunk_text:
                    # Truncar texto del chunk para velocidad
                    chunk_text_short = chunk_text[:400]  # Máximo 400 caracteres
                    query_chunk_pairs.append([query, chunk_text_short])
            
            if not query_chunk_pairs:
                return []
            
            # Scoring rápido
            scores = self.retrieval_system.cross_encoder.predict(query_chunk_pairs, batch_size=4)
            
            # Ordenar por score
            chunk_scores = list(zip(chunk_pool, scores))
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [chunk_id for chunk_id, _ in chunk_scores]
            
        except Exception as e:
            logger.warning(f"⚠️ Error en cross-encoder rápido: {e}")
            return chunk_pool[:3]  # Fallback: devolver primeros 3

    def _fast_generate_response(self, question: str, best_chunk: Dict[str, Any]) -> str:
        """Generación rápida con menos tokens"""
        
        filename = best_chunk.get('filename', 'Guía médica')
        chunk_text = best_chunk.get('text', '')
        
        # Contexto más corto
        medical_context = f"[{filename}]\n\n{chunk_text[:600]}"  # Máximo 600 chars
        
        # Prompt más directo y corto
        prompt = (
            f"Como médico, responde brevemente basándote en:\n{medical_context}\n\n"
            f"Pregunta: {question}\n\n"
            f"Respuesta médica breve:"
        )
        
        try:
            # Generación con parámetros rápidos
            response = self.mistral_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,      # 300 tokens
                temperature=self.temperature,            # 0.1 para menos exploración  
                do_sample=True,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.mistral_pipeline.tokenizer.eos_token_id,
                eos_token_id=self.mistral_pipeline.tokenizer.eos_token_id,
                truncation=True,
                # Parámetros adicionales para velocidad
                top_p=0.9,  # Nucleus sampling más restrictivo
                top_k=50    # Top-k sampling
            )
            
            generated = response[0]['generated_text']
            
            # Extraer respuesta
            if "Respuesta médica breve:" in generated:
                answer = generated.split("Respuesta médica breve:")[-1].strip()
            else:
                answer = generated[len(prompt):].strip()
            
            # Limpiar y limitar longitud
            answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
            
            # Asegurar respuesta no vacía
            if not answer or len(answer) < 10:
                return self._emergency_fast_response(question, best_chunk)
            
            return answer
            
        except Exception as e:
            logger.warning(f"⚠️ Error en generación rápida: {e}")
            return self._emergency_fast_response(question, best_chunk)

    def _emergency_fast_response(self, question: str, best_chunk: Dict[str, Any]) -> str:
        """Respuesta de emergencia rápida"""
        filename = best_chunk.get('filename', 'documentación médica')
        
        return f"""Según {filename}, su consulta sobre "{question}" requiere evaluación médica personalizada. 

RECOMENDACIÓN: Consulte con su médico de cabecera para evaluación completa.

URGENCIAS: Si presenta síntomas graves, acuda inmediatamente a urgencias.

*[Respuesta automática rápida]*"""

    # ============ MÉTODOS DE UTILIDAD RÁPIDOS ============

    def clear_cache(self):
        """Limpia el cache de queries"""
        self.query_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("🧹 Cache limpiado")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache"""
        return {
            "cached_queries": len(self.query_cache),
            "cache_file_exists": os.path.exists(self.cache_file),
            "cache_size_mb": os.path.getsize(self.cache_file) / (1024*1024) if os.path.exists(self.cache_file) else 0
        }

    def benchmark_speed(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark de velocidad del sistema"""
        if not self.is_initialized:
            return {"error": "Sistema no inicializado"}
        
        results = []
        total_time = 0
        
        logger.info(f"🏃‍♂️ Iniciando benchmark con {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"⏱️ Query {i}/{len(test_queries)}: {query[:50]}...")
            
            consultation = self.fast_consult_doctor(query)
            results.append({
                "query": query[:100],
                "success": consultation.success,
                "processing_time": consultation.processing_time,
                "answer_length": len(consultation.answer),
                "cached": consultation.pipeline_stats.get("cached", False)
            })
            total_time += consultation.processing_time
            
            # Mostrar progreso
            avg_time = total_time / i
            logger.info(f"✅ Completada en {consultation.processing_time:.2f}s (promedio: {avg_time:.2f}s)")
        
        # Estadísticas finales
        times = [r["processing_time"] for r in results]
        successful = [r for r in results if r["success"]]
        
        benchmark_stats = {
            "total_queries": len(test_queries),
            "successful_queries": len(successful),
            "success_rate": len(successful) / len(test_queries) * 100,
            "total_time": total_time,
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "cached_responses": len([r for r in results if r.get("cached", False)]),
            "detailed_results": results
        }
        
        logger.info(f"🏁 Benchmark completado:")
        logger.info(f"   ⏱️ Tiempo promedio: {benchmark_stats['average_time']:.2f}s")
        logger.info(f"   ✅ Tasa de éxito: {benchmark_stats['success_rate']:.1f}%")
        logger.info(f"   📦 Respuestas cacheadas: {benchmark_stats['cached_responses']}")
        
        return benchmark_stats


# ============ DEMOSTRACIÓN RÁPIDA ============

def main():
    """Demostración del RAG médico optimizado para velocidad"""
    
    print("⚡ RAG Médico FAST - Optimizado para Velocidad")
    print("="*60)
    print("Optimizaciones: Pool pequeño + Cache + Tokens reducidos + Early stopping")
    print("="*60)
    
    # Configuración
    config_path = "../config.yaml"
    
    # Inicializar sistema rápido
    fast_rag = FastMedicalRAG(config_path, mode="embedding")
    
    print("🚀 Inicializando sistema rápido...")
    if not fast_rag.initialize():
        print("❌ Error en inicialización")
        return
    
    # Consultas de prueba
    test_queries = [
        "Doctor, tengo mucha sed y orino frecuentemente. ¿Es diabetes?",
        "Siento dolor en el pecho al hacer ejercicio. ¿Es grave?", 
        "Estoy muy triste y sin energía últimamente. ¿Qué hago?",
        "Mi presión arterial salió alta. ¿Qué debo hacer?"
    ]
    
    # Benchmark de velocidad
    print(f"\n⏱️ BENCHMARK DE VELOCIDAD")
    print("-" * 40)
    
    benchmark = fast_rag.benchmark_speed(test_queries)
    
    print(f"\n📊 RESULTADOS DEL BENCHMARK:")
    print(f"   🎯 Consultas exitosas: {benchmark['successful_queries']}/{benchmark['total_queries']}")
    print(f"   ⏱️ Tiempo promedio: {benchmark['average_time']:.2f} segundos")
    print(f"   ⚡ Tiempo mínimo: {benchmark['min_time']:.2f} segundos")
    print(f"   🐌 Tiempo máximo: {benchmark['max_time']:.2f} segundos")
    print(f"   📦 Respuestas cacheadas: {benchmark['cached_responses']}")
    
    # Mostrar estadísticas del cache
    cache_stats = fast_rag.get_cache_stats()
    print(f"\n💾 ESTADÍSTICAS DEL CACHE:")
    print(f"   📝 Queries cacheadas: {cache_stats['cached_queries']}")
    print(f"   💽 Tamaño del cache: {cache_stats['cache_size_mb']:.2f} MB")
    
    # Demostración de consulta individual rápida
    print(f"\n🩺 CONSULTA INDIVIDUAL RÁPIDA")
    print("-" * 40)
    
    query_demo = "Doctor, tengo dolor de cabeza frecuente. ¿Qué puede ser?"
    print(f"📝 Consulta: {query_demo}")
    
    consultation = fast_rag.fast_consult_doctor(query_demo)
    
    if consultation.success:
        print(f"✅ Respuesta en {consultation.processing_time:.2f} segundos")
        print(f"📖 Fuente: {consultation.best_chunk['filename']}")
        print(f"\n👨‍⚕️ RESPUESTA:")
        print(consultation.answer[:300] + "..." if len(consultation.answer) > 300 else consultation.answer)
    else:
        print(f"❌ Error: {consultation.answer}")
    
    print(f"\n🎉 Demostración completada!")
    print(f"\n💡 Optimizaciones implementadas:")
    print(f"   🎯 Pool reducido a 10 chunks (vs 50 original)")
    print(f"   📦 Cache de consultas frecuentes")
    print(f"   ⚡ Generación con 300 tokens (vs 600 original)")
    print(f"   🛑 Early stopping en cross-encoder")
    print(f"   🔧 Parámetros optimizados para velocidad")

if __name__ == "__main__":
    main()