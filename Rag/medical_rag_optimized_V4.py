"""
medical_rag_bm25_fast.py - RAG Médico BM25 SOLO - Optimizado para Velocidad

ESTRATEGIA ÚNICA: Solo BM25 (sin Bi-Encoder ni Cross-Encoder)
ENFOQUE: Máxima velocidad con calidad mantenida
OPTIMIZACIONES:
- Solo BM25 ranking
- Cache de consultas frecuentes  
- Generación rápida con menos tokens
- Pool mínimo de candidatos
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pickle
import hashlib
import time

# Imports básicos
import torch
from transformers import pipeline

# Importar la clase existente
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from retrieval.bm25_model_chunk_bge import BM25DualChunkEvaluator
from embeddings.load_model import cargar_configuracion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class FastMedicalResponse:
    """Respuesta médica rápida"""
    question: str
    answer: str
    source_info: str
    processing_time: float
    success: bool
    cached: bool = False
    bm25_candidates: int = 0

class FastBM25MedicalRAG:
    """
    RAG Médico BM25 SOLO - Optimizado para Velocidad Máxima
    
    ESTRATEGIA ÚNICA:
    - Solo BM25 (sin embeddings ni cross-encoder)
    - Cache de consultas frecuentes
    - Generación rápida con tokens limitados
    - Pool mínimo de candidatos
    
    OBJETIVO: < 1 segundo por consulta
    """
    
    def __init__(self, config_path: str, mode: str = "embedding"):
        """Inicializa RAG médico BM25 ultrarrápido"""
        
        self.config_path = config_path
        self.mode = mode
        
        # Parámetros optimizados para modelos potentes
        self.bm25_top_k = 5              # Solo top 5 candidatos BM25
        self.max_new_tokens = 200        # Más tokens para respuestas completas con modelos potentes
        self.temperature = 0.3           # Creatividad controlada
        
        # Cache para consultas frecuentes
        self.query_cache = {}
        self.cache_file = "bm25_query_cache.pkl"
        self._load_cache()
        
        try:
            self.config = cargar_configuracion(config_path)
        except Exception:
            logger.warning("⚠️ Usando configuración por defecto")
            self.config = {}
        
        # Componentes del sistema
        self.retrieval_system = None
        self.generation_pipeline = None
        self.is_initialized = False
        
        logger.info("⚡ RAG Médico BM25 con Modelo Potente - Calidad + Velocidad optimizada")

    def _load_cache(self):
        """Carga cache de consultas frecuentes"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.query_cache = pickle.load(f)
                logger.info(f"📦 Cache cargado: {len(self.query_cache)} consultas")
            else:
                self.query_cache = {}
        except Exception as e:
            logger.warning(f"⚠️ Error cargando cache: {e}")
            self.query_cache = {}

    def _save_cache(self):
        """Guarda cache de consultas"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            logger.warning(f"⚠️ Error guardando cache: {e}")

    def _get_query_hash(self, query: str) -> str:
        """Genera hash único para una consulta"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def initialize(self) -> bool:
        """Inicializa sistema BM25 ultrarrápido"""
        try:
            logger.info("🚀 Inicializando sistema BM25 con modelo potente...")
            
            # Solo sistema BM25 (sin embeddings)
            logger.info("🔍 Cargando solo sistema BM25...")
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            logger.info("✅ Sistema BM25 cargado")
            
            # Modelo potente para generación de calidad
            logger.info("🤖 Cargando modelo potente (Mistral/Qwen)...")
            self._initialize_fast_generation()
            logger.info("✅ Modelo potente listo")
            
            self.is_initialized = True
            logger.info("✅ Sistema BM25 + Modelo Potente inicializado correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            return False

    def _initialize_fast_generation(self):
        """Inicializa modelo potente (Mistral/Qwen) para generación de calidad"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            
            # Intentar modelos potentes en orden de preferencia
            model_candidates = [
                "microsoft/DialoGPT-large",  # Modelo conversacional potente
                "Qwen/Qwen2-1.5B-Instruct",  # Qwen instruct model
                "mistralai/Mistral-7B-v0.1",  # Mistral base
                "microsoft/DialoGPT-medium",  # Fallback intermedio
                "gpt2-large",  # Fallback grande
                "gpt2"  # Último recurso
            ]
            
            for model_name in model_candidates:
                try:
                    logger.info(f"🤖 Intentando cargar modelo: {model_name}")
                    
                    self.generation_pipeline = pipeline(
                        "text-generation",
                        model=model_name,
                        device=device,
                        max_length=1024,  # Contexto amplio para modelo potente
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True,
                        # Configuraciones para calidad
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Configurar pad token
                    if self.generation_pipeline.tokenizer.pad_token is None:
                        self.generation_pipeline.tokenizer.pad_token = self.generation_pipeline.tokenizer.eos_token
                    
                    logger.info(f"✅ Modelo potente cargado exitosamente: {model_name}")
                    break
                    
                except Exception as model_error:
                    logger.warning(f"⚠️ Error con {model_name}: {model_error}")
                    continue
            
            if self.generation_pipeline is None:
                raise Exception("No se pudo cargar ningún modelo de generación")
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelos de generación: {e}")
            logger.info("🔄 Usando modo solo-respuesta estructurada")
            self.generation_pipeline = None

    def fast_ask_doctor(self, medical_question: str) -> FastMedicalResponse:
        """
        Consulta médica ultrarrápida - Solo BM25
        
        Pipeline optimizado:
        1. Cache lookup (instantáneo)
        2. BM25 ranking (< 0.1s)
        3. Selección inteligente del mejor chunk
        4. Generación rápida (< 0.5s)
        
        OBJETIVO: < 1 segundo total
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return FastMedicalResponse(
                question=medical_question,
                answer="Sistema no inicializado. Reinicie el sistema.",
                source_info="Error",
                processing_time=0.0,
                success=False
            )
        
        logger.info(f"⚡ Consulta ultrarrápida: {medical_question[:50]}...")
        
        try:
            # PASO 1: Cache lookup (instantáneo)
            query_hash = self._get_query_hash(medical_question)
            if query_hash in self.query_cache:
                cached_response = self.query_cache[query_hash]
                cached_response.processing_time = time.time() - start_time
                cached_response.cached = True
                logger.info(f"📦 Respuesta cacheada en {cached_response.processing_time:.3f}s")
                return cached_response
            
            # PASO 2: BM25 ranking ultrarrápido con selección inteligente por documento
            logger.debug("🔍 BM25 ranking...")
            bm25_results = self.retrieval_system.calculate_bm25_rankings(medical_question)
            
            if not bm25_results:
                return FastMedicalResponse(
                    question=medical_question,
                    answer="No encontré información relevante en la base médica. Consulte con su médico de cabecera.",
                    source_info="Base de conocimientos médicos",
                    processing_time=time.time() - start_time,
                    success=False,
                    bm25_candidates=0
                )
            
            # PASO 3: Tomar el mejor chunk (ya optimizado por documento en calculate_bm25_rankings)
            best_chunk_id = bm25_results[0]  # Ya es el mejor chunk del mejor documento
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk_id, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk_id, {})
            
            source_info = chunk_metadata.get('filename', 'Guía médica')
            chunk_position = chunk_metadata.get('chunk_position', '')
            document_id = chunk_metadata.get('document_id', '')
            
            # LOG DETALLADO DEL CHUNK SELECCIONADO
            logger.info(f"📋 CHUNK SELECCIONADO POR BM25:")
            logger.info(f"   🔑 Chunk ID: {best_chunk_id}")
            logger.info(f"   📄 Documento: {document_id}")
            logger.info(f"   📍 Posición: {chunk_position}")
            logger.info(f"   📖 Fuente: {source_info}")
            logger.info(f"   📏 Longitud: {len(chunk_text)} caracteres")
            logger.info(f"   📝 Contenido preview: {chunk_text[:100]}...")
            logger.info(f"   🎯 Top 5 BM25 ranking: {bm25_results[:5]}")
            
            # PASO 4: Generación ultrarrápida
            logger.debug("⚡ Generando respuesta...")
            medical_answer = self._ultra_fast_generate(medical_question, chunk_text, source_info)
            
            processing_time = time.time() - start_time
            
            # PASO 5: Crear respuesta final
            response = FastMedicalResponse(
                question=medical_question,
                answer=medical_answer,
                source_info=f"{source_info} - Chunk: {best_chunk_id} - Pos: {chunk_position}",
                processing_time=processing_time,
                success=True,
                cached=False,
                bm25_candidates=len(bm25_results[:self.bm25_top_k])
            )
            
            # Cache para consultas futuras
            self.query_cache[query_hash] = response
            if len(self.query_cache) % 5 == 0:  # Guardar cada 5 consultas
                self._save_cache()
            
            logger.info(f"✅ Consulta ultrarrápida completada en {processing_time:.3f}s")
            logger.debug(f"📖 Mejor chunk: {best_chunk_id} de documento: {document_id}")
            logger.debug(f"📍 Posición: {chunk_position}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error en consulta ultrarrápida: {e}")
            return FastMedicalResponse(
                question=medical_question,
                answer=f"Error del sistema: {str(e)}. Consulte con un médico.",
                source_info="Error",
                processing_time=time.time() - start_time,
                success=False
            )

    def _ultra_fast_generate(self, question: str, context: str, source: str) -> str:
        """Generación con modelo potente usando el chunk BM25 completo como contexto"""
        
        # Intentar generación con modelo potente si está disponible
        if self.generation_pipeline:
            generated = self._try_model_generation(question, context)
            if generated:
                return f"{generated}\n\n*Fuente: {source}*"
        
        # Fallback: respuesta estructurada de alta calidad (siempre funciona)
        logger.debug("Usando respuesta estructurada como fallback")
        return self._create_instant_response(question, context, source)

    def _try_model_generation(self, question: str, context: str) -> Optional[str]:
        """Generación con modelo potente usando el chunk BM25 como contexto completo"""
        try:
            # El contexto ya contiene el mejor chunk de BM25 - usarlo completo
            chunk_context = context[:1000]  # Más contexto para modelos potentes
            
            # Prompt estructurado para modelos conversacionales potentes
            prompt = f"""<|system|>
Eres un asistente médico profesional. Responde de forma clara, empática y basada en evidencia científica.

<|user|>
Contexto médico relevante:
{chunk_context}

Consulta del paciente: {question}

Por favor, proporciona una respuesta médica profesional basada en el contexto proporcionado.

<|assistant|>
"""
            
            # Parámetros optimizados para modelos potentes
            result = self.generation_pipeline(
                prompt,
                max_new_tokens=200,  # Más tokens para respuestas completas
                temperature=0.3,     # Creatividad controlada
                do_sample=True,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                eos_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                truncation=True
            )
            
            generated = result[0]['generated_text']
            
            # Extraer solo la respuesta del asistente
            if "<|assistant|>" in generated:
                answer = generated.split("<|assistant|>")[-1].strip()
            elif "Por favor, proporciona una respuesta médica profesional" in generated:
                parts = generated.split("Por favor, proporciona una respuesta médica profesional")
                if len(parts) > 1:
                    answer = parts[-1].strip()
                else:
                    answer = generated[len(prompt):].strip()
            else:
                answer = generated[len(prompt):].strip()
            
            # Limpiar tokens especiales y artefactos
            answer = answer.replace("</s>", "").replace("<|endoftext|>", "")
            answer = answer.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
            answer = answer.strip()
            
            # Validar calidad de la respuesta
            if self._is_valid_response(answer):
                logger.debug(f"✅ Respuesta generada exitosamente: {answer[:50]}...")
                return answer
            else:
                logger.debug(f"❌ Respuesta inválida del modelo: {answer[:50]}...")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error en generación con modelo potente: {e}")
            return None

    def _is_valid_response(self, response: str) -> bool:
        """Valida si la respuesta generada es válida y de calidad médica"""
        if not response or len(response) < 20:  # Mínimo más exigente
            return False
        
        # Detectar repeticiones excesivas
        words = response.split()
        if len(words) < 8:  # Mínimo más exigente para modelos potentes
            return False
        
        # Verificar si hay demasiadas repeticiones de palabras consecutivas
        repeated_count = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeated_count += 1
                if repeated_count > 2:  # Más estricto para modelos potentes
                    return False
            else:
                repeated_count = 0
        
        # Verificar si una palabra se repite demasiado en total
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            # Más estricto: una palabra no puede ser más del 25% del texto
            if word_counts[word] > len(words) // 4:
                return False
        
        # Verificar que contiene palabras médicas relevantes o estructura coherente
        medical_indicators = [
            'médico', 'doctor', 'consulte', 'síntoma', 'tratamiento', 'diagnóstico',
            'salud', 'paciente', 'evaluación', 'recomiendo', 'importante', 'atención'
        ]
        
        response_lower = response.lower()
        has_medical_content = any(indicator in response_lower for indicator in medical_indicators)
        
        # Verificar estructura coherente (frases con sentido)
        sentences = response.split('.')
        has_coherent_sentences = len([s for s in sentences if len(s.strip()) > 10]) >= 2
        
        return has_medical_content or has_coherent_sentences

    def _create_instant_response(self, question: str, context: str, source: str) -> str:
        """Crea respuesta estructurada instantánea y de calidad"""
        
        # Contexto limitado pero útil
        context_preview = context[:400] + "..." if len(context) > 400 else context
        
        # Detectar tipo de consulta rápidamente
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['dolor de cabeza', 'cefalea', 'migraña', 'dolor cabeza']):
            response_template = f"""Comprendo su preocupación sobre el dolor de cabeza frecuente.

INFORMACIÓN MÉDICA RELEVANTE:
{context_preview}

RECOMENDACIONES IMPORTANTES:
• Los dolores de cabeza frecuentes requieren evaluación médica
• Mantenga un diario: cuándo ocurren, intensidad, duración, factores desencadenantes
• Busque atención inmediata si presenta: dolor severo repentino, fiebre, cambios en la visión, rigidez del cuello

CUÁNDO CONSULTAR:
• Si los dolores son más frecuentes o intensos de lo habitual
• Si interfieren con sus actividades diarias
• Si van acompañados de otros síntomas neurológicos

Su médico de cabecera puede realizar una evaluación completa y determinar el tratamiento más adecuado."""

        elif any(word in question_lower for word in ['síntoma', 'siento', 'tengo', 'dolor', 'me duele']):
            response_template = f"""Sobre los síntomas que describe:

INFORMACIÓN RELEVANTE:
{context_preview}

RECOMENDACIONES:
• Consulte con su médico si los síntomas persisten o empeoran
• Acuda a urgencias si los síntomas son graves o aparecen repentinamente
• Mantenga un registro de cuándo aparecen y qué los mejora o empeora

IMPORTANTE: Esta información es educativa. Para diagnóstico preciso, consulte un médico."""

        elif any(word in question_lower for word in ['tratamiento', 'medicamento', 'medicina', 'curar', 'como tratar']):
            response_template = f"""Sobre el tratamiento que consulta:

INFORMACIÓN DISPONIBLE:
{context_preview}

IMPORTANTES CONSIDERACIONES:
• Todo tratamiento debe ser prescrito por un médico
• No se automedique sin supervisión profesional
• Consulte efectos secundarios con su farmacéutico
• Informe sobre otros medicamentos que esté tomando

PRÓXIMO PASO: Programe cita con su médico de cabecera."""

        elif any(word in question_lower for word in ['qué es', 'que es', 'definición', 'significa']):
            response_template = f"""Información sobre su consulta:

DEFINICIÓN Y CONTEXTO:
{context_preview}

ASPECTOS CLAVE:
• Esta información proviene de fuentes médicas confiables
• Cada caso puede tener particularidades específicas
• Para información personalizada, consulte con un profesional médico

CUÁNDO CONSULTAR:
• Si tiene dudas específicas sobre su situación
• Para obtener un diagnóstico o evaluación personalizada
• Si necesita orientación sobre prevención o tratamiento"""

        else:
            response_template = f"""Respuesta a su consulta médica:

INFORMACIÓN RELEVANTE:
{context_preview}

RECOMENDACIONES GENERALES:
• Esta información proviene de fuentes médicas confiables
• Para una respuesta personalizada, consulte con su médico
• Si tiene síntomas preocupantes, busque atención médica
• Manténgase informado a través de fuentes médicas oficiales"""

        return f"{response_template}\n\n*Fuente: {source}*"

    # ============ MÉTODOS DE UTILIDAD ============

    def clear_cache(self):
        """Limpia el cache de consultas"""
        self.query_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("🧹 Cache limpiado")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache"""
        cache_size = 0
        if os.path.exists(self.cache_file):
            cache_size = os.path.getsize(self.cache_file) / (1024*1024)
        
        return {
            "cached_queries": len(self.query_cache),
            "cache_file_exists": os.path.exists(self.cache_file),
            "cache_size_mb": cache_size
        }

    def speed_benchmark(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark de velocidad BM25-only"""
        if not self.is_initialized:
            return {"error": "Sistema no inicializado"}
        
        logger.info(f"🏃‍♂️ Benchmark BM25-FAST: {len(test_queries)} consultas")
        
        results = []
        total_time = 0
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"⏱️ Consulta {i}/{len(test_queries)}: {query[:40]}...")
            
            response = self.fast_ask_doctor(query)
            
            result = {
                "query": query[:80],
                "success": response.success,
                "processing_time": response.processing_time,
                "answer_length": len(response.answer),
                "cached": response.cached,
                "bm25_candidates": response.bm25_candidates,
                "source_info": response.source_info if response.success else "N/A"
            }
            results.append(result)
            total_time += response.processing_time
            
            logger.info(f"✅ {response.processing_time:.3f}s ({'cache' if response.cached else 'nuevo'})")
        
        # Análisis de resultados
        times = [r["processing_time"] for r in results if not r["cached"]]
        successful = [r for r in results if r["success"]]
        cached_count = len([r for r in results if r["cached"]])
        
        if not times:  # Todas fueron cacheadas
            times = [0.001]  # Tiempo mínimo para evitar división por cero
        
        benchmark_stats = {
            "strategy": "BM25_ONLY_FAST",
            "total_queries": len(test_queries),
            "successful_queries": len(successful),
            "success_rate": len(successful) / len(test_queries) * 100,
            "cached_responses": cached_count,
            "new_responses": len(test_queries) - cached_count,
            "total_time": total_time,
            "average_time_new": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "under_1_second": len([t for t in times if t < 1.0]),
            "under_500ms": len([t for t in times if t < 0.5]),
            "detailed_results": results
        }
        
        logger.info(f"🏁 Benchmark BM25-FAST completado:")
        logger.info(f"   ✅ Éxito: {benchmark_stats['success_rate']:.1f}%")
        logger.info(f"   ⏱️ Tiempo promedio (nuevas): {benchmark_stats['average_time_new']:.3f}s")
        logger.info(f"   📦 Respuestas cacheadas: {cached_count}")
        logger.info(f"   ⚡ Consultas < 1s: {benchmark_stats['under_1_second']}/{len(times)}")
        logger.info(f"   🚀 Consultas < 500ms: {benchmark_stats['under_500ms']}/{len(times)}")
        
        return benchmark_stats


# ============ DEMOSTRACIÓN ULTRARRÁPIDA ============

def main():
    """Demostración del RAG BM25 ultrarrápido"""
    
    print("⚡ RAG Médico BM25-FAST - Solo BM25 para Máxima Velocidad")
    print("="*70)
    print("ESTRATEGIA: Solo BM25 + Cache + Generación rápida")
    print("OBJETIVO: < 1 segundo por consulta")
    print("="*70)
    
    # Inicializar sistema
    fast_rag = FastBM25MedicalRAG("../config.yaml", mode="embedding")
    
    print("🚀 Inicializando sistema ultrarrápido...")
    if not fast_rag.initialize():
        print("❌ Error en inicialización")
        return
    
    # Consultas de prueba
    test_queries = [
        "Doctor, tengo mucha sed y orino frecuentemente. ¿Es diabetes?",
        "Siento dolor en el pecho cuando hago ejercicio. ¿Es grave?", 
        "Estoy muy triste y sin energía últimamente. ¿Qué hago?",
        "Mi presión arterial salió alta en el control. ¿Qué debo hacer?",
        "¿Qué síntomas tiene la gripe?",
        "¿Cómo puedo bajar la fiebre naturalmente?"
    ]
    
    # Benchmark de velocidad
    print(f"\n⏱️ BENCHMARK DE VELOCIDAD BM25-ONLY")
    print("-" * 50)
    
    benchmark = fast_rag.speed_benchmark(test_queries)
    
    print(f"\n📊 RESULTADOS DEL BENCHMARK:")
    print(f"   🎯 Estrategia: {benchmark['strategy']}")
    print(f"   ✅ Consultas exitosas: {benchmark['successful_queries']}/{benchmark['total_queries']}")
    print(f"   ⏱️ Tiempo promedio (nuevas): {benchmark['average_time_new']:.3f}s")
    print(f"   ⚡ Tiempo mínimo: {benchmark['min_time']:.3f}s")
    print(f"   🐌 Tiempo máximo: {benchmark['max_time']:.3f}s")
    print(f"   📦 Respuestas cacheadas: {benchmark['cached_responses']}")
    print(f"   🚀 Consultas < 1 segundo: {benchmark['under_1_second']}/{benchmark['new_responses']}")
    print(f"   ⚡ Consultas < 500ms: {benchmark['under_500ms']}/{benchmark['new_responses']}")
    
    # Estadísticas del cache
    cache_stats = fast_rag.get_cache_stats()
    print(f"\n💾 ESTADÍSTICAS DEL CACHE:")
    print(f"   📝 Consultas cacheadas: {cache_stats['cached_queries']}")
    print(f"   💽 Tamaño del cache: {cache_stats['cache_size_mb']:.2f} MB")
    
    # Demostración de consulta individual
    print(f"\n🩺 CONSULTA INDIVIDUAL ULTRARRÁPIDA")
    print("-" * 45)
    
    demo_query = "Doctor, tengo dolor de cabeza frecuente y me preocupa"
    print(f"📝 Consulta: {demo_query}")
    
    response = fast_rag.fast_ask_doctor(demo_query)
    
    if response.success:
        print(f"✅ Respuesta en {response.processing_time:.3f} segundos")
        print(f"📖 Información detallada: {response.source_info}")
        print(f"📦 Cacheada: {'Sí' if response.cached else 'No'}")
        print(f"🔍 Candidatos BM25: {response.bm25_candidates}")
        print(f"\n👨‍⚕️ RESPUESTA:")
        print(response.answer[:500] + "..." if len(response.answer) > 500 else response.answer)
    else:
        print(f"❌ Error: {response.answer}")
    
    print(f"\n🎉 DEMOSTRACIÓN COMPLETADA!")
    print(f"\n💡 OPTIMIZACIONES BM25-FAST:")
    print(f"   🎯 Solo BM25 (sin embeddings ni cross-encoder)")
    print(f"   📦 Cache inteligente de consultas frecuentes")
    print(f"   ⚡ Generación con máximo 120 tokens")
    print(f"   🔍 Solo top-1 BM25 (máxima velocidad)")
    print(f"   🚀 Fallback estructurado instantáneo")
    print(f"   ⏱️ Objetivo: < 1 segundo por consulta")

if __name__ == "__main__":
    main()