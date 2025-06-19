"""
medical_rag_minimal.py - RAG Médico con MÍNIMAS ESTRATEGIAS

ANÁLISIS DE ELIMINACIÓN DE ESTRATEGIAS:
❌ Cross-Encoder Balanced (50 chunks) → ELIMINAR (es el más lento)
❌ Bi-Encoder completo → SIMPLIFICAR 
✅ BM25 solo → MANTENER (más rápido)
✅ Bi-Encoder directo (sin pool) → MANTENER como opción 2

ESTRATEGIAS FINALES (solo 2):
1. BM25_ONLY: Solo BM25 → top chunk directamente
2. HYBRID_FAST: BM25 + Bi-Encoder (SIN Cross-Encoder)
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

# Imports para Mistral
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
class MedicalConsultation:
    """Consulta médica con métricas de tiempo"""
    question: str
    answer: str
    best_chunk: Dict[str, Any]
    strategy_used: str
    processing_time: float
    success: bool

class MinimalMedicalRAG:
    """
    RAG Médico MÍNIMO - Solo 2 estrategias esenciales
    
    ESTRATEGIA 1: BM25_ONLY 
    - Solo BM25 → mejor chunk → respuesta
    - Tiempo estimado: ~30-60 segundos
    
    ESTRATEGIA 2: HYBRID_FAST
    - BM25 (top 5) + Bi-Encoder (top 5) → mejor de ambos → respuesta
    - SIN Cross-Encoder (ahorra 5-8 minutos)
    - Tiempo estimado: ~2-3 minutos
    """
    
    def __init__(self, config_path: str, mode: str = "embedding"):
        """Inicializa RAG médico mínimo"""
        
        self.config_path = config_path
        self.mode = mode
        
        # Parámetros mínimos
        self.max_new_tokens = 250     # Respuestas cortas
        self.temperature = 0.1
        
        try:
            self.config = cargar_configuracion(config_path)
        except Exception:
            logger.warning("⚠️ Config por defecto")
            self.config = {}
        
        # Modelo simple
        self.mistral_model_name = "microsoft/DialoGPT-medium"  # Modelo pequeño
        
        # Componentes
        self.retrieval_system = None
        self.mistral_pipeline = None
        self.is_initialized = False

        logger.info("🔥 RAG Médico MÍNIMO - Solo 2 estrategias esenciales")

    def initialize(self) -> bool:
        """Inicializa sistema mínimo"""
        try:
            logger.info("🚀 Inicializando RAG MÍNIMO...")
            
            # Sistema de recuperación
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            logger.info("✅ Sistema de recuperación cargado")
            
            # Modelo de generación simple
            device = 0 if torch.cuda.is_available() else -1
            self.mistral_pipeline = pipeline(
                "text-generation",
                model=self.mistral_model_name,
                device=device,
                max_length=512,  # Contexto limitado
                torch_dtype=torch.float32
            )
            
            if self.mistral_pipeline.tokenizer.pad_token is None:
                self.mistral_pipeline.tokenizer.pad_token = self.mistral_pipeline.tokenizer.eos_token
            
            logger.info("✅ Modelo de generación simple cargado")
            
            self.is_initialized = True
            self._log_minimal_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False

    def _log_minimal_stats(self):
        """Muestra estadísticas del sistema mínimo"""
        if self.retrieval_system:
            total_chunks = len(self.retrieval_system.chunk_ids)
            logger.info(f"📊 Sistema MÍNIMO cargado:")
            logger.info(f"   📚 {total_chunks} chunks disponibles")
            logger.info(f"   🎯 2 estrategias: BM25_ONLY + HYBRID_FAST")
            logger.info(f"   ⚡ Sin Cross-Encoder (ahorro: 5-8 min)")

    def consult_bm25_only(self, medical_question: str) -> MedicalConsultation:
        """
        ESTRATEGIA 1: BM25_ONLY - Solo BM25, sin embeddings
        
        Pipeline: Query → BM25 → Top 1 chunk → Respuesta
        Tiempo estimado: 30-60 segundos
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return self._error_consultation(medical_question, "Sistema no inicializado", 0.0)
        
        logger.info(f"🔥 BM25_ONLY: {medical_question[:50]}...")
        
        try:
            # Solo BM25
            logger.debug("1️⃣ Ejecutando BM25...")
            bm25_ranking = self.retrieval_system.calculate_bm25_rankings(medical_question)
            
            if not bm25_ranking:
                return self._error_consultation(
                    medical_question, 
                    "No encontré información relevante.", 
                    time.time() - start_time
                )
            
            # Tomar directamente el primer chunk
            best_chunk_id = bm25_ranking[0]
            best_chunk_info = self._get_chunk_info(best_chunk_id)
            
            # Generar respuesta
            logger.debug("2️⃣ Generando respuesta...")
            medical_response = self._generate_simple_response(medical_question, best_chunk_info)
            
            processing_time = time.time() - start_time
            
            consultation = MedicalConsultation(
                question=medical_question,
                answer=medical_response,
                best_chunk=best_chunk_info,
                strategy_used="BM25_ONLY",
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"✅ BM25_ONLY completado en {processing_time:.1f}s")
            return consultation
            
        except Exception as e:
            logger.error(f"❌ Error BM25_ONLY: {e}")
            return self._error_consultation(medical_question, f"Error: {e}", time.time() - start_time)

    def consult_hybrid_fast(self, medical_question: str) -> MedicalConsultation:
        """
        ESTRATEGIA 2: HYBRID_FAST - BM25 + Bi-Encoder SIN Cross-Encoder
        
        Pipeline: Query → BM25(top 5) + Bi-Encoder(top 5) → Mejor score → Respuesta
        Tiempo estimado: 2-3 minutos (sin Cross-Encoder ahorra 5-8 min)
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return self._error_consultation(medical_question, "Sistema no inicializado", 0.0)
        
        logger.info(f"⚡ HYBRID_FAST: {medical_question[:50]}...")
        
        try:
            # BM25 top 5
            logger.debug("1️⃣ BM25 (top 5)...")
            bm25_ranking = self.retrieval_system.calculate_bm25_rankings(medical_question)
            bm25_top5 = bm25_ranking[:5]
            
            # Bi-Encoder top 5
            logger.debug("2️⃣ Bi-Encoder (top 5)...")
            biencoder_ranking = self.retrieval_system.calculate_biencoder_rankings(medical_question)
            biencoder_top5 = biencoder_ranking[:5]
            
            # Combinar candidatos (SIN Cross-Encoder)
            logger.debug("3️⃣ Combinando candidatos...")
            all_candidates = list(set(bm25_top5 + biencoder_top5))  # Únicos
            
            if not all_candidates:
                return self._error_consultation(
                    medical_question, 
                    "No encontré candidatos relevantes.", 
                    time.time() - start_time
                )
            
            # Seleccionar mejor candidato por posición en rankings
            best_chunk_id = self._select_best_hybrid_candidate(
                all_candidates, bm25_ranking, biencoder_ranking
            )
            
            best_chunk_info = self._get_chunk_info(best_chunk_id)
            
            # Generar respuesta
            logger.debug("4️⃣ Generando respuesta...")
            medical_response = self._generate_simple_response(medical_question, best_chunk_info)
            
            processing_time = time.time() - start_time
            
            consultation = MedicalConsultation(
                question=medical_question,
                answer=medical_response,
                best_chunk=best_chunk_info,
                strategy_used="HYBRID_FAST",
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"✅ HYBRID_FAST completado en {processing_time:.1f}s")
            return consultation
            
        except Exception as e:
            logger.error(f"❌ Error HYBRID_FAST: {e}")
            return self._error_consultation(medical_question, f"Error: {e}", time.time() - start_time)

    def _select_best_hybrid_candidate(self, candidates: List[str], 
                                    bm25_ranking: List[str], 
                                    biencoder_ranking: List[str]) -> str:
        """
        Selecciona mejor candidato híbrido SIN Cross-Encoder
        
        Usa posiciones en rankings BM25 + Bi-Encoder para puntuar
        """
        best_chunk = candidates[0]  # Default
        best_score = float('inf')
        
        for chunk_id in candidates:
            # Posición en BM25 (menor = mejor)
            bm25_pos = bm25_ranking.index(chunk_id) if chunk_id in bm25_ranking else 999
            # Posición en Bi-Encoder (menor = mejor)
            biencoder_pos = biencoder_ranking.index(chunk_id) if chunk_id in biencoder_ranking else 999
            
            # Score combinado (menor = mejor)
            combined_score = bm25_pos + biencoder_pos
            
            if combined_score < best_score:
                best_score = combined_score
                best_chunk = chunk_id
        
        return best_chunk

    def _get_chunk_info(self, chunk_id: str) -> Dict[str, Any]:
        """Obtiene información completa del chunk"""
        chunk_text = self.retrieval_system.docs_raw.get(chunk_id, '')
        chunk_metadata = self.retrieval_system.metadatas.get(chunk_id, {})
        
        return {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "document_id": chunk_metadata.get('document_id', ''),
            "filename": chunk_metadata.get('filename', 'Guía médica'),
            "chunk_position": chunk_metadata.get('chunk_position', ''),
            "categoria": chunk_metadata.get('categoria', 'medicina'),
            "text_length": len(chunk_text)
        }

    def _generate_simple_response(self, question: str, best_chunk: Dict[str, Any]) -> str:
        """Generación simple y rápida"""
        
        filename = best_chunk.get('filename', 'Guía médica')
        chunk_text = best_chunk.get('text', '')
        
        # Contexto muy corto
        context = f"[{filename}]\n{chunk_text[:400]}"  # Solo 400 chars
        
        # Prompt muy directo
        prompt = f"Pregunta médica: {question}\n\nInformación: {context}\n\nRespuesta:"
        
        try:
            response = self.mistral_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.mistral_pipeline.tokenizer.eos_token_id,
                truncation=True
            )
            
            generated = response[0]['generated_text']
            
            # Extraer respuesta
            if "Respuesta:" in generated:
                answer = generated.split("Respuesta:")[-1].strip()
            else:
                answer = generated[len(prompt):].strip()
            
            # Limpiar
            answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
            
            if len(answer) < 10:
                return self._emergency_simple_response(question, filename)
            
            return answer
            
        except Exception as e:
            logger.warning(f"⚠️ Error generación: {e}")
            return self._emergency_simple_response(question, filename)

    def _emergency_simple_response(self, question: str, filename: str) -> str:
        """Respuesta de emergencia simple"""
        return f"""Su consulta "{question}" requiere evaluación médica personalizada.

RECOMENDACIÓN: Consulte con su médico de cabecera.

Si presenta síntomas graves, acuda a urgencias.

*[Respuesta basada en {filename}]*"""

    def _error_consultation(self, question: str, error_msg: str, processing_time: float) -> MedicalConsultation:
        """Consulta de error estándar"""
        return MedicalConsultation(
            question=question,
            answer=error_msg,
            best_chunk={},
            strategy_used="ERROR",
            processing_time=processing_time,
            success=False
        )

    # ============ COMPARACIÓN DE ESTRATEGIAS ============

    def compare_strategies(self, medical_question: str) -> Dict[str, MedicalConsultation]:
        """
        Compara las 2 estrategias mínimas en la misma consulta
        
        Returns:
            Dict con resultados de ambas estrategias
        """
        logger.info(f"🔬 COMPARANDO ESTRATEGIAS: {medical_question[:50]}...")
        
        results = {}
        
        # Estrategia 1: BM25_ONLY
        logger.info("🔥 Probando BM25_ONLY...")
        results["bm25_only"] = self.consult_bm25_only(medical_question)
        
        # # Estrategia 2: HYBRID_FAST  
        # logger.info("⚡ Probando HYBRID_FAST...")
        # results["hybrid_fast"] = self.consult_hybrid_fast(medical_question)
        
        # # Resumen comparativo
        # self._log_comparison_summary(results)
        
        return results

    def _log_comparison_summary(self, results: Dict[str, MedicalConsultation]):
        """Log resumen de comparación"""
        logger.info("\n" + "="*50)
        logger.info("📊 RESUMEN COMPARACIÓN DE ESTRATEGIAS")
        logger.info("="*50)
        
        for strategy_name, consultation in results.items():
            status = "✅" if consultation.success else "❌"
            logger.info(f"{status} {strategy_name.upper()}:")
            logger.info(f"   ⏱️ Tiempo: {consultation.processing_time:.1f}s")
            logger.info(f"   📝 Respuesta: {len(consultation.answer)} caracteres")
            if consultation.best_chunk:
                logger.info(f"   📖 Fuente: {consultation.best_chunk.get('filename', 'N/A')}")
        
        # Recomendar estrategia
        if results.get("bm25_only", {}).success and results.get("hybrid_fast", {}).success:
            bm25_time = results["bm25_only"].processing_time
            hybrid_time = results["hybrid_fast"].processing_time
            
            if bm25_time < 120:  # < 2 minutos
                logger.info(f"\n💡 RECOMENDACIÓN: BM25_ONLY (rápido: {bm25_time:.1f}s)")
            else:
                logger.info(f"\n💡 RECOMENDACIÓN: HYBRID_FAST (mejor calidad)")

    def benchmark_minimal(self, test_queries: List[str]) -> Dict[str, Any]:
        """Benchmark de las estrategias mínimas"""
        
        logger.info(f"🏃‍♂️ BENCHMARK MÍNIMO: {len(test_queries)} consultas")
        
        bm25_times = []
        hybrid_times = []
        bm25_successes = 0
        hybrid_successes = 0
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"📋 Query {i}/{len(test_queries)}")
            
            # BM25 Only
            result_bm25 = self.consult_bm25_only(query)
            bm25_times.append(result_bm25.processing_time)
            if result_bm25.success:
                bm25_successes += 1
            
            # # Hybrid Fast
            # result_hybrid = self.consult_hybrid_fast(query)
            # hybrid_times.append(result_hybrid.processing_time)
            # if result_hybrid.success:
            #     hybrid_successes += 1
        
        return {
            "total_queries": len(test_queries),
            "bm25_only": {
                "avg_time": sum(bm25_times) / len(bm25_times),
                "min_time": min(bm25_times),
                "max_time": max(bm25_times),
                "success_rate": bm25_successes / len(test_queries) * 100,
                "times": bm25_times
            },
            "hybrid_fast": {
                "avg_time": sum(hybrid_times) / len(hybrid_times),
                "min_time": min(hybrid_times),
                "max_time": max(hybrid_times),
                "success_rate": hybrid_successes / len(test_queries) * 100,
                "times": hybrid_times
            }
        }


# ============ DEMOSTRACIÓN ============

def main():
    """Demostración RAG médico mínimo"""
    
    print("🔥 RAG Médico MÍNIMO - Solo 2 Estrategias Esenciales")
    print("="*65)
    print("ESTRATEGIA 1: BM25_ONLY (30-60s)")
    print("ESTRATEGIA 2: HYBRID_FAST - BM25+Bi-Encoder SIN Cross-Encoder (2-3min)")
    print("="*65)
    
    # Inicializar
    minimal_rag = MinimalMedicalRAG("../config.yaml", mode="embedding")
    
    if not minimal_rag.initialize():
        print("❌ Error inicialización")
        return
    
    # Consultas de prueba
    test_queries = [
        "Doctor, tengo sed excesiva y orino mucho. ¿Es diabetes?",
        "Siento dolor en el pecho al caminar. ¿Es grave?",
        "Estoy muy triste y sin energía. ¿Qué hago?",
    ]
    
    print("\n🔬 COMPARACIÓN DE ESTRATEGIAS")
    print("-" * 45)
    
    # Probar primera consulta con ambas estrategias
    comparison = minimal_rag.compare_strategies(test_queries[0])
    
    print(f"\n📋 CONSULTA: {test_queries[0]}")
    print("-" * 60)
    
    for strategy, result in comparison.items():
        print(f"\n🎯 {strategy.upper()}:")
        print(f"   ⏱️ Tiempo: {result.processing_time:.1f} segundos")
        print(f"   ✅ Éxito: {result.success}")
        if result.success:
            print(f"   📖 Fuente: {result.best_chunk.get('filename', 'N/A')}")
            print(f"   📝 Respuesta: {result.answer[:150]}...")
    
    # Benchmark completo
    print(f"\n🏃‍♂️ BENCHMARK COMPLETO")
    print("-" * 30)
    
    benchmark = minimal_rag.benchmark_minimal(test_queries)
    
    print(f"\n📊 RESULTADOS BENCHMARK:")
    print(f"   🔥 BM25_ONLY:")
    print(f"      ⏱️ Promedio: {benchmark['bm25_only']['avg_time']:.1f}s")
    print(f"      ✅ Éxito: {benchmark['bm25_only']['success_rate']:.1f}%")
    
    print(f"   ⚡ HYBRID_FAST:")
    print(f"      ⏱️ Promedio: {benchmark['hybrid_fast']['avg_time']:.1f}s")
    print(f"      ✅ Éxito: {benchmark['hybrid_fast']['success_rate']:.1f}%")
    
    # Recomendación final
    bm25_avg = benchmark['bm25_only']['avg_time']
    hybrid_avg = benchmark['hybrid_fast']['avg_time']
    
    print(f"\n💡 RECOMENDACIÓN FINAL:")
    if bm25_avg < 90:  # < 1.5 minutos
        print(f"   🔥 Usar BM25_ONLY para velocidad máxima ({bm25_avg:.1f}s)")
    else:
        print(f"   ⚡ Usar HYBRID_FAST para mejor equilibrio ({hybrid_avg:.1f}s)")
    
    print(f"\n🎉 ¡Sistema optimizado para velocidad!")
    print(f"   ❌ Eliminado: Cross-Encoder (ahorro: 5-8 minutos)")
    print(f"   ❌ Eliminado: Pool de 50 chunks")
    print(f"   ✅ Mantenido: Solo lo esencial")

if __name__ == "__main__":
    main()