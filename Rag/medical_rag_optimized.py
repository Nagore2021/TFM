"""
medical_rag_optimized.py - RAG Médico Optimizado con BM25DualChunkEvaluator

Sistema RAG médico enfocado SOLO en la mejor estrategia:
Cross-Encoder Balanced (BM25 + Bi-Encoder → Pool Balanceado → Cross-Encoder)

Aprovecha BM25DualChunkEvaluator existente + Mistral como médico de cabecera.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

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


pool_size = 10
@dataclass
class MedicalConsultation:
    """Consulta médica completa con respuesta"""
    question: str
    answer: str
    best_chunk: Dict[str, Any]
    pipeline_stats: Dict[str, int]
    success: bool

class OptimizedMedicalRAG:
    """
    RAG Médico Optimizado - Solo Cross-Encoder Balanced Strategy
    
    Pipeline fijo y optimizado:
    1. BM25 Rankings → Top 25 chunks
    2. Bi-Encoder Rankings → Top 25 chunks  
    3. Pool Balanceado → 50 chunks candidatos
    4. Cross-Encoder Re-ranking → Mejor chunk
    5. Mistral → Respuesta médica profesional
    """
    
    def __init__(self, config_path: str, mode: str = "embedding"):
        """
        Inicializa RAG médico optimizado
        
        Args:
            config_path: Ruta al config.yaml
            mode: 'embedding' o 'finetuneado'
            mistral_model: Modelo Mistral para generación médica
        """

        self.config_path = config_path
        # Cargar configuración YAML usando loader centralizado

        self.max_new_tokens     =  600    # antes 450 o 600
        self.temperature        =  0.2
        self.repetition_penalty = 1.15
        self.pool_size = pool_size  # Tamaño del pool balanceado
        try:
            self.config = cargar_configuracion(config_path)
        except Exception:
            logger.warning("⚠️ No se pudo cargar config.yaml, usando valores por defecto.")
            self.config = {}

        # Modo embeddings o fine-tune
        self.mode = mode

      
        # ───────── Rutas de modelos ─────────
        default_models_dir = os.path.join(PROJECT_ROOT, 'models')
        base_path = self.config.get('model_path', default_models_dir)
        llm = self.config.get('models', {}).get('llm_model', None)
        if llm:
            self.mistral_model_name = os.path.join(base_path, llm)
        else:
            # Por defecto, usar la versión local del modelo 0.2
            self.mistral_model_name = os.path.join(default_models_dir,
                'models--mistralai--Mistral-7B-Instruct-v0.2')

      
        # Componentes
        self.retrieval_system = None
        self.mistral_pipeline = None
        self.is_initialized = False

        logger.info("🏥 RAG Médico Optimizado - Estrategia: Cross-Encoder Balanced")
        logger.info(f"🤖 Carga modelo: {self.mistral_model_name}")
    def initialize(self) -> bool:
        """
        Inicializa sistema RAG médico optimizado
        
        Returns:
            bool: True si inicialización exitosa
        """
        try:
            logger.info("🚀 Inicializando sistema RAG médico optimizado...")
            
            # 1. Sistema de recuperación (BM25DualChunkEvaluator)
            logger.info("🔍 Cargando sistema de recuperación...")
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            logger.info("✅ BM25DualChunkEvaluator cargado")
            
            # 2. Sistema de generación (Mistral)
            logger.info("🤖 Cargando Mistral...")
            self._initialize_mistral()
            logger.info("✅ Mistral cargado")
            
            self.is_initialized = True
            logger.info("✅ Sistema RAG médico optimizado listo")
            
            # Mostrar estadísticas del sistema
            self._log_system_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            self.is_initialized = False
            return False

    def _initialize_mistral(self):
        """Inicializa pipeline Mistral para generación médica, con fallback si falla la carga local"""

        device = 0 if torch.cuda.is_available() else -1
        try:
            self.mistral_pipeline = pipeline(
                "text-generation",
                model=self.mistral_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=device,
                trust_remote_code=True
            )
            logger.info(f"✅ Modelo cargado desde: {self.mistral_model_name}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el modelo local '{self.mistral_model_name}': {e}")
            # Fallback a un modelo alternativo
            fallback = self.config.get('models', {}).get('fallback_model', 'microsoft/DialoGPT-medium')
            logger.info(f"🔄 Cargando modelo alternativo: {fallback}")
            self.mistral_pipeline = pipeline(
                "text-generation",
                model=fallback,
                torch_dtype=torch.float32,
                device=device
            )
        # Asegurar pad_token
        if self.mistral_pipeline.tokenizer.pad_token is None:
            self.mistral_pipeline.tokenizer.pad_token = self.mistral_pipeline.tokenizer.eos_token






    def _log_system_stats(self):
        """Muestra estadísticas del sistema cargado"""
        if self.retrieval_system:
            total_chunks = len(self.retrieval_system.chunk_ids)
            total_docs = len(set(meta.get('document_id', '') for meta in self.retrieval_system.metadatas.values()))
            
            logger.info(f"📊 Sistema cargado:")
            logger.info(f"   📚 {total_chunks} chunks médicos")
            logger.info(f"   📖 {total_docs} documentos únicos")
            logger.info(f"   🎯 Estrategia: Cross-Encoder Balanced")

    def consult_doctor(self, medical_question: str) -> MedicalConsultation:
        """
        Consulta médica principal - Pipeline optimizado Cross-Encoder Balanced
        
        Pipeline fijo:
        1. BM25 → Top 25 chunks
        2. Bi-Encoder → Top 25 chunks
        3. Pool Balanceado → 50 chunks
        4. Cross-Encoder → Mejor chunk  
        5. Mistral → Respuesta médica
        
        Args:
            medical_question: Consulta médica del paciente
            
        Returns:
            MedicalConsultation con respuesta completa
        """
        if not self.is_initialized:
            return MedicalConsultation(
                question=medical_question,
                answer="❌ Sistema no inicializado. Ejecutar initialize() primero.",
                best_chunk={},
                pipeline_stats={},
                success=False
            )
        
        logger.info(f"🩺 Nueva consulta médica: {medical_question[:60]}...")
        
        try:
            # ============ PIPELINE CROSS-ENCODER BALANCED ============
            
            logger.debug("1️⃣ Ejecutando BM25 rankings...")
            bm25_ranking = self.retrieval_system.calculate_bm25_rankings(medical_question)
            bm25_pool = bm25_ranking[:10]  # Top 25 chunks BM25
            
            logger.debug("2️⃣ Ejecutando Bi-Encoder rankings...")
            biencoder_ranking = self.retrieval_system.calculate_biencoder_rankings(medical_question)
            biencoder_pool = biencoder_ranking[:10]  # Top 25 chunks Bi-Encoder
            
            logger.debug("3️⃣ Creando pool balanceado...")
            balanced_pool = self.retrieval_system.create_balanced_chunk_pool(
                bm25_pool, biencoder_pool, pool_size=pool_size
            )
            
            logger.debug("4️⃣ Re-ranking con Cross-Encoder...")
            final_ranking = self.retrieval_system.calculate_crossencoder_rankings(
                medical_question, balanced_pool
            )
            
            # Estadísticas del pipeline
            stats = {
                "bm25_candidates": len(bm25_pool),
                "biencoder_candidates": len(biencoder_pool),
                "balanced_pool_size": len(balanced_pool),
                "final_ranking_size": len(final_ranking)
            }
            
            if not final_ranking:
                return MedicalConsultation(
                    question=medical_question,
                    answer="Lo siento, no encontré información médica relevante para su consulta. Le recomiendo que acuda a consulta presencial para una evaluación detallada.",
                    best_chunk={},
                    pipeline_stats=stats,
                    success=False
                )
            
            # ============ PREPARAR MEJOR CHUNK ============
            
            best_chunk_id = final_ranking[0]  # Mejor chunk según Cross-Encoder
            
            # Obtener información completa del chunk
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
            
            # ============ GENERAR RESPUESTA MÉDICA ============
            
            logger.debug("5️⃣ Generando respuesta médica...")
            medical_response = self._generate_doctor_response(medical_question, best_chunk_info)
            
            # ============ CONSULTA COMPLETA ============
            
            consultation = MedicalConsultation(
                question=medical_question,
                answer=medical_response,
                best_chunk=best_chunk_info,
                pipeline_stats=stats,
                success=True
            )
            
            logger.info(f"✅ Consulta completada - Chunk: {best_chunk_info['filename']}")
            return consultation
            
        except Exception as e:
            logger.error(f"❌ Error procesando consulta: {e}")
            return MedicalConsultation(
                question=medical_question,
                answer=f"Error interno del sistema: {str(e)}. Por favor, consulte con un profesional médico.",
                best_chunk={},
                pipeline_stats={},
                success=False
            )

    def _generate_doctor_response(self, question: str, best_chunk: Dict[str, Any]) -> str:
        """
        Genera respuesta médica usando Mistral como médico de cabecera
        
        Args:
            question: Consulta del paciente
            best_chunk: Mejor chunk recuperado
            
        Returns:
            Respuesta médica profesional
        """
        
        # Construir contexto médico
        filename = best_chunk.get('filename', 'Guía médica')
        chunk_position = best_chunk.get('chunk_position', '')
        chunk_text = best_chunk.get('text', '')
        categoria = best_chunk.get('categoria', '')
        
        # Contexto estructurado
        medical_context = f"""[Información médica de: {filename}"""
        if chunk_position:
            medical_context += f" - Sección: {chunk_position}"
        if categoria:
            medical_context += f" - Categoría: {categoria}"
        medical_context += f"]\n\n{chunk_text}"""
        
       # Prompt refinado
        prompt = (
        "Eres un médico de cabecera experimentado, empático y profesional.\n"
        "Usa **solo** la siguiente información médica para responder al paciente:\n\n"
        f"{medical_context}\n\n"
        f"Pregunta del paciente: {question}\n\n"
        "Por favor, responde de forma clara y estructurada, incluyendo:\n"
        "  1. Breve explicación de posibles causas.\n"
        "  2. Recomendaciones prácticas (tratamiento inicial / seguimiento).\n"
        "  3. Signos de alarma que requieran ir a urgencias.\n\n"
        "RESPUESTA:"
    )

        # Generación
        response = self.mistral_pipeline(
        prompt,
        max_new_tokens=self.max_new_tokens,   # ej. 600
        temperature=self.temperature,
        do_sample=True,
        repetition_penalty=self.repetition_penalty,
        pad_token_id=self.mistral_pipeline.tokenizer.eos_token_id,
        eos_token_id=self.mistral_pipeline.tokenizer.eos_token_id,
        truncation=True
    )
        generated = response[0]['generated_text']

        # Extraer todo lo que venga después de "RESPUESTA:"
        if "RESPUESTA:" in generated:
            answer = generated.split("RESPUESTA:")[-1].strip()
        else:
            # fallback ligero, evita el full emergency
            answer = generated[len(prompt):].strip()

        # Limpiar tokens sobrantes
        return answer.replace("</s>", "").replace("<|endoftext|>", "").strip()

    def _emergency_medical_response(self, question: str, best_chunk: Dict[str, Any]) -> str:
        """Respuesta médica de emergencia si Mistral falla"""
        filename = best_chunk.get('filename', 'documentación médica')
        categoria = best_chunk.get('categoria', 'medicina general')
        
        return f"""Como su médico de cabecera, he revisado la información disponible en {filename} sobre su consulta: "{question}"

Basándome en la documentación médica de {categoria}, puedo confirmar que su consulta requiere una evaluación médica personalizada. La información que he consultado contiene elementos relevantes para su situación.

**RECOMENDACIONES IMPORTANTES:**

1. **Consulta presencial**: Le recomiendo programar una cita para realizar una evaluación completa de su caso
2. **Síntomas de alarma**: Si presenta síntomas graves o urgentes, acuda inmediatamente a urgencias
3. **Seguimiento**: Mantenga un registro de sus síntomas para la próxima consulta

**IMPORTANTE:** Esta respuesta se basa en información médica general. Un diagnóstico preciso requiere examen físico y evaluación personalizada.

*[Respuesta médica de emergencia - Sistema de generación con limitaciones técnicas]*"""

    # ============ MÉTODOS DE UTILIDAD ============

    def get_retrieval_details(self, question: str) -> Dict[str, Any]:
        """
        Obtiene detalles del pipeline de recuperación para análisis
        
        Args:
            question: Consulta médica
            
        Returns:
            Información detallada del pipeline Cross-Encoder Balanced
        """
        if not self.is_initialized:
            return {"error": "Sistema no inicializado"}
        
        try:
            # Ejecutar pipeline completo con detalles
            bm25_ranking = self.retrieval_system.calculate_bm25_rankings(question)
            biencoder_ranking = self.retrieval_system.calculate_biencoder_rankings(question)
            
            bm25_pool = bm25_ranking[:25]
            biencoder_pool = biencoder_ranking[:25]
            balanced_pool = self.retrieval_system.create_balanced_chunk_pool(bm25_pool, biencoder_pool, pool_size=self.pool_size)
            final_ranking = self.retrieval_system.calculate_crossencoder_rankings(question, balanced_pool)
            
            # Información de los top 5 chunks finales
            top_chunks = []
            for i, chunk_id in enumerate(final_ranking[:5], 1):
                if chunk_id in self.retrieval_system.metadatas:
                    meta = self.retrieval_system.metadatas[chunk_id]
                    top_chunks.append({
                        "rank": i,
                        "chunk_id": chunk_id,
                        "filename": meta.get('filename', ''),
                        "chunk_position": meta.get('chunk_position', ''),
                        "document_id": meta.get('document_id', ''),
                        "categoria": meta.get('categoria', ''),
                        "text_preview": self.retrieval_system.docs_raw.get(chunk_id, '')[:150] + "..."
                    })
            
            return {
                "query": question,
                "strategy": "cross_encoder_balanced",
                "pipeline_steps": {
                    "bm25_total": len(bm25_ranking),
                    "bm25_pool": len(bm25_pool),
                    "biencoder_total": len(biencoder_ranking),
                    "biencoder_pool": len(biencoder_pool),
                    "balanced_pool": len(balanced_pool),
                    "final_ranking": len(final_ranking)
                },
                "top_chunks": top_chunks,
                "system_info": {
                    "total_chunks_available": len(self.retrieval_system.chunk_ids),
                    "total_documents": len(set(meta.get('document_id', '') for meta in self.retrieval_system.metadatas.values()))
                }
            }
            
        except Exception as e:
            return {"error": f"Error analizando pipeline: {e}"}

    def get_chunk_info(self, chunk_id: str) -> Dict[str, Any]:
        """Obtiene información completa de un chunk específico"""
        if not self.is_initialized:
            return {"error": "Sistema no inicializado"}
        
        if chunk_id not in self.retrieval_system.metadatas:
            return {"error": f"Chunk {chunk_id} no encontrado"}
        
        metadata = self.retrieval_system.metadatas[chunk_id]
        text = self.retrieval_system.docs_raw.get(chunk_id, '')
        
        return {
            "chunk_id": chunk_id,
            "metadata": metadata,
            "full_text": text,
            "text_stats": {
                "length": len(text),
                "words": len(text.split()),
                "lines": len(text.split('\n'))
            }
        }

    def system_health_check(self) -> Dict[str, Any]:
        """Verifica el estado de salud del sistema"""
        health = {
            "status": "healthy" if self.is_initialized else "unhealthy",
            "timestamp": None,
            "components": {}
        }
        
        if self.retrieval_system:
            health["components"]["retrieval"] = {
                "status": "ok",
                "collection_loaded": self.retrieval_system.collection is not None,
                "chunks_count": len(self.retrieval_system.chunk_ids),
                "models": {
                    "bm25": self.retrieval_system.bm25 is not None,
                    "biencoder": self.retrieval_system.biencoder is not None,
                    "cross_encoder": self.retrieval_system.cross_encoder is not None
                }
            }
        else:
            health["components"]["retrieval"] = {"status": "not_loaded"}
        
        health["components"]["generation"] = {
            "status": "ok" if self.mistral_pipeline else "not_loaded",
            "model": self.mistral_model_name if self.mistral_pipeline else None
        }
        
        return health


# ============ DEMOSTRACIÓN ============

def main():
    """Demostración del RAG médico optimizado"""
    
    print("🏥 RAG Médico Optimizado - Solo Cross-Encoder Balanced")
    print("="*60)
    print("Pipeline: BM25(25) + Bi-Encoder(25) → Pool(50) → Cross-Encoder → Mistral")
    print("="*60)
    
    # Configuración
    config_path = "../config.yaml"  # Ajustar según tu estructura
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Inicializar sistema
    medical_rag = OptimizedMedicalRAG(config_path, mode="embedding")
    
    print("🚀 Inicializando sistema optimizado...")
    if not medical_rag.initialize():
        print("❌ Error en inicialización")
        return
    
    # Health check
    health = medical_rag.system_health_check()
    print(f"✅ Estado del sistema: {health['status']}")
    
    # Consultas médicas reales
    medical_consultations = [
        "Doctor, tengo mucha sed, orino frecuentemente y me siento cansado. ¿Podría ser diabetes?"
        # "Siento dolor en el pecho y dificultad para respirar cuando hago ejercicio. ¿Es grave?",
        # "He estado muy triste últimamente, sin energía y he perdido el interés en todo. ¿Qué me pasa?",
        # "Doctor, mi madre tuvo cáncer de mama. ¿Qué puedo hacer para prevenirlo?",
        # "Mi presión arterial salió alta en un chequeo. ¿Qué debo hacer?"
    ]
    
    print(f"\n🩺 CONSULTAS MÉDICAS")
    print("-" * 50)
    
    for i, consultation_text in enumerate(medical_consultations[:3], 1):  # Limitar para demo
        print(f"\n💬 CONSULTA {i}: {consultation_text}")
        print("-" * 65)
        
        try:
            # Consultar al sistema médico
            consultation = medical_rag.consult_doctor(consultation_text)
            
            if consultation.success:
                print(f"✅ Consulta exitosa")
                
                # Información del chunk usado
                chunk = consultation.best_chunk
                print(f"📖 Fuente: {chunk['filename']} - {chunk['chunk_position']}")
                print(f"📊 Pipeline: {consultation.pipeline_stats}")
                
                # Respuesta del médico
                print(f"\n👨‍⚕️ RESPUESTA DEL MÉDICO DE CABECERA:")
                print(f"{consultation.answer}")
                
            else:
                print(f"❌ Error en consulta: {consultation.answer}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("="*70)
    
    # Análisis detallado del pipeline
    print(f"\n🔍 ANÁLISIS DEL PIPELINE")
    print("-" * 40)
    
    test_query = medical_consultations[0]
    details = medical_rag.get_retrieval_details(test_query)
    
    if "error" not in details:
        print(f"Query: {test_query}")
        print(f"Pipeline: {details['pipeline_steps']}")
        print(f"Sistema: {details['system_info']}")
        
        print(f"\nTop 3 chunks recuperados:")
        for chunk in details["top_chunks"][:3]:
            print(f"  {chunk['rank']}. {chunk['filename']} ({chunk['chunk_position']})")
            print(f"     {chunk['text_preview'][:80]}...")
    
    print(f"\n🎉 Demostración completada!")
    print(f"\n💡 Sistema optimizado:")
    print(f"   🎯 Una sola estrategia: Cross-Encoder Balanced")
    print(f"   ⚡ Pipeline eficiente y enfocado")
    print(f"   🏥 Respuestas médicas profesionales")
    print(f"   🔧 Fácil mantenimiento y debugging")

if __name__ == "__main__":
    main()