"""
medical_rag_productdion.py - RAG Médico para Producción
Este módulo implementa un sistema de Recuperación y Generación (RAG) optimizado para aplicaciones médicas en producción.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

from langchain_huggingface import HuggingFacePipeline

# Imports básicos
import torch
from transformers import pipeline

# Importar la clase existente
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from retrieval.bm25_model_chunk_bge import BM25DualChunkEvaluator
from embeddings.load_model import cargar_configuracion

@dataclass
class MedicalResponse:
    """Respuesta médica del sistema RAG"""
    question: str
    answer: str
    chunk_used: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class MedicalRAGProduction:
    """
    Sistema RAG Médico para Producción
    
    Optimizado para integración con Streamlit:
    - Logging controlado por nivel
    - Sin prints en consola durante uso normal
    - Inicialización silenciosa
    - Respuestas optimizadas para UI
    """
    
    def __init__(self, config_path: str, mode: str = "embedding", verbose: bool = False):
        """
        Inicializa el sistema RAG médico para producción
        
        Args:
            config_path: Ruta al archivo de configuración YAML
            mode: Modo de operación del sistema de recuperación
            verbose: Si True, muestra información de debug (solo para desarrollo)
        """
        
        self.config_path = config_path
        self.mode = mode
        self.verbose = verbose
        
        # Configurar logging según nivel
        log_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        self.config = cargar_configuracion(config_path)
        
        # Componentes del sistema
        self.retrieval_system = None
        self.generation_pipeline = None
        self.model_info = {}
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Inicializa el sistema RAG médico
        Carga el sistema de recuperación y el modelo de generación.
        
        Returns:
            bool: True si inicialización exitosa, False en caso contrario
        """
        try:
            if self.verbose:
                print("Inicializando Sistema RAG Médico...")
            
            # 1. Cargar sistema de recuperación
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            
            if self.verbose:
                total_chunks = len(self.retrieval_system.chunk_ids)
                print(f"Base de conocimientos: {total_chunks} chunks")
            
            # 2. Cargar modelo de generación
            if not self._load_generation_model():
                self.logger.error("Error cargando modelo de generación")
                return False
            
            self.is_initialized = True
            
            if self.verbose:
                print("Sistema RAG inicializado correctamente")
                print(f"Modelo: {self.model_info.get('name', 'No especificado')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en inicialización: {e}")
            return False
    
    def _load_generation_model(self) -> bool:
        """
            Carga el modelo de generación 
        Returns:
            bool: True si carga exitosa, False en caso contrario
        """
        try:

            print(f"Modelo de generación cargado en: {'GPU' if torch.cuda.is_available() else 'CPU'}")

            # Obtener configuración del modelo
            model_config = self.config.get('model', {})
            paths_config = self.config.get('paths', {})
            

            model_name = model_config.get('llm_model')
            if not model_name:
                self.logger.error("'llm_model' no encontrado en config.yaml")
                return False
            
            model_path = paths_config.get('model_path', '../models/')
            
            # Construir ruta del modelo
            if model_name.startswith('models--'):
                full_model_path = os.path.join(model_path, model_name)
            else:
                full_model_path = model_name
            
            # Determinar modelo a cargar
            if os.path.exists(full_model_path):
                model_to_load = full_model_path
            else:
                if model_name.startswith('models--'):
                    model_to_load = model_name.replace('models--', '').replace('--', '/')
                else:
                    model_to_load = model_name
            
            if self.verbose:
                print(f"Cargando modelo: {model_to_load}")
            
            # Configurar dispositivo
            device = 0 if torch.cuda.is_available() else -1
            
            # Cargar pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=model_to_load,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                model_kwargs={
                    "low_cpu_mem_usage": True,  # Reduce uso de memoria en CPU
                    "use_cache": True,  # Habilitar caché para acelerar generación
                }
            )

            self.llm = HuggingFacePipeline(pipeline=self.generation_pipeline)
            
            # Configurar tokens
            if self.generation_pipeline.tokenizer.pad_token is None:
                self.generation_pipeline.tokenizer.pad_token = self.generation_pipeline.tokenizer.eos_token
            
            # Guardar info del modelo
            self.model_info = {
                'name': model_to_load,
                'config_name': model_name,
                'device': "GPU" if torch.cuda.is_available() else "CPU",  # Dispositivo utilizado. 
                'type': 'Qwen2.5-1.5B-Instruct'
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            return False
    
    def ask_doctor(self, medical_question: str) -> MedicalResponse:
        """
        Procesa consulta médica de forma optimizada para UI
        
        Args:
            medical_question: Pregunta médica del usuario
            
        Returns:
            MedicalResponse: Respuesta médica con información relevante
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return MedicalResponse(
                question=medical_question,
                answer="Sistema no inicializado correctamente.",
                chunk_used={},
                processing_time=0.0,
                success=False,
                error_message="Sistema no inicializado"
            )
        
        try:
            # PASO 1: Recuperar información médica (silencioso)
            chunk_info = self._find_medical_information(medical_question)
            
            if not chunk_info:
                return MedicalResponse(
                    question=medical_question,
                    answer="No se encontró información médica relevante para su consulta. Le recomiendo consultar con su médico de cabecera.",
                    chunk_used={},
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Sin información relevante"
                )
            
            # PASO 2: Generar respuesta médica
            medical_answer = self._generate_medical_response(medical_question, chunk_info)
            
            processing_time = time.time() - start_time
            
            response = MedicalResponse(
                question=medical_question,
                answer=medical_answer,
                chunk_used=chunk_info,
                processing_time=processing_time,
                success=True
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error procesando consulta: {e}")
            return MedicalResponse(
                question=medical_question,
                answer="Lo siento, ocurrió un error procesando su consulta. Por favor, intente nuevamente o consulte con personal médico.",
                chunk_used={},
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _find_medical_information(self, question: str) -> Optional[Dict[str, Any]]:
        """Busca información médica relevante """
        try:
            # Pipeline híbrido 
            hybrid_results = self.retrieval_system.calculate_hybrid_pipeline(
                query=question,
                pool_size=10,
                batch_size=8
            )
            
            if not hybrid_results:
                hybrid_results = self.retrieval_system.calculate_bm25_rankings(question)
            
            if not hybrid_results:
                return None
            
            # Obtener mejor resultado
            best_chunk_id = hybrid_results[0]
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk_id, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk_id, {})
            
            return {
                "chunk_id": best_chunk_id,
                "text": chunk_text,
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "document_id": chunk_metadata.get('document_id', ''),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "strategy_used": "Pipeline Híbrido",
                "all_results": hybrid_results[:3]  # Solo top 3 para info
            }
            
        except Exception as e:
            self.logger.error(f"Error en recuperación: {e}")
            return None
    
    def _generate_medical_response(self, question: str, chunk_info: Dict[str, Any]) -> str:
        """Genera respuesta médica de forma silenciosa"""
        
        chunk_text = chunk_info['text']
        source = chunk_info['filename']
        
        try:
            return self._generate_with_model(question, chunk_text, source)
        except Exception as e:
            self.logger.error(f"Error con modelo: {e}")
            return self._create_structured_response(question, chunk_text, source)
    
    def _generate_with_model(self, question: str, context: str, source: str) -> str:
        """Genera respuesta usando modelo sin outputs verbosos"""
        
        # Prompt optimizado
        medical_prompt = f"""<|im_start|>system
Eres una doctora de atención primaria profesional y empática. Proporciona respuestas claras y útiles. Si la pregunta no es médica, responde educadamente que solo atiendes consultas de salud. Con lenguaje médico fácil de entender.<|im_end|>
<|im_start|>user
{question}

Información médica relevante:
{context}<|im_end|>
<|im_start|>assistant
"""
        
        try:
            # Generación optimizada para Streamlit
            response = self.generation_pipeline(
                medical_prompt,
                max_new_tokens=400,  # Respuestas completas pero no excesivas
                temperature=0.3,  # Temperatura baja para respuestas coherentes
                do_sample=True,
                top_p=0.85, # Top-p para diversidad controlada
                repetition_penalty=1.1,  # Penalización de repetición para evitar redundancias
                pad_token_id=self.generation_pipeline.tokenizer.pad_token_id,
            )
            
            # Extraer respuesta limpia
            generated_text = response[0]['generated_text']
            
            if "<|im_start|>assistant" in generated_text:
                medical_answer = generated_text.split("<|im_start|>assistant")[-1]
                medical_answer = medical_answer.replace("<|im_end|>", "").strip()
            else:
                medical_answer = generated_text[len(medical_prompt):].strip()
            
            # Limpiar tokens
            medical_answer = medical_answer.replace("<|endoftext|>", "").strip()
            
            # Formato para Streamlit (más limpio)
            final_answer = f"""{medical_answer}

---

**Información basada en:** {source}

 **Importante:** Esta información es educativa y no reemplaza la consulta médica presencial. Para diagnóstico preciso y tratamiento personalizado, consulte personal médico."""
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"Error en generación: {e}")
            return self._create_structured_response(question, context, source)
    
    def _create_structured_response(self, question: str, context: str, source: str) -> str:
        """Respuesta estructurada para cuando falla el modelo"""
        
        context_preview = context[:400] + "..." if len(context) > 400 else context
        
        return f"""Como doctora de atención primaria, proporciono la siguiente información basada en la documentación médica disponible:

**Información médica relevante:**
{context_preview}

**Recomendaciones generales:**
• Para una evaluación personalizada, programe una cita médica
• Mantenga un registro de sus síntomas
• Siga las medidas generales de cuidado de la salud
• Busque atención médica inmediata si presenta síntomas severos

**Próximos pasos:**
• Consulta médica presencial para evaluación completa
• Traiga su medicación actual y resultados de estudios previos

---

**Información basada en:** {source}

 **Importante:** Esta información es educativa y no reemplaza la consulta médica presencial."""
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtiene información del sistema para mostrar en UI
        
        Returns:
            Dict con información del sistema para Streamlit
        """
        if not self.is_initialized:
            return {"status": "No inicializado", "error": True}
        
        return {
            "status": "Inicializado correctamente",
            "model": self.model_info.get('type', 'No especificado'),
            "device": self.model_info.get('device', 'No especificado'),
            "chunks_loaded": len(self.retrieval_system.chunk_ids) if self.retrieval_system else 0,
            "error": False
        }
    
    def health_check(self) -> bool:
        """
        Verifica el estado de salud del sistema
        
        Returns:
            bool: True si el sistema está funcionando correctamente
        """
        try:
            if not self.is_initialized:
                return False
            
            if not self.retrieval_system or not self.generation_pipeline:
                return False
            
            # Test rápido silencioso
            test_results = self.retrieval_system.calculate_bm25_rankings("test")
            return len(test_results) > 0
            
        except Exception:
            return False

# Función helper para Streamlit
def create_medical_rag_system(config_path: str = "../config.yaml", verbose: bool = False) -> MedicalRAGProduction:
    """
    Función helper para crear e inicializar el sistema RAG en Streamlit
    
    Args:
        config_path: Ruta al archivo de configuración
        verbose: Si mostrar información de debug
        
    Returns:
        MedicalRAGProduction: Sistema inicializado y listo para usar
    """
    rag_system = MedicalRAGProduction(config_path, mode="embedding", verbose=verbose)
    
    if not rag_system.initialize():
        raise Exception("Error inicializando sistema RAG médico")
    
    return rag_system

# Para testing local (se puede eliminar en producción)
if __name__ == "__main__":
    # Test rápido del sistema
    print("Test del Sistema RAG Médico de Producción")
    
    try:
        rag = create_medical_rag_system(verbose=True)
        
        test_question = "¿Cuáles son los síntomas de la diabetes?"
        response = rag.ask_doctor(test_question)
        
        print(f"Pregunta: {test_question}")
        print(f"Éxito: {response.success}")
        print(f"Tiempo: {response.processing_time:.2f}s")
        print(f"Respuesta: {response.answer[:200]}...")
        
    except Exception as e:
        print(f"Error en test: {e}")