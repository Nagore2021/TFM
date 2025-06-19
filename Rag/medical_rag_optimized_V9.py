"""
medical_rag_optimized_finetuned.py - RAG Médico Optimizado con Modelo Fine-tuneado

OBJETIVO: Mejor rendimiento usando modelo fine-tuneado médico
PROCESO: Pregunta → Pipeline Híbrido Optimizado → Mejor Chunk → Respuesta Médica
OPTIMIZACIONES: Modelo fine-tuneado + Pool reducido + Transparencia mejorada
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
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
class OptimizedMedicalResponse:
    """Respuesta médica optimizada"""
    question: str
    answer: str
    chunk_used: Dict[str, Any]
    processing_time: float
    success: bool

class OptimizedMedicalRAG:
    """
    RAG Médico Optimizado con Modelo Fine-tuneado
    
    OPTIMIZACIONES IMPLEMENTADAS:
    1. Modelo fine-tuneado médico (más rápido y preciso)
    2. Pipeline híbrido con pool reducido (velocidad)
    3. Transparencia mejorada (no se hace pasar por médico)
    4. Respuestas estructuradas profesionales
    """
    
    def __init__(self, config_path: str, mode: str = "finetuneado"):
        """Inicializa RAG médico optimizado con modelo fine-tuneado"""
        
        self.config_path = config_path
        self.mode = mode
        
        try:
            self.config = cargar_configuracion(config_path)
        except Exception:
            logger.warning("⚠️ Usando configuración por defecto")
            self.config = {}
        
        # Componentes del sistema
        self.retrieval_system = None
        self.generation_pipeline = None
        self.is_initialized = False
        
        logger.info("🎯 RAG Médico Optimizado con Modelo Fine-tuneado - Velocidad + Precisión")

    def initialize(self) -> bool:
        """Inicializa el sistema optimizado"""
        try:
            print("\n🔧 INICIALIZANDO SISTEMA OPTIMIZADO...")
            
            # 1. Cargar sistema con modelo fine-tuneado
            print("📚 Cargando base de conocimientos médicos (modelo fine-tuneado)...")
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            
            total_chunks = len(self.retrieval_system.chunk_ids)
            print(f"✅ Base cargada: {total_chunks} fragmentos médicos disponibles")
            print(f"🎯 Modelo: {self.mode} (optimizado para consultas médicas)")
            
            # 2. No cargar modelo de generación - solo respuestas estructuradas
            print("💡 Usando respuestas estructuradas profesionales (más confiables)")
            self.generation_pipeline = None
            
            self.is_initialized = True
            print("✅ Sistema optimizado listo para consultas médicas\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando: {e}")
            return False

    def ask_doctor(self, medical_question: str) -> OptimizedMedicalResponse:
        """
        Consulta médica optimizada - Proceso transparente y eficiente
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return OptimizedMedicalResponse(
                question=medical_question,
                answer="Sistema no inicializado",
                chunk_used={},
                processing_time=0.0,
                success=False
            )
        
        print(f"\n💬 CONSULTA MÉDICA: {medical_question}")
        print("="*60)
        
        try:
            # PASO 1: Buscar información médica relevante
            print("🔍 PASO 1: Buscando información médica relevante...")
            chunk_info = self._find_best_medical_chunk_optimized(medical_question)
            
            if not chunk_info:
                return OptimizedMedicalResponse(
                    question=medical_question,
                    answer="No encontré información médica relevante para su consulta. Le recomiendo consultar con un profesional médico.",
                    chunk_used={},
                    processing_time=time.time() - start_time,
                    success=False
                )
            
            # PASO 2: Mostrar información encontrada
            self._show_chunk_info_detailed(chunk_info)
            
            # PASO 3: Generar respuesta médica profesional
            print("\n🤖 PASO 3: Generando respuesta médica profesional...")
            medical_answer = self._generate_professional_answer(medical_question, chunk_info)
            
            processing_time = time.time() - start_time
            
            response = OptimizedMedicalResponse(
                question=medical_question,
                answer=medical_answer,
                chunk_used=chunk_info,
                processing_time=processing_time,
                success=True
            )
            
            # PASO 4: Mostrar respuesta final
            self._show_final_response(response)
            
            return response
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return OptimizedMedicalResponse(
                question=medical_question,
                answer=f"Error del sistema: {str(e)}",
                chunk_used={},
                processing_time=time.time() - start_time,
                success=False
            )

    def _find_best_medical_chunk_optimized(self, question: str) -> Optional[Dict[str, Any]]:
        """Busca el mejor fragmento médico usando métodos preparados de BM25DualChunkEvaluator"""
        try:
            print("🎯 Usando métodos de la clase BM25DualChunkEvaluator preparada...")
            
            # OPCIÓN 1: Usar método híbrido preparado (el que da mejor calidad)
            print("🔍 Ejecutando calculate_hybrid_pipeline...")
            hybrid_results = self.retrieval_system.calculate_hybrid_pipeline(
                query=question, 
                pool_size=6,     # Pool reducido para velocidad
                batch_size=4     # Batch pequeño para eficiencia
            )
            
            if hybrid_results:
                best_chunk = hybrid_results[0]
                strategy_name = "Híbrido Preparado (BM25+Bi-Encoder+Cross-Encoder)"
                print(f"✅ Híbrido exitoso: {best_chunk}")
            else:
                # OPCIÓN 2: Fallback a BM25 preparado
                print("⚠️ Híbrido vacío, usando calculate_bm25_rankings...")
                bm25_results = self.retrieval_system.calculate_bm25_rankings(question)
                if bm25_results:
                    best_chunk = bm25_results[0]
                    strategy_name = "BM25 Preparado"
                    hybrid_results = bm25_results
                    print(f"✅ BM25 exitoso: {best_chunk}")
                else:
                    return None
            
            # Obtener información del chunk usando métodos preparados
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk, {})
            
            return {
                "chunk_id": best_chunk,
                "text": chunk_text,
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "document_id": chunk_metadata.get('document_id', ''),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "strategy_used": f"{strategy_name} (modelo {self.mode})",
                "pool_size": 6,
                "model_type": self.mode,
                "all_results": hybrid_results[:5],  # Top 5 para mostrar
                "class_methods": "BM25DualChunkEvaluator preparada"
            }
            
        except Exception as e:
            print(f"❌ Error usando métodos preparados: {e}")
            print("🔄 Fallback a métodos básicos...")
            return self._find_best_medical_chunk_basic_fallback(question)

    def _find_best_medical_chunk_basic_fallback(self, question: str) -> Optional[Dict[str, Any]]:
        """Fallback básico si fallan los métodos preparados"""
        try:
            print("🔄 Usando métodos básicos como fallback...")
            
            # Intentar métodos individuales preparados
            methods_to_try = [
                ("calculate_bm25_rankings", "BM25 Individual"),
                ("calculate_biencoder_rankings", "Bi-Encoder Individual"),
            ]
            
            for method_name, strategy_name in methods_to_try:
                try:
                    if hasattr(self.retrieval_system, method_name):
                        print(f"🔍 Probando {method_name}...")
                        method = getattr(self.retrieval_system, method_name)
                        results = method(question)
                        
                        if results:
                            best_chunk = results[0]
                            chunk_text = self.retrieval_system.docs_raw.get(best_chunk, '')
                            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk, {})
                            
                            print(f"✅ {method_name} exitoso: {best_chunk}")
                            
                            return {
                                "chunk_id": best_chunk,
                                "text": chunk_text,
                                "filename": chunk_metadata.get('filename', 'Guía médica'),
                                "document_id": chunk_metadata.get('document_id', ''),
                                "chunk_position": chunk_metadata.get('chunk_position', ''),
                                "categoria": chunk_metadata.get('categoria', 'medicina'),
                                "strategy_used": f"{strategy_name} Fallback (modelo {self.mode})",
                                "model_type": self.mode,
                                "all_results": results[:5],
                                "class_methods": "BM25DualChunkEvaluator métodos individuales"
                            }
                except Exception as method_error:
                    print(f"⚠️ Error con {method_name}: {method_error}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ Error en fallback básico: {e}")
            return None

    def _find_best_medical_chunk_bm25_fallback(self, question: str) -> Optional[Dict[str, Any]]:
        """Fallback usando método BM25 preparado de la clase"""
        try:
            print("🔄 Usando calculate_bm25_rankings de la clase preparada...")
            bm25_results = self.retrieval_system.calculate_bm25_rankings(question)
            
            if not bm25_results:
                return None
            
            best_chunk = bm25_results[0]
            print(f"✅ BM25 de clase preparada exitoso: {best_chunk}")
            
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk, {})
            
            return {
                "chunk_id": best_chunk,
                "text": chunk_text,
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "document_id": chunk_metadata.get('document_id', ''),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "strategy_used": f"BM25 de clase preparada (modelo {self.mode})",
                "model_type": self.mode,
                "all_results": bm25_results[:5],
                "class_methods": "BM25DualChunkEvaluator.calculate_bm25_rankings"
            }
            
        except Exception as e:
            print(f"❌ Error en BM25 de clase preparada: {e}")
            return None

    def _show_chunk_info_detailed(self, chunk_info: Dict[str, Any]):
        """Muestra información detallada del chunk encontrado"""
        print(f"✅ INFORMACIÓN MÉDICA ENCONTRADA:")
        print(f"   🎯 Estrategia: {chunk_info.get('strategy_used', 'No especificada')}")
        print(f"   🤖 Modelo: {chunk_info.get('model_type', 'No especificado')}")
        print(f"   📄 Documento: {chunk_info['document_id']}")
        print(f"   📂 Archivo: {chunk_info['filename']}")
        print(f"   📍 Posición: {chunk_info['chunk_position']}")
        print(f"   🏷️ Categoría: {chunk_info['categoria']}")
        print(f"   📏 Tamaño: {len(chunk_info['text'])} caracteres")
        
        # Mostrar preview del contenido
        preview = chunk_info['text'][:350] + "..." if len(chunk_info['text']) > 350 else chunk_info['text']
        print(f"   📝 Contenido: {preview}")
        
        # Mostrar alternativas evaluadas
        print(f"   🔄 Top 3 alternativas: {chunk_info['all_results'][:3]}")
        
        # Mostrar métodos de clase utilizados
        class_methods = chunk_info.get('class_methods', 'No especificado')
        pool_size = chunk_info.get('pool_size', 'N/A')
        
        print(f"   🏗️ Métodos utilizados: {class_methods}")
        
        if 'preparada' in class_methods:
            print(f"   ✅ Usando clase BM25DualChunkEvaluator con métodos preparados")
            if pool_size != 'N/A':
                print(f"   ⚡ Pool optimizado: {pool_size} chunks")
        
        if 'finetuneado' in model_type:
            print(f"   🎯 Optimización: Modelo fine-tuneado médico especializado")
            print(f"   ⚡ Ventaja: Velocidad + Precisión + Métodos experimentales preparados")
        else:
            print(f"   💡 Método: Estrategia estándar con métodos preparados")

    def _generate_professional_answer(self, question: str, chunk_info: Dict[str, Any]) -> str:
        """Genera respuesta médica profesional y transparente"""
        
        chunk_text = chunk_info['text']
        source = chunk_info['filename']
        
        print("💡 Generando respuesta estructurada profesional")
        return self._create_professional_structured_answer(question, chunk_text, source)

    def _create_professional_structured_answer(self, question: str, context: str, source: str) -> str:
        """Crea respuesta estructurada profesional y transparente"""
        
        context_preview = context[:600] + "..." if len(context) > 600 else context
        question_lower = question.lower()
        
        # Introducción transparente y profesional
        professional_intro = "Basándome en la información médica disponible en nuestras fuentes especializadas, puedo proporcionarle información relevante sobre su consulta."
        
        # Adaptar respuesta según tipo de consulta
        if any(word in question_lower for word in ['dolor de cabeza', 'cefalea', 'migraña']):
            specific_advice = """
🩺 INFORMACIÓN SOBRE CEFALEAS:
Los dolores de cabeza frecuentes pueden tener múltiples causas y requieren evaluación médica profesional para un diagnóstico preciso.

📋 RECOMENDACIONES INFORMATIVAS:
• Considere mantener un diario de cefaleas: frecuencia, intensidad, duración y posibles desencadenantes
• Factores que pueden influir: patrones de sueño, hidratación, estrés, alimentación
• La evaluación médica es importante para descartar causas subyacentes

⚠️ SIGNOS QUE REQUIEREN ATENCIÓN MÉDICA URGENTE:
• Dolor de cabeza severo y súbito de inicio
• Cefalea con fiebre, rigidez de cuello o alteración de conciencia
• Cambios en la visión, debilidad o dificultades del habla
• Dolor que empeora progresivamente o patrón inusual"""

        elif any(word in question_lower for word in ['diabetes', 'azúcar', 'sed', 'orinar']):
            specific_advice = """
🩺 INFORMACIÓN SOBRE SÍNTOMAS RELACIONADOS CON GLUCOSA:
Los síntomas que menciona pueden estar relacionados con alteraciones metabólicas y requieren evaluación médica.

📋 INFORMACIÓN GENERAL:
• Los análisis de laboratorio son fundamentales para el diagnóstico
• El registro de síntomas puede ser útil para la evaluación médica
• El control metabólico incluye aspectos dietéticos y de actividad física

⚠️ SITUACIONES QUE REQUIEREN ATENCIÓN MÉDICA INMEDIATA:
• Síntomas severos como náuseas, vómitos persistentes
• Dificultad respiratoria o dolor abdominal intenso
• Alteración del estado de conciencia
• Signos de deshidratación severa"""

        elif any(word in question_lower for word in ['presión', 'tensión', 'hipertensión']):
            specific_advice = """
🩺 INFORMACIÓN SOBRE PRESIÓN ARTERIAL:
El control de la presión arterial es fundamental para la salud cardiovascular y requiere seguimiento médico.

📋 INFORMACIÓN GENERAL:
• El diagnóstico de hipertensión requiere mediciones repetidas
• El control incluye aspectos dietéticos, ejercicio y, a menudo, medicación
• El seguimiento médico regular es esencial

⚠️ SITUACIONES QUE REQUIEREN ATENCIÓN MÉDICA:
• Cifras muy elevadas de presión arterial
• Síntomas como dolor de cabeza severo, mareos, problemas visuales
• Dolor torácico o dificultad respiratoria"""

        else:
            specific_advice = """
🩺 INFORMACIÓN MÉDICA GENERAL:
Basándome en su consulta, puedo proporcionarle información general relevante.

📋 RECOMENDACIONES INFORMATIVAS:
• Para una evaluación personalizada, consulte con un profesional médico
• Mantenga un registro de síntomas si es relevante
• Siga las medidas generales de promoción de la salud

⚠️ SIGNOS QUE REQUIEREN ATENCIÓN MÉDICA:
• Síntomas que empeoran o persisten
• Cualquier síntoma que le cause preocupación
• Cambios significativos en su estado de salud"""

        return f"""{professional_intro}

📚 INFORMACIÓN MÉDICA RELEVANTE:
{context_preview}

{specific_advice}

📞 RECOMENDACIONES PARA SEGUIMIENTO:
• Consulte con su médico de cabecera para evaluación personalizada
• Proporcione información completa sobre sus síntomas y antecedentes
• Siga las indicaciones médicas profesionales

💡 *Esta información proviene de: {source}*

ℹ️ IMPORTANTE: Soy un asistente de información médica que proporciona contenido educativo basado en fuentes confiables. Esta información no reemplaza la consulta médica profesional. Para diagnóstico, tratamiento y recomendaciones personalizadas, es fundamental la evaluación directa de un profesional de la salud."""

    def _show_final_response(self, response: OptimizedMedicalResponse):
        """Muestra la respuesta final de forma clara"""
        print(f"\n🔬 RESPUESTA MÉDICA PROFESIONAL:")
        print("="*60)
        print(response.answer)
        print("="*60)
        print(f"⏱️ Tiempo de procesamiento: {response.processing_time:.2f} segundos")
        print(f"✅ Estado: {'Exitoso' if response.success else 'Error'}")
        print(f"🎯 Optimización aplicada: Modelo fine-tuneado médico")


# ============ DEMOSTRACIÓN OPTIMIZADA ============

def main():
    """Demostración del RAG médico optimizado"""
    
    print("🎯 RAG MÉDICO OPTIMIZADO CON MODELO FINE-TUNEADO - DEMOSTRACIÓN")
    print("="*70)
    print("Optimizaciones: Modelo fine-tuneado + Pipeline híbrido + Pool reducido")
    print("Objetivo: Máxima velocidad manteniendo calidad profesional")
    print("="*70)
    
    # Inicializar sistema optimizado
    rag = OptimizedMedicalRAG("../config.yaml", mode="finetuneado")
    
    if not rag.initialize():
        print("❌ Error en inicialización")
        return
    
    # Consultas de prueba (las mismas que funcionaron bien)
    test_questions = [
        "¿Cuáles son los síntomas de la diabetes?",
        "Doctor, tengo dolor de cabeza frecuente",
        "¿Qué puedo hacer para la presión alta?"
    ]
    
    print("🧪 PROBANDO CONSULTAS MÉDICAS OPTIMIZADAS:")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔬 PRUEBA {i}/{len(test_questions)}")
        response = rag.ask_doctor(question)
        
        if not response.success:
            print(f"❌ Error en consulta: {response.answer}")
        
        # Pausa entre consultas para claridad
        if i < len(test_questions):
            input("\n⏸️ Presiona Enter para continuar con la siguiente consulta...")
    
    print(f"\n🎉 DEMOSTRACIÓN COMPLETADA")
    print("Sistema optimizado que combina:")
    print("  🎯 Modelo fine-tuneado: Especializado en medicina")
    print("  🔍 BM25: Búsqueda por palabras clave")
    print("  🧠 Bi-Encoder: Comprensión semántica mejorada") 
    print("  ⚡ Cross-Encoder: Precisión final en pool reducido")
    print("  📋 Transparencia: No se hace pasar por médico")
    print("  🔬 Respuestas profesionales: Información confiable")

if __name__ == "__main__":
    main()