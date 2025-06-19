"""
medical_rag_simple_clear.py - RAG Médico Simple y Claro

OBJETIVO: Sistema fácil de entender sin confusión
PROCESO: Pregunta → BM25 → Mejor Chunk → Respuesta Médica
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
class SimpleMedicalResponse:
    """Respuesta médica simple"""
    question: str
    answer: str
    chunk_used: Dict[str, Any]
    processing_time: float
    success: bool

class SimpleMedicalRAG:
    """
    RAG Médico Simple y Claro
    
    PROCESO SIMPLIFICADO:
    1. Usuario hace pregunta médica
    2. BM25 busca el chunk más relevante
    3. Se muestra qué chunk encontró
    4. Se genera respuesta médica usando ese chunk
    5. Se muestra la respuesta final
    """
    
    def __init__(self, config_path: str, mode: str = "embedding"):
        """Inicializa RAG médico simple"""
        
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
        
        print("🩺 RAG Médico Simple - Fácil de entender")

    def initialize(self) -> bool:
        """Inicializa el sistema de forma simple"""
        try:
            print("\n🔧 INICIALIZANDO SISTEMA...")
            
            # 1. Cargar sistema BM25
            print("📚 Cargando base de conocimientos médicos...")
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            
            total_chunks = len(self.retrieval_system.chunk_ids)
            print(f"✅ Base cargada: {total_chunks} fragmentos médicos disponibles")
            
            # 2. Cargar modelo de generación
            print("🤖 Cargando modelo de respuestas...")
            self._load_generation_model()
            
            self.is_initialized = True
            print("✅ Sistema listo para consultas médicas\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando: {e}")
            return False

    def _load_generation_model(self):
        """Carga modelo de generación optimizado para medicina"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            
            # Intentar modelos más potentes primero
            model_candidates = [
                "microsoft/DialoGPT-large",    # Modelo conversacional potente
                "microsoft/DialoGPT-medium",   # Modelo intermedio
                "gpt2-large",                  # GPT-2 grande
                "gpt2"                         # Fallback básico
            ]
            
            for model_name in model_candidates:
                try:
                    print(f"🤖 Probando modelo: {model_name}")
                    
                    self.generation_pipeline = pipeline(
                        "text-generation",
                        model=model_name,
                        device=device,
                        max_length=1024,  # Contexto amplio para prompts médicos
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True
                    )
                    
                    if self.generation_pipeline.tokenizer.pad_token is None:
                        self.generation_pipeline.tokenizer.pad_token = self.generation_pipeline.tokenizer.eos_token
                        
                    print(f"✅ Modelo cargado exitosamente: {model_name}")
                    break
                    
                except Exception as model_error:
                    print(f"⚠️ Error con {model_name}: {model_error}")
                    continue
            
            if self.generation_pipeline is None:
                print("⚠️ No se pudo cargar ningún modelo, usando respuestas estructuradas")
                
        except Exception as e:
            print(f"⚠️ Error general cargando modelos: {e}")
            print("🔄 Usando modo respuestas estructuradas como médico")
            self.generation_pipeline = None

    def ask_doctor(self, medical_question: str) -> SimpleMedicalResponse:
        """
        Pregunta al doctor - Proceso simple y claro
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return SimpleMedicalResponse(
                question=medical_question,
                answer="Sistema no inicializado",
                chunk_used={},
                processing_time=0.0,
                success=False
            )
        
        print(f"\n💬 PREGUNTA: {medical_question}")
        print("="*60)
        
        try:
            # PASO 1: Buscar información médica relevante
            print("🔍 PASO 1: Buscando información médica relevante...")
            chunk_info = self._find_best_medical_chunk(medical_question)
            
            if not chunk_info:
                return SimpleMedicalResponse(
                    question=medical_question,
                    answer="No encontré información médica relevante para su consulta. Le recomiendo consultar con su médico.",
                    chunk_used={},
                    processing_time=time.time() - start_time,
                    success=False
                )
            
            # PASO 2: Mostrar qué información se encontró
            self._show_chunk_info(chunk_info)
            
            # PASO 3: Generar respuesta médica
            print("\n🤖 PASO 3: Generando respuesta médica...")
            medical_answer = self._generate_medical_answer(medical_question, chunk_info)
            
            processing_time = time.time() - start_time
            
            response = SimpleMedicalResponse(
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
            return SimpleMedicalResponse(
                question=medical_question,
                answer=f"Error del sistema: {str(e)}",
                chunk_used={},
                processing_time=time.time() - start_time,
                success=False
            )

    def _find_best_medical_chunk(self, question: str) -> Optional[Dict[str, Any]]:
        """Busca el mejor fragmento médico usando BM25"""
        try:
            # Usar BM25 para encontrar el mejor chunk
            bm25_results = self.retrieval_system.calculate_bm25_rankings(question)
            
            if not bm25_results:
                return None
            
            # Tomar el mejor resultado
            best_chunk_id = bm25_results[0]
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk_id, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk_id, {})
            
            return {
                "chunk_id": best_chunk_id,
                "text": chunk_text,
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "document_id": chunk_metadata.get('document_id', ''),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "all_results": bm25_results[:5]  # Top 5 para mostrar
            }
            
        except Exception as e:
            print(f"Error en búsqueda BM25: {e}")
            return None

    def _show_chunk_info(self, chunk_info: Dict[str, Any]):
        """Muestra información clara del chunk encontrado"""
        print(f"✅ INFORMACIÓN ENCONTRADA:")
        print(f"   📄 Documento: {chunk_info['document_id']}")
        print(f"   📂 Archivo: {chunk_info['filename']}")
        print(f"   📍 Posición: {chunk_info['chunk_position']}")
        print(f"   🏷️ Categoría: {chunk_info['categoria']}")
        print(f"   📏 Tamaño: {len(chunk_info['text'])} caracteres")
        
        # Mostrar preview del contenido
        preview = chunk_info['text'][:200] + "..." if len(chunk_info['text']) > 200 else chunk_info['text']
        print(f"   📝 Contenido: {preview}")
        
        # Mostrar alternativas consideradas
        print(f"   🎯 Alternativas BM25: {chunk_info['all_results'][:3]}")

    def _generate_medical_answer(self, question: str, chunk_info: Dict[str, Any]) -> str:
        """Genera respuesta médica usando el chunk encontrado"""
        
        chunk_text = chunk_info['text']
        source = chunk_info['filename']
        
        # Intentar generación con modelo si está disponible
        if self.generation_pipeline:
            generated = self._try_generate_with_model(question, chunk_text)
            if generated:
                return f"{generated}\n\n💡 *Información basada en: {source}*"
        
        # Respuesta estructurada como alternativa
        return self._create_structured_answer(question, chunk_text, source)

    def _try_generate_with_model(self, question: str, context: str) -> Optional[str]:
        """Intenta generar respuesta con el modelo usando prompt médico profesional"""
        try:
            # PROMPT MÉDICO PROFESIONAL
            prompt = f"""Eres un médico de atención primaria con amplia experiencia clínica. Tu objetivo es proporcionar respuestas médicas claras, empáticas y basadas en evidencia científica.

INFORMACIÓN MÉDICA DISPONIBLE:
{context[:600]}

CONSULTA DEL PACIENTE:
{question}

INSTRUCCIONES:
- Responde como un médico profesional y empático
- Basa tu respuesta en la información médica proporcionada
- Sé claro y accesible para el paciente
- Incluye recomendaciones prácticas cuando sea apropiado
- Indica cuándo es necesario buscar atención médica urgente
- Mantén un tono tranquilizador pero profesional

RESPUESTA DEL MÉDICO:"""
            
            result = self.generation_pipeline(
                prompt,
                max_new_tokens=200,  # Respuestas más completas
                temperature=0.3,     # Creatividad controlada para precisión médica
                do_sample=True,
                repetition_penalty=1.1,  # Evitar repeticiones
                top_p=0.9,
                top_k=50,
                pad_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                truncation=True
            )
            
            generated = result[0]['generated_text']
            
            # Extraer solo la respuesta del médico
            if "RESPUESTA DEL MÉDICO:" in generated:
                answer = generated.split("RESPUESTA DEL MÉDICO:")[-1].strip()
            else:
                answer = generated[len(prompt):].strip()
            
            # Limpiar tokens especiales
            answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
            
            # Validar que la respuesta sea coherente y médicamente apropiada
            if len(answer) > 30 and not self._has_excessive_repetition(answer) and self._is_medical_response(answer):
                return answer
                    
        except Exception as e:
            print(f"⚠️ Error en generación: {e}")
            
        return None

    def _is_medical_response(self, text: str) -> bool:
        """Verifica si la respuesta parece médicamente apropiada"""
        medical_terms = [
            'paciente', 'síntoma', 'diagnóstico', 'tratamiento', 'consulte', 
            'médico', 'doctor', 'evaluación', 'recomiendo', 'importante',
            'salud', 'clínico', 'atención', 'cuidado'
        ]
        
        text_lower = text.lower()
        medical_content = sum(1 for term in medical_terms if term in text_lower)
        
        # Debe tener al menos 2 términos médicos y estructura coherente
        has_structure = '.' in text or ',' in text
        return medical_content >= 2 and has_structure

    def _has_excessive_repetition(self, text: str) -> bool:
        """Detecta si hay repeticiones excesivas en el texto"""
        words = text.split()
        if len(words) < 5:
            return True
        
        # Verificar repeticiones consecutivas
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
        
        return False

    def _create_structured_answer(self, question: str, context: str, source: str) -> str:
        """Crea respuesta estructurada como médico de atención primaria cuando falla el modelo"""
        
        context_preview = context[:500] + "..." if len(context) > 500 else context
        question_lower = question.lower()
        
        # Respuesta como médico de atención primaria
        medical_intro = "Como médico de atención primaria, comprendo su preocupación y quiero ayudarle."
        
        # Adaptar respuesta según tipo de consulta
        if any(word in question_lower for word in ['dolor de cabeza', 'cefalea', 'migraña']):
            specific_advice = """
🩺 EVALUACIÓN INICIAL:
Los dolores de cabeza frecuentes requieren una evaluación médica adecuada para determinar su causa y el tratamiento más apropiado.

📋 RECOMENDACIONES INMEDIATAS:
• Mantenga un diario de cefaleas: anote cuándo ocurren, intensidad (1-10), duración y posibles desencadenantes
• Asegúrese de mantener una hidratación adecuada y patrones de sueño regulares
• Evite factores desencadenantes comunes como estrés, ayuno prolongado o ciertos alimentos

⚠️ SIGNOS DE ALARMA - BUSQUE ATENCIÓN INMEDIATA SI PRESENTA:
• Dolor de cabeza severo y súbito ("el peor de su vida")
• Cefalea acompañada de fiebre, rigidez de cuello o alteración de la conciencia
• Cambios en la visión, debilidad o dificultad para hablar
• Dolor de cabeza que empeora progresivamente"""

        elif any(word in question_lower for word in ['diabetes', 'azúcar', 'sed', 'orinar']):
            specific_advice = """
🩺 EVALUACIÓN INICIAL:
Los síntomas que describe pueden sugerir alteraciones en los niveles de glucosa y requieren evaluación médica.

📋 RECOMENDACIONES INMEDIATAS:
• Programe una cita para realizarse análisis de glucosa en sangre en ayunas
• Mantenga un registro de síntomas: sed, micción frecuente, cambios en el apetito
• Continúe con una dieta equilibrada y ejercicio moderado según su capacidad

⚠️ SIGNOS DE ALARMA - BUSQUE ATENCIÓN INMEDIATA SI PRESENTA:
• Náuseas o vómitos persistentes
• Dificultad para respirar o dolor abdominal intenso
• Confusión o alteración del nivel de conciencia
• Deshidratación severa"""

        else:
            specific_advice = """
🩺 EVALUACIÓN INICIAL:
Basándome en su consulta y la información médica disponible, le proporciono las siguientes recomendaciones.

📋 RECOMENDACIONES GENERALES:
• Para una evaluación personalizada, programe una cita en consulta
• Mantenga un registro de sus síntomas para facilitar el diagnóstico
• Siga las medidas generales de cuidado de la salud

⚠️ SIGNOS DE ALARMA - BUSQUE ATENCIÓN INMEDIATA SI PRESENTA:
• Síntomas severos o que empeoran rápidamente
• Dificultad respiratoria o dolor torácico
• Alteración del nivel de conciencia o síntomas neurológicos"""

        return f"""{medical_intro}

📚 INFORMACIÓN MÉDICA RELEVANTE:
{context_preview}

{specific_advice}

📞 PRÓXIMOS PASOS:
• Programe una cita en consulta para evaluación presencial
• Traiga consigo cualquier medicación actual y resultados de estudios previos
• No dude en contactar si presenta síntomas de alarma

💡 *Esta respuesta está basada en: {source}*

ℹ️ Recuerde: Esta información es de carácter educativo y no reemplaza la consulta médica presencial. Para un diagnóstico preciso y tratamiento personalizado, es fundamental la evaluación clínica directa."""

    def _show_final_response(self, response: SimpleMedicalResponse):
        """Muestra la respuesta final de forma clara"""
        print(f"\n👨‍⚕️ RESPUESTA MÉDICA:")
        print("="*60)
        print(response.answer)
        print("="*60)
        print(f"⏱️ Tiempo de procesamiento: {response.processing_time:.2f} segundos")
        print(f"✅ Estado: {'Exitoso' if response.success else 'Error'}")


# ============ DEMOSTRACIÓN SIMPLE ============

def main():
    """Demostración simple del RAG médico"""
    
    print("🩺 RAG MÉDICO SIMPLE - DEMOSTRACIÓN")
    print("="*50)
    print("Objetivo: Sistema fácil de entender")
    print("="*50)
    
    # Inicializar sistema
    rag = SimpleMedicalRAG("../config.yaml", mode="embedding")
    
    if not rag.initialize():
        print("❌ Error en inicialización")
        return
    
    # Consultas de prueba simples
    test_questions = [
        "¿Cuáles son los síntomas de la diabetes?",
        "Doctor, tengo dolor de cabeza frecuente",
        "¿Qué puedo hacer para la presión alta?"
    ]
    
    print("🧪 PROBANDO CONSULTAS MÉDICAS:")
    print("="*40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔬 PRUEBA {i}/{len(test_questions)}")
        response = rag.ask_doctor(question)
        
        if not response.success:
            print(f"❌ Error en consulta: {response.answer}")
        
        # Pausa entre consultas para claridad
        if i < len(test_questions):
            input("\n⏸️ Presiona Enter para continuar con la siguiente consulta...")
    
    print(f"\n🎉 DEMOSTRACIÓN COMPLETADA")
    print("Sistema simple que muestra claramente:")
    print("  📚 Qué información encontró")
    print("  🔍 De dónde viene la información") 
    print("  🤖 Cómo genera la respuesta")
    print("  👨‍⚕️ La respuesta médica final")

if __name__ == "__main__":
    main()