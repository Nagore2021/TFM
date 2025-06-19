"""
medical_rag_simple_clear.py - RAG Médico Simple y Claro

OBJETIVO: Sistema fácil de entender sin confusión
PROCESO: Pregunta → Pipeline Híbrido → Mejor Chunk → Respuesta Médica
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
    RAG Médico Simple y Claro con Pipeline Híbrido
    

    1. Usuario hace pregunta médica
    2. Pipeline Híbrido busca el chunk más relevante (BM25 + Bi-Encoder + Cross-Encoder)
    3. Se muestra qué chunk encontró
    4. Se genera respuesta médica usando ese chunk
    5. Se muestra la respuesta final
    """
    
    def __init__(self, config_path: str, mode: str = "embedding"):
        """Inicializa RAG médico simple"""
        
        self.config_path = config_path
        self.mode = mode
        
        self.config = cargar_configuracion(config_path)
        
        # Componentes del sistema
        self.retrieval_system = None
        self.generation_pipeline = None
        self.is_initialized = False
    

    def initialize(self) -> bool:
        """Inicializa el sistema de forma simple"""
        try:
           
            
            # 1. Cargar sistema BM25
            print("Cargando base de conocimientos médicos...")
            self.retrieval_system = BM25DualChunkEvaluator(self.config_path, self.mode)
            self.retrieval_system.load_collection()
            
            total_chunks = len(self.retrieval_system.chunk_ids)
           
            # 2. Cargar modelo de generación
            print("Cargando modelo de respuestas...")
            self._load_generation_model()
            
            self.is_initialized = True
            # print("Sistema listo para consultas médicas\n")
            
            return True
            
        except Exception as e:
            print(f"Error inicializando: {e}")
            return False

    

    def _load_generation_model(self):
        """Carga modelos Qwen - FUNCIONAN PERFECTAMENTE para medicina"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            
            # 🥇 QWEN MODELS - EXCELENTES para medicina y SIN RESTRICCIONES
            model_candidates = [
                # 🚀 QWEN 2.5 - MUY ESTABLES Y SIN RESTRICCIONES
                "Qwen/Qwen2.5-7B-Instruct",    # Perfecto para instrucciones médicas
                "Qwen/Qwen2.5-3B-Instruct",    # Más ligero pero potente
                "Qwen/Qwen2.5-1.5B-Instruct",  # Para recursos limitados
                
                # 🔥 QWEN 3 - ÚLTIMA GENERACIÓN (Si funciona en tu setup)
                "Qwen/Qwen3-8B",          # Excelente equilibrio calidad/velocidad
                "Qwen/Qwen3-4B",          # Más rápido, muy bueno
                "Qwen/Qwen3-1.7B",        # Súper eficiente
                
                # 📱 BACKUP ESPECIALIZADOS
                "microsoft/BioGPT-Large",       # Especializado en medicina
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                
                # 🆘 FALLBACK BÁSICOS
                "microsoft/DialoGPT-large",
                "microsoft/DialoGPT-medium"
            ]
            model_candidates = ["Qwen/Qwen2.5-1.5B-Instruct"]
            for model_name in model_candidates:
                try:
                    print(f"🔄 Probando modelo: {model_name}")
                    
                    # CONFIGURACIÓN OPTIMIZADA PARA CADA MODELO
                    if "Qwen3" in model_name:
                        # 🚀 QWEN 3 - ÚLTIMA GENERACIÓN
                        self.generation_pipeline = pipeline(
                            "text-generation",
                            model=model_name,
                            device=device,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            trust_remote_code=True,
                            model_kwargs={
                                "max_position_embeddings": 131072,  # Contexto largo para medicina
                            }
                        )
                        
                    elif "Qwen2.5" in model_name:
                        # 🔥 QWEN 2.5 - INSTRUCT MODELS
                        self.generation_pipeline = pipeline(
                            "text-generation",
                            model=model_name,
                            device=device,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            trust_remote_code=True,
                            model_kwargs={
                                "max_position_embeddings": 32768,  # Contexto amplio
                            }
                        )
                        
                    elif "BioGPT" in model_name:
                        # 🧬 BIOGPT BACKUP
                        self.generation_pipeline = pipeline(
                            "text-generation",
                            model=model_name,
                            device=device,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            trust_remote_code=True,
                            max_length=1024,
                        )
                        
                    else:
                        # 📱 OTROS MODELOS
                        self.generation_pipeline = pipeline(
                            "text-generation",
                            model=model_name,
                            device=device,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            trust_remote_code=True,
                        )
                    
                    # CONFIGURAR TOKENS
                    if self.generation_pipeline.tokenizer.pad_token is None:
                        if hasattr(self.generation_pipeline.tokenizer, 'eos_token'):
                            self.generation_pipeline.tokenizer.pad_token = self.generation_pipeline.tokenizer.eos_token
                        else:
                            self.generation_pipeline.tokenizer.pad_token = "<|endoftext|>"
                    
                    # TEST MÉDICO ESPECÍFICO
                    test_prompt = "Como médico, la diabetes es"
                    test_output = self.generation_pipeline(
                        test_prompt,
                        max_new_tokens=20,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.generation_pipeline.tokenizer.pad_token_id
                    )
                    
                    print(f"✅ Modelo {model_name} cargado correctamente")
                    print(f"🧪 Test médico: {test_output[0]['generated_text'][len(test_prompt):].strip()[:80]}...")
                    break
                    
                except Exception as model_error:
                    print(f"❌ Error con {model_name}: {model_error}")
                    continue
            
            if self.generation_pipeline is None:
                print("❌ No se pudo cargar ningún modelo")
                print("📋 Se usarán solo respuestas estructuradas")
                
        except Exception as e:
            print(f"❌ Error general: {e}")
            self.generation_pipeline = None

    def ask_doctor(self, medical_question: str) -> SimpleMedicalResponse:
        """
        Pregunta al doctor 
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
        
        
        try:
            # PASO 1: Buscar información médica relevante
            print(" PASO 1: Buscando información médica relevante...")
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
            print("\PASO 3: Generando respuesta médica...")
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
            print(f"Error del sistema: {str(e)}")
            return SimpleMedicalResponse(
                question=medical_question,
                answer=f"Error del sistema: {str(e)}",
                chunk_used={},
                processing_time=time.time() - start_time,
                success=False
            )

    def _find_best_medical_chunk(self, question: str) -> Optional[Dict[str, Any]]:
        """Busca el mejor fragmento médico usando Pipeline Híbrido - Confianza total en el algoritmo"""
        try:
            print("Usando Pipeline Híbrido: BM25 + Bi-Encoder + Cross-Encoder...")
            
           
            hybrid_results = self.retrieval_system.calculate_hybrid_pipeline(
                query=question, 
                pool_size=10,    # Pool balanceado de 10 chunks
                batch_size=8     # Procesamiento eficiente
            )
            
            if not hybrid_results:
                print("Pipeline híbrido vacío, usando solo BM25...")
                # Fallback a solo BM25
                hybrid_results = self.retrieval_system.calculate_bm25_rankings(question)
            
            if not hybrid_results:
                return None
            
            # Confiar en el resultado del pipeline híbrido
            best_chunk = hybrid_results[0]
            print(f"Mejor chunk según pipeline híbrido: {best_chunk}")
            
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk, {})
            
            return {
                "chunk_id": best_chunk,
                "text": chunk_text,
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "document_id": chunk_metadata.get('document_id', ''),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "strategy_used": "Pipeline Híbrido (BM25+Bi-Encoder+Cross-Encoder)",
                "all_results": hybrid_results[:5]  # Top 5 para mostrar
            }
            
        except Exception as e:
            print(f"Error en búsqueda híbrida: {e}")
            return self._find_best_medical_chunk_bm25_only(question)

    def _find_best_medical_chunk_bm25_only(self, question: str) -> Optional[Dict[str, Any]]:
        """Fallback: Busca usando solo BM25 - Sin filtros, confianza en el algoritmo"""
        try:
            # Usar BM25 para encontrar candidatos
            bm25_results = self.retrieval_system.calculate_bm25_rankings(question)
            
            if not bm25_results:
                return None
            
            # Confiar en el primer resultado de BM25
            best_chunk = bm25_results[0]
            print(f"Mejor chunk según BM25: {best_chunk}")
            
            chunk_text = self.retrieval_system.docs_raw.get(best_chunk, '')
            chunk_metadata = self.retrieval_system.metadatas.get(best_chunk, {})
            
            return {
                "chunk_id": best_chunk,
                "text": chunk_text,
                "filename": chunk_metadata.get('filename', 'Guía médica'),
                "document_id": chunk_metadata.get('document_id', ''),
                "chunk_position": chunk_metadata.get('chunk_position', ''),
                "categoria": chunk_metadata.get('categoria', 'medicina'),
                "strategy_used": "BM25 ranking directo",
                "all_results": bm25_results[:5]  # Top 5 para mostrar
            }
            
        except Exception as e:
            print(f"Error en búsqueda BM25: {e}")
            return None

    def _show_chunk_info(self, chunk_info: Dict[str, Any]):
        """Muestra información clara del chunk encontrado"""
        print(f" INFORMACIÓN ENCONTRADA:")
        print(f" Estrategia: {chunk_info.get('strategy_used', 'No especificada')}")
        print(f" Documento: {chunk_info['document_id']}")
        print(f" Archivo: {chunk_info['filename']}")
        print(f" Posición: {chunk_info['chunk_position']}")
        print(f"  Categoría: {chunk_info['categoria']}")
        print(f" Tamaño: {len(chunk_info['text'])} caracteres")
        
        # Mostrar preview del contenido
        preview = chunk_info['text'][:300] + "..." if len(chunk_info['text']) > 300 else chunk_info['text']
        print(f"  Contenido: {preview}")
        
        # Mostrar alternativas consideradas
        print(f"  Top 3 resultados: {chunk_info['all_results'][:3]}")
        print(f"  Estrategia: {chunk_info.get('strategy_used', 'No especificada')}")

    def _generate_medical_answer(self, question: str, chunk_info: Dict[str, Any]) -> str:
        """Genera respuesta médica usando el chunk encontrado"""
        
        chunk_text = chunk_info['text']
        source = chunk_info['filename']
        
        # INTENTAR USAR EL MODELO PRIMERO
        if self.generation_pipeline is not None:
            try:
                print("💡 Generando respuesta con modelo + contexto médico...")
                return self._generate_with_model(question, chunk_text, source)
            except Exception as e:
                print(f"❌ Error con modelo: {e}")
                print("💡 Fallback a respuesta estructurada...")
        
        # FALLBACK: respuestas estructuradas
        print("💡 Usando respuesta médica estructurada")
        return self._create_structured_answer(question, chunk_text, source)

    def _generate_with_model(self, question: str, context: str, source: str) -> str:
        """Genera respuesta médica usando modelo especializado"""
        
        if hasattr(self, 'model_info'):
            print(f"🤖 Generando con: {self.model_info['name']} ({self.model_info['device']})")
        else:
            print(f"🤖 Generando con: {self.generation_pipeline.model.name_or_path}")
        model_name = str(self.generation_pipeline.model.name_or_path)
        
        if "Qwen3" in model_name:
            # 🚀 QWEN 3 - FORMATO OPTIMIZADO CON THINKING MODE
            medical_prompt = f"""<|im_start|>system
Eres un médico de atención primaria experto. Responde consultas médicas de forma profesional y cercana usando la información proporcionada.<|im_end|>
<|im_start|>user
CONSULTA MÉDICA: {question}

INFORMACIÓN MÉDICA RELEVANTE:
{context}

Por favor, proporciona una respuesta médica completa que incluya:
1. Evaluación de la consulta
2. Información basada en el contexto médico
3. Recomendaciones apropiadas
4. Cuándo buscar atención médica urgente<|im_end|>
<|im_start|>assistant
"""
            
            generation_params = {
                "max_new_tokens": 300,
                "temperature": 0.3,      # Conservador para medicina
                "do_sample": False,
                # "top_p": 0.8,
                # "repetition_penalty": 1.05,
                "pad_token_id": self.generation_pipeline.tokenizer.pad_token_id,
            }
            
        elif "Qwen2.5" in model_name and "Instruct" in model_name:
            # 🔥 QWEN 2.5 INSTRUCT - FORMATO INSTRUCT
            medical_prompt = f"""<|im_start|>system
Eres un médico de atención primaria profesional y empático.<|im_end|>
<|im_start|>user
{question}

Información médica para consultar:
{context}<|im_end|>
<|im_start|>assistant
"""
            
            generation_params = {
                "max_new_tokens": 500,
                "temperature": 0.4,
                "do_sample": True,
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "pad_token_id": self.generation_pipeline.tokenizer.pad_token_id,
            }
            
        elif "BioGPT" in model_name:
            # 🧬 BIOGPT - FORMATO EN INGLÉS
            medical_prompt = f"""Medical Question: {question}

Relevant Medical Information:
{context}

Medical Answer:"""
            
            generation_params = {
                "max_new_tokens": 300,
                "temperature": 0.2,
                "do_sample": False,
                # "top_p": 0.9,
                "pad_token_id": self.generation_pipeline.tokenizer.pad_token_id,
            }
            
        else:
            # 📱 OTROS MODELOS - FORMATO GENÉRICO
            medical_prompt = f"""Como médico especialista, responde esta consulta médica:

CONSULTA: {question}

INFORMACIÓN: {context}

RESPUESTA MÉDICA:"""
            
            generation_params = {
                "max_new_tokens": 350,
                "temperature": 0.5,
                "do_sample": True,
                "pad_token_id": self.generation_pipeline.tokenizer.pad_token_id,
            }

        try:
            # GENERAR RESPUESTA
            response = self.generation_pipeline(medical_prompt, **generation_params)
            
            # EXTRAER RESPUESTA LIMPIA
            generated_text = response[0]['generated_text']
            
            # LIMPIAR SEGÚN EL MODELO
            if "Qwen" in model_name:
                if "<|im_start|>assistant" in generated_text:
                    medical_answer = generated_text.split("<|im_start|>assistant")[-1]
                    medical_answer = medical_answer.replace("<|im_end|>", "").strip()
                else:
                    medical_answer = generated_text[len(medical_prompt):].strip()
                    
            elif "BioGPT" in model_name:
                if "Medical Answer:" in generated_text:
                    medical_answer = generated_text.split("Medical Answer:")[-1].strip()
                else:
                    medical_answer = generated_text[len(medical_prompt):].strip()
                    
            else:
                medical_answer = generated_text[len(medical_prompt):].strip()
            
            # LIMPIAR TOKENS ESPECIALES
            medical_answer = medical_answer.replace("<|endoftext|>", "")
            medical_answer = medical_answer.replace("<|im_end|>", "")
            medical_answer = medical_answer.strip()
            
            # NOTA PARA BIOGPT SI RESPONDE EN INGLÉS
            language_note = ""
            if "BioGPT" in model_name and any(word in medical_answer.lower() for word in ['the', 'and', 'is', 'are', 'this']):
                language_note = "\n\n*Nota: Respuesta generada por modelo médico especializado BioGPT*"
            
            # FORMATO FINAL CON DISCLAIMERS MÉDICOS
            final_answer = f"""{medical_answer}{language_note}

📋 *Información basada en: {source}*

⚠️ IMPORTANTE: Esta información es de carácter educativo y no reemplaza la consulta médica presencial. Para diagnóstico preciso y tratamiento personalizado, consulte con su médico de cabecera."""
            
            return final_answer
            
        except Exception as e:
            print(f"❌ Error en generación: {e}")
            return self._create_structured_answer(question, context, source)

    def _create_structured_answer(self, question: str, context: str, source: str) -> str:
        """Crea respuesta estructurada como médico de atención primaria cuando falla el modelo"""
        
        context_preview = context[:500] + "..." if len(context) > 500 else context
        question_lower = question.lower()
        
        # Respuesta como médico de atención primaria
        medical_intro = "Como médico de atención primaria, comprendo su preocupación y quiero ayudarle."
        
        # Adaptar respuesta según tipo de consulta
        if any(word in question_lower for word in ['dolor de cabeza', 'cefalea', 'migraña']):
            specific_advice = """
 EVALUACIÓN INICIAL:
Los dolores de cabeza frecuentes requieren una evaluación médica adecuada para determinar su causa y el tratamiento más apropiado.

 RECOMENDACIONES INMEDIATAS:
• Mantenga un diario de cefaleas: anote cuándo ocurren, intensidad (1-10), duración y posibles desencadenantes
• Asegúrese de mantener una hidratación adecuada y patrones de sueño regulares
• Evite factores desencadenantes comunes como estrés, ayuno prolongado o ciertos alimentos

 SIGNOS DE ALARMA - BUSQUE ATENCIÓN INMEDIATA SI PRESENTA:
• Dolor de cabeza severo y súbito ("el peor de su vida")
• Cefalea acompañada de fiebre, rigidez de cuello o alteración de la conciencia
• Cambios en la visión, debilidad o dificultad para hablar
• Dolor de cabeza que empeora progresivamente"""

        elif any(word in question_lower for word in ['diabetes', 'azúcar', 'sed', 'orinar']):
            specific_advice = """
 EVALUACIÓN INICIAL:
Los síntomas que describe pueden sugerir alteraciones en los niveles de glucosa y requieren evaluación médica.

 RECOMENDACIONES INMEDIATAS:
• Programe una cita para realizarse análisis de glucosa en sangre en ayunas
• Mantenga un registro de síntomas: sed, micción frecuente, cambios en el apetito
• Continúe con una dieta equilibrada y ejercicio moderado según su capacidad

 SIGNOS DE ALARMA - BUSQUE ATENCIÓN INMEDIATA SI PRESENTA:
• Náuseas o vómitos persistentes
• Dificultad para respirar o dolor abdominal intenso
• Confusión o alteración del nivel de conciencia
• Deshidratación severa"""

        else:
            specific_advice = """
 EVALUACIÓN INICIAL:
Basándome en su consulta y la información médica disponible, le proporciono las siguientes recomendaciones.

 RECOMENDACIONES GENERALES:
• Para una evaluación personalizada, programe una cita en consulta
• Mantenga un registro de sus síntomas para facilitar el diagnóstico
• Siga las medidas generales de cuidado de la salud

 SIGNOS DE ALARMA - BUSQUE ATENCIÓN INMEDIATA SI PRESENTA:
• Síntomas severos o que empeoran rápidamente
• Dificultad respiratoria o dolor torácico
• Alteración del nivel de conciencia o síntomas neurológicos"""

        return f"""{medical_intro}

 INFORMACIÓN MÉDICA RELEVANTE:
{context_preview}

{specific_advice}

 PRÓXIMOS PASOS:
• Programe una cita en consulta para evaluación presencial
• Traiga consigo cualquier medicación actual y resultados de estudios previos
• No dude en contactar si presenta síntomas de alarma

 *Esta respuesta está basada en: {source}*

ℹ Recuerde: Esta información es de carácter educativo y no reemplaza la consulta médica presencial. Para un diagnóstico preciso y tratamiento personalizado, es fundamental la evaluación clínica directa."""

    def _show_final_response(self, response: SimpleMedicalResponse):
        """Muestra la respuesta final de forma clara"""
        print(f"\n RESPUESTA MÉDICA:")
        print("="*60)
        print(response.answer)
        print("="*60)
        print(f" Tiempo de procesamiento: {response.processing_time:.2f} segundos")


# main.py - Punto de entrada para el sistema RAG médico simple
def main():
  

    print("Estrategia: BM25 + Bi-Encoder + Cross-Encoder")
    
    print("="*60)
    
    # Inicializar sistema
    rag = SimpleMedicalRAG("../config.yaml", mode="finetuning")
    
    if not rag.initialize():
        print("Error en inicialización")
        return
    
    # Consultas de prueba simples
    test_questions = [
        # "¿Cuáles son los síntomas de la diabetes?"
        # "Doctor, tengo dolor de cabeza frecuente",
        "¿Qué puedo hacer para la presión alta?"
    ]
  
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n PRUEBA {i}/{len(test_questions)}")
        response = rag.ask_doctor(question)
        
        if not response.success:
            print(f"Error en consulta: {response.answer}")
        
        # Pausa entre consultas para claridad
        if i < len(test_questions):
            input("\n Presiona Enter para continuar con la siguiente consulta...")
    

if __name__ == "__main__":
    main()