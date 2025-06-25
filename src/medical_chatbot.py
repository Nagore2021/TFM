import streamlit as st
import sys
import logging
import os
import time
from typing import Optional
from transformers import pipeline as hf_pipeline
import langid
import torch

# Imports del proyecto 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medical_rag_produccion import MedicalRAGProduction, MedicalResponse
from retrieval.chroma_utils import translate_eu_to_es
from embeddings.load_model import cargar_configuracion

# Imports para memoria conversacional
from langchain.memory import ConversationSummaryBufferMemory

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración de página y estilo global
st.set_page_config(
    page_title="Osasun Laguntzailea / Asistente Médico",
    page_icon="🏥",
    layout="wide"
)

# CSS para mejorar la UI
st.markdown(
    """
    <style>
      /* Fondo y contenedor principal */
      body, .block-container {
        background-color: #f5f7fa;
        padding: 20px;
      }
      /* Título principal  */
      .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #006BB6, #00c6ff);
        
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
      }
      /* Subtítulo con borde inferior */
      .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-top: 0;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
      }
      /* Aviso médico */
      .medical-warning {
        background: #fffbea;
        border-left: 6px solid #ffc107;
        padding: 18px;
        border-radius: 10px;
        color: #856404;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
      }
      /* Badge de idioma */
      .language-badge {
        background: #e2e3e5;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #495057;
        display: inline-block;
        margin-top: 10px;
        margin-bottom: 10px;
      }
      /* Información del chunk recuperado */
      .chunk-info {
        background: #e8f4f8;
        border-left: 4px solid #0072ff;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #0056b3;
        margin: 10px 0;
      }
      /* Tiempo de procesamiento */
      .processing-time {
        background: #f8f9fa;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #6c757d;
        text-align: right;
        margin-top: 10px;
      }
      /* Burbujas de chat mejoradas */
      .user-message {
        background-color: #d1e7dd;
        padding: 12px 16px;
        border-radius: 15px 15px 0 15px;
        color: #155724;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin: 8px 0;
      }
      .assistant-message {
        background-color: #f0f8ff;
        padding: 12px 16px;
        border-radius: 15px 15px 15px 0;
        color: #2c3e50;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin: 8px 0;
        border-left: 4px solid #0072ff;
      }
      /* Botones primarios */
      .stButton > button {
        background-color: #0072ff;
        color: #ffffff;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
      }
      .stButton > button:hover {
        background-color: #005bb5;
        color: #ffffff;
      }
      /* Input de chat con sombra */
      .stChatInput > div {
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        border-radius: 20px;
        padding: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicializar langid , configurando idiomas
langid.set_languages(['eu', 'es'])

def is_euskera(text: str) -> bool:
    """Detecta si el texto está en euskera """
    lang, conf = langid.classify(text)
    logger.info(f"langid → texto: '{text[:50]}...' → idioma: {lang}, confianza: {conf:.2f}")
    
    # Solo usar langid con umbral más permisivo
    is_euskera_detected = (lang == 'eu')  # Sin umbral de confianza
    
    logger.info(f"Detección final: euskera={is_euskera_detected}")
    return is_euskera_detected

@st.cache_resource  # Decorador para caché de recursos, evita recargar en cada ejecución y carga una vez
def load_euskera_translator():
    """Carga el traductor euskera→español usando tu configuración"""
    try:
        logger.info("Cargando traductor euskera→español...")
        
        # Usar tu función de configuración
        cfg = cargar_configuracion("../config.yaml")
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cargar traductor con tu configuración
        translator = hf_pipeline(
            'translation', 
            model=cfg['model']['translation_model'],
            device=0 if device=="cuda" else -1
        )
        logger.info("Traductor euskera→español cargado exitosamente")
        return translator
    except Exception as e:
        logger.error(f"Error cargando traductor euskera→español: {e}")
        return None

@st.cache_resource
def load_reverse_translator():
    """Carga traductor español → euskera (para respuestas)"""
    try:
        logger.info("Cargando traductor español→euskera...")
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        reverse_translator = hf_pipeline(
            'translation', 
            model="Helsinki-NLP/opus-mt-es-eu", 
            device=0 if device=="cuda" else -1
        )
        logger.info("Traductor español→euskera cargado exitosamente")
        return reverse_translator
    except Exception as e:
        logger.error(f"Error cargando traductor español→euskera: {e}")
        return None

@st.cache_resource
def load_rag_system() -> Optional[MedicalRAGProduction]:
    """Carga el sistema RAG médico de producción con memoria conversacional"""
    try:
        logger.info("Inicializando sistema RAG médico...")
        
        # Crear sistema RAG (silencioso para Streamlit)
        rag = MedicalRAGProduction(
            config_path="../config.yaml",
            mode="embedding",
            verbose=False  # para evitar logs excesivos en Streamlit
        )
        
        # Inicializar sistema
        if rag.initialize():
            logger.info("Sistema RAG inicializado correctamente")
            
            # VERIFICAR que el LLM existe antes de crear memoria
            if hasattr(rag, 'llm') and rag.llm is not None:
                # Crear memoria conversacional
                memory = ConversationSummaryBufferMemory(
                    llm=rag.llm, 
                    max_token_limit=150,
                    memory_key="chat_history",
                    return_messages=True
                )
                rag.memory = memory
                logger.info("Memoria conversacional activada")
            else:
                logger.warning("LLM no disponible, memoria desactivada")
                rag.memory = None
            
            return rag
        else:
            logger.error("Error en inicialización del sistema RAG")
            return None
            
    except Exception as e:
        logger.error(f"Error cargando RAG: {e}")
        return None

def translate_euskera_to_spanish(text: str, translator) -> str:
    """Usa tu función existente de chroma_utils con traductor independiente"""
    if translator is not None:
        translated = translate_eu_to_es(text, translator)
        logger.info(f"Traducción: '{text}' → '{translated}'")
        return translated
    else:
        logger.warning("No hay traductor disponible")
        return text

def translate_spanish_to_euskera(text: str, translator) -> str:
    """
    Traduce texto de español a euskera - Similar a tu función existente
    """
    if translator is None or not text:
        return text
    try:
        # Usar el mismo patrón que tu función translate_eu_to_es
        return translator([text])[0]["translation_text"]
    except Exception as e:
        logger.error(f"Traducción es→eu fallida: {e}")
        return text

def process_medical_query_with_memory(rag_system, spanish_query, original_query, is_euskera, reverse_translator):
    """
    Procesa consulta médica con memoria y traducción de respuesta
    """
    start_time = time.time()
    
    try:
        # 1. Verificar si hay memoria disponible
        if hasattr(rag_system, 'memory') and rag_system.memory is not None:
            memory_vars = rag_system.memory.load_memory_variables({})
            
            if memory_vars.get('chat_history'):
                # Añadir contexto del historial
                contextualized_query = f"Historial médico previo: {memory_vars['chat_history']}\n\nConsulta actual: {spanish_query}"
            else:
                contextualized_query = spanish_query
            
            # Obtener respuesta del RAG (en español)
            response = rag_system.ask_doctor(contextualized_query)
            
            # 2. Si la consulta era en euskera, traducir la respuesta
            if is_euskera and reverse_translator and response.success:
                try:
                    # Guardar respuesta original en español (para memoria)
                    spanish_response = response.answer
                    
                    # Traducir respuesta a euskera
                    euskera_response = translate_spanish_to_euskera(response.answer, reverse_translator)
                    response.answer = euskera_response
                    logger.info("Respuesta traducida al euskera")
                    
                    # Guardar en memoria (en español para coherencia)
                    rag_system.memory.save_context(
                        {"input": spanish_query},
                        {"output": spanish_response}  # Respuesta original en español
                    )
                except Exception as e:
                    logger.warning(f"Error traduciendo respuesta: {e}")
                    response.answer = f"[Respuesta en español - error de traducción]\n\n{response.answer}"
                    
                    # Guardar en memoria de todas formas
                    if response.success:
                        rag_system.memory.save_context(
                            {"input": spanish_query},
                            {"output": response.answer}
                        )
            else:
                # Respuesta en español, guardar directamente si hay memoria
                if response.success and rag_system.memory:
                    rag_system.memory.save_context(
                        {"input": spanish_query},
                        {"output": response.answer}
                    )
        else:
            # Sin memoria, funciona como antes
            logger.info("Procesando sin memoria conversacional")
            response = rag_system.ask_doctor(spanish_query)
            
            # Traducir si es necesario
            if is_euskera and reverse_translator and response.success:
                try:
                    euskera_response = translate_spanish_to_euskera(response.answer, reverse_translator)
                    response.answer = euskera_response
                except Exception as e:
                    logger.warning(f"Error traduciendo: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        # Fallback
        return rag_system.ask_doctor(spanish_query)

def format_medical_response(response: MedicalResponse, is_euskera: bool = False) -> str:
    """Formatea la respuesta médica para Streamlit"""
    
    if not response.success:
        error_msg = "Barkatu, errorea gertatu da" if is_euskera else "Lo siento, ocurrió un error"
        return f"**{error_msg}:**\n\n{response.answer}"
    
    title = "OSASUN ERANTZUNA" if is_euskera else "RESPUESTA MÉDICA"
    
    # Formatear respuesta principal
    formatted_response = f"**{title}:**\n\n{response.answer}"
    
    return formatted_response

def show_chunk_info(response: MedicalResponse):
    """Muestra información del chunk recuperado"""
    if response.success and response.chunk_used:
        chunk = response.chunk_used
        
        info_html = f"""
        <div class="chunk-info">
            <strong>Información recuperada:</strong><br>
            • <strong>Documento:</strong> {chunk.get('document_id', 'No especificado')}<br>
            • <strong>Archivo:</strong> {chunk.get('filename', 'No especificado')}<br>
            • <strong>Categoría:</strong> {chunk.get('categoria', 'No especificada')}<br>
            • <strong>Estrategia:</strong> {chunk.get('strategy_used', 'No especificada')}
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

        
        chunk_text = chunk.get('text', 'Texto no disponible')
        
        # Crear un expander para el texto completo
        with st.expander(" Ver contenido utilizado como contexto", expanded=False):
            st.text_area(
                "Texto enviado al LLM:",
                value=chunk_text,
                height=200,
                disabled=True,
                help="Este es el fragmento de texto que se utilizó como contexto para generar la respuesta"
            )
            
            # Información adicional del texto
            st.caption(f" Longitud: {len(chunk_text)} caracteres | {len(chunk_text.split())} palabras")

def show_processing_time(processing_time: float):
    """Muestra el tiempo de procesamiento"""
    time_html = f"""
    <div class="processing-time">
        Procesado en {processing_time:.2f} segundos
    </div>
    """
    st.markdown(time_html, unsafe_allow_html=True)

def add_simple_controls(rag_system):
    """
    Controles mínimos sin sidebar
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:  # Centrado
        if st.button("Limpiar Chat", type="secondary"):
            st.session_state.messages = []
            # Limpiar memoria si existe
            if rag_system and hasattr(rag_system, 'memory') and rag_system.memory:
                rag_system.memory.clear()
                st.success("Chat y memoria limpiados")
            else:
                st.success("Chat limpiado")
            st.rerun()

def main():
    # Cabecera principal
    st.markdown('<h1 class="main-title">Osasun Eskola</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistema RAG Médico / Osasun RAG Sistema</p>', unsafe_allow_html=True)
    
    # Cargar componentes del sistema
    with st.spinner("Cargando sistema médico..."):
        euskera_translator = load_euskera_translator()  # euskera→español
        reverse_translator = load_reverse_translator()  # español→euskera
        rag_system = load_rag_system()
    
    # Verificar si el sistema está disponible
    if not rag_system:
        st.error("**Sistema médico no disponible**")
        st.stop()

    
 
    if torch.cuda.is_available():
        st.success(f"GPU activa: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("GPU no detectada. Ejecutando en CPU")
    
    # Aviso médico
    st.markdown(
        '<div class="medical-warning">'
        '<strong>AVISO MÉDICO / OHAR MEDIKOA:</strong><br>'
        'Sistema de información médica  para orientación general. '
        '<strong>NO SUSTITUYE a la consulta médica profesional.</strong><br>'
        '<strong>Osakidetzako informazio medikoko sistema orientazio orokorrerako. '
        'Ez du ordeztzen kontsulta mediko profesionala.</strong><br>'
        '<strong>Emergencias / Larrialdiak: 112</strong>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Controles simples
    add_simple_controls(rag_system)
    
    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome = (
            "**Kaixo! Osasun laguntzailea naiz**\n\n"
            "Gaixotasunen inguruko galderak erantzuteko hemen nago: "
            "sintomak, tratamenduak, zainketa egokiak...\n\n"
            "**¡Hola! Soy tu asistente médico**\n\n"
            "Estoy aquí para responder preguntas sobre salud: "
            "síntomas, tratamientos, cuidados apropiados...\n\n"
            "**¿En qué puedo ayudarte hoy? / Zertan lagun zaitzaket gaur?**"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Input del usuario
    if prompt := st.chat_input("Idatzi zure galdera hemen / Escribe tu pregunta aquí..."):
        
        # Detectar idioma
        is_eu = is_euskera(prompt)
        language_badge = "Euskera" if is_eu else "Español"
        
        # Mostrar pregunta del usuario
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="language-badge">{language_badge}</div>', unsafe_allow_html=True)
        
        # Añadir al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Procesar consulta
        with st.chat_message("assistant"):
            with st.spinner("Analizando consulta médica..."):
                
                # Traducir consulta si es necesario (para procesamiento)
                if is_eu:
                    spanish_query = translate_euskera_to_spanish(prompt, euskera_translator)
                    st.info(f"**Procesando:** {spanish_query}")
                else:
                    spanish_query = prompt
                
                try:
                    # PROCESAR CON MEMORIA Y TRADUCCIÓN
                    response = process_medical_query_with_memory(
                        rag_system, 
                        spanish_query,    # Para procesamiento RAG
                        prompt,          # Consulta original
                        is_eu,           # Si necesita traducir respuesta
                        reverse_translator # Traductor español→euskera
                    )
                    
                    # Formatear y mostrar respuesta (ya traducida si era necesario)
                    formatted_response = format_medical_response(response, is_eu)
                    st.markdown(f'<div class="assistant-message">{formatted_response}</div>', unsafe_allow_html=True)
                    
                    # Mostrar información adicional si fue exitoso
                    if response.success:
                        show_chunk_info(response)
                    
                    show_processing_time(response.processing_time)
                    
                    # Añadir al historial de Streamlit
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    
                except Exception as e:
                    error_msg = f"**Error del sistema:** {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()