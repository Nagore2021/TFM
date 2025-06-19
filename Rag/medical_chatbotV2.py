import streamlit as st
import sys
import logging
import os
from typing import Optional
from transformers import pipeline as hf_pipeline
import langid

# Imports del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medical_rag_optimized import OptimizedMedicalRAG, MedicalConsultation
from retrieval.chroma_utils import translate_eu_to_es

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración de página y estilo global
st.set_page_config(
    page_title="🏥 Osasun Laguntzailea / Asistente Médico",
    page_icon="🏥",
    layout="wide"
)

# Inyectar CSS para una interfaz más colorida y clara
st.markdown(
    """
    <style>
      /* Fondo y contenedor principal */
      body, .block-container {
        background-color: #f5f7fa;
        padding: 20px;
      }
      /* Título principal con degradado */
      .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
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
      /* Aviso médico mejorado */
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
      /* Burbujas de chat con sombra y espaciado */
      .stChatMessage {
        margin: 10px 0;
      }
      .stChatMessage.stChatMessage--user .stMarkdown p {
        background-color: #d1e7dd;
        padding: 12px 16px;
        border-radius: 15px 15px 0 15px;
        color: #155724;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
      }
      .stChatMessage.stChatMessage--assistant .stMarkdown p {
        background-color: #f8d7da;
        padding: 12px 16px;
        border-radius: 15px 15px 15px 0;
        color: #721c24;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
      }
      /* Sidebar estilo */
      .stSidebar {
        background-color: #ffffff;
        border-left: 4px solid #0072ff;
        padding: 20px;
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


# Inicializar langid
langid.set_languages(['eu', 'es', 'en'])

def is_euskera(text: str) -> bool:
    lang, conf = langid.classify(text)
    logger.info(f"langid → idioma: {lang}, conf: {conf:.2f}")
    return lang == 'eu'

@st.cache_resource
def load_translator():
    try:
        return hf_pipeline('translation', model="Helsinki-NLP/opus-mt-eu-es", device=-1)
    except:
        return None

@st.cache_resource
def load_rag_system() -> Optional[OptimizedMedicalRAG]:
    try:
        rag = OptimizedMedicalRAG(
            config_path="../config.yaml",
            mode="embedding",
            mistral_model="mistralai/Mistral-7B-Instruct-v0.3"
        )
        if rag.initialize():
            return rag
    except Exception as e:
        logger.error(f"Error cargando RAG: {e}")
    return None


def format_response(consult: MedicalConsultation, is_eu: bool) -> str:
    title = "OSASUN ERANTZUNA" if is_eu else "RESPUESTA MÉDICA"
    if not consult.success:
        return f"❌ {consult.answer}"
    return f"👨‍⚕️ **{title}:**\n\n{consult.answer}"


def main():
    # Cabecera
    st.markdown('<h1 class="main-title">Osasun eskola </h1>', unsafe_allow_html=True)
    
    # Cargar componentes
    translator = load_translator()
    # rag_system = load_rag_system()
    # if not rag_system:
    #     st.error("❌ Sistema no disponible")
    #     return

    # Aviso médico
    st.markdown(
        '<div class="medical-warning">'
        '<strong> AVISO / OHAR:</strong> '
        'Somos un servicio de Osakidetza y Departamento de Salud que ofrecemos información y formación a la ciudadanía, '
        'con el fin de lograr una actitud responsable y activa en torno a tu salud y enfermedad.<br>'
        '<strong>Osakidetzako eta Osasun Sailaren zerbitzu bat gara; informazioa eta prestakuntza eskaintzen ditugu '
        'zure osasunaren eta gaixotasunaren aurrean jokabide arduratsua eta aktiboa izan dezazula lortzeko.</strong><br>'
        '<strong>Información de salud general. No sustituye al médico.</strong> Emergencias: 112'
        '</div>',

      
        unsafe_allow_html=True
    )

    # Inicializar historial
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome = (
            "Kaixo! / ¡Hola! \n\n"
            "Osasun galderak erantzuteko hemen nago.\n"
            "Zein dira gaixotasun bakoitzerako zainketa egokiak? Eta sintomak, probak eta ohiko tratamenduak?.\n"
            "¿Cuáles son los cuidados apropiados para cada enfermedad? ¿Y los síntomas, pruebas y tratamientos habituales? \n"
            "**Zertan lagun zaitzaket gaur?** \n"
            "**¿En qué puedo ayudarte hoy?**"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Entrada del usuario
    if prompt := st.chat_input(" Idatzi zure galdera / Escribe tu pregunta..."):
        # Detectar idioma
        eu = is_euskera(prompt)
        badge = "Euskera" if eu else "Español"
        st.markdown(f'<div class="language-badge">{badge}</div>', unsafe_allow_html=True)

        # Mostrar pregunta
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Preparar consulta
        if eu and translator:
            query = translate_eu_to_es(prompt, translator)
            st.info(f"**Traducción:** {query}")
        else:
            query = prompt

        # # Llamar RAG
        # with st.chat_message("assistant"):
        #     with st.spinner("🔍 Procesando..."):
        #         try:
        #             consult = rag_system.consult_doctor(query)
        #             resp = format_response(consult, eu)
        #             st.markdown(resp, unsafe_allow_html=True)
        #             st.session_state.messages.append({"role": "assistant", "content": resp})
        #         except Exception:
        #             err = "❌ Error en el sistema"
        #             st.error(err)
        #             st.session_state.messages.append({"role": "assistant", "content": err})

    # Sidebar
    with st.sidebar:
        st.header("💬 Control")
        count = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.metric("Preguntas realizadas", count)
        if st.button("🗑️ Limpiar chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
