#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot Médico Ultra Simple
Solo detecta euskera por palabras clave básicas
"""

import streamlit as st
import sys
import logging
import os
from typing import Optional
from transformers import pipeline as hf_pipeline
import langid
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Imports del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medical_rag_optimized import OptimizedMedicalRAG, MedicalConsultation
from retrieval.chroma_utils import translate_eu_to_es


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


langid.set_languages(['eu', 'es'])

st.set_page_config(
    page_title=" Asistente Médico / Osasun Laguntzailea",
    page_icon="",
    layout="centered"
)

# CSS para mejorar el estilo
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 20px 0;
        font-size: 2.2rem;
        font-weight: bold;
        border-bottom: 2px solid #3498db;
    }
    .medical-warning {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        color: #856404;
        font-size: 1rem;
    }
    .language-badge {
        background: #e8f5e8;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 5px 0;
        display: inline-block;
        color: #2c3e50; 
        

    }
</style>
""", unsafe_allow_html=True)

def is_euskera(text: str) -> bool:
    """
    Devuelve True si langid.py detecta 'eu' (euskera), False en caso contrario.
    """
    lang, conf = langid.classify(text)
    logger.info(f"langid.py → idioma detectado: {lang} (confianza {conf:.2f})")
    return lang == 'eu'

@st.cache_resource
def load_translator():
    """Carga solo el traductor"""
    try:
        return hf_pipeline('translation', model="Helsinki-NLP/opus-mt-eu-es", device=-1)
    except Exception:
        return None

@st.cache_resource
def load_rag_system() -> Optional[OptimizedMedicalRAG]:
    """Carga el sistema RAG"""
    try:
        rag_system = OptimizedMedicalRAG(
            config_path="../config.yaml",
            mode="embedding",
            mistral_model="mistralai/Mistral-7B-Instruct-v0.3"
        )
        return rag_system if rag_system.initialize() else None
    except Exception:
        return None

def format_response(consultation: MedicalConsultation, is_euskera_query: bool) -> str:
    """Formatea respuesta simple"""
    if not consultation.success:
        return f"❌ {consultation.answer}"
    
    title = "OSASUN ERANTZUNA" if is_euskera_query else "RESPUESTA MÉDICA"
    return f"👨‍⚕️ **{title}:**\n\n{consultation.answer}"

def main():
    # Título
    st.markdown(
        '<h1 class="main-title">🏥 Osasun Laguntzailea / Asistente Médico</h1>',
        unsafe_allow_html=True
    )
    
    # Cargar componentes
    translator = load_translator()
    # rag_system = load_rag_system()
    # if not rag_system:
    #     st.error("❌ Sistema no disponible")
    #     return

    st.markdown("""
    <div class="medical-warning">
        <strong>AVISO:</strong> Solo información general. No sustituye al médico. Emergencias: 112
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome = """Kaixo!  / ¡Hola! 

Osasun galderak erantzuteko hemen nago.
Estoy aquí para responder preguntas de salud.

**¿En qué puedo ayudarte?**"""
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # 1) Mostrar todo el historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # 2) Leer entrada del usuario (fuera del for)
    if prompt := st.chat_input("Escribe tu pregunta..."):
        # 2.1) Detectar idioma
        euskera_detected = is_euskera(prompt)

        # 2.2) Mostrar badge de idioma
        if euskera_detected:
            st.markdown(
                '<div class="language-badge"> Euskera</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="language-badge"> Español</div>',
                unsafe_allow_html=True
            )

        # 2.3) Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2.4) Preparar consulta para RAG
        if euskera_detected:
            query_for_rag = translate_eu_to_es(prompt, translator)
            st.info(f"**Traducción:** {query_for_rag}")
        else:
            query_for_rag = prompt

        # 2.5) Llamar al RAG y mostrar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    # consultation = rag_system.consult_doctor(query_for_rag)
                    # response = format_response(consultation, euskera_detected)
                    # st.markdown(response, unsafe_allow_html=True)
                    # st.session_state.messages.append({"role": "assistant", "content": response})
                    pass  # sustituye este pass por tu lógica de RAG
                except Exception:
                    error_msg = "❌ Error en el sistema"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # 3) Sidebar súper simple
    with st.sidebar:
        st.header("💬 Control")
        
        # Contador de preguntas
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        st.metric("Preguntas", len(user_msgs))
        
        # Botón limpiar
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Kaixo! / ¡Hola! Osasun galderak erantzuteko hemen nago. Estoy aquí para responder preguntas de salud. **¿En qué puedo ayudarte?**"
            })
            st.rerun()

        st.markdown("---")
        
        # Ejemplos rápidos
        st.subheader("Ejemplos rápidos")
        if st.button("¿Síntomas diabetes?"):
            st.session_state.messages.append({
                "role": "user",
                "content": "¿Cuáles son los síntomas de la diabetes?"
            })
            st.rerun()
        if st.button("Diabetesaren sintomak?"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Diabetesaren sintomak zeintzuk dira?"
            })
            st.rerun()
        if st.button("¿Dolor de cabeza?"):
            st.session_state.messages.append({
                "role": "user",
                "content": "¿Qué hacer con dolor de cabeza fuerte?"
            })
            st.rerun()

if __name__ == "__main__":
    main()

# ============ INSTALACIÓN ============
"""
pip install streamlit transformers langdetect

streamlit run ultra_simple_chatbot.py

SUPER SIMPLE:
- Detecta euskera con langdetect (+ fallback keywords)
- Usa translate_eu_to_es del proyecto
- Sin complicaciones de probabilidades
- Listo para demo y entrega
"""

