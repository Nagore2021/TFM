#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot Médico con OptimizedMedicalRAG
Interfaz web para consultas médicas usando nuestro sistema RAG optimizado
"""

import streamlit as st
import time
import sys
import os
from typing import Optional

# Añadir path para importar nuestro sistema RAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medical_rag_optimized import OptimizedMedicalRAG, MedicalConsultation

# Configuración básica de la página
st.set_page_config(
    page_title="🏥 Asistente Médico Virtual",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS personalizado para la interfaz médica
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 20px 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .bot-msg {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    .medical-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #e74c3c;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-weight: 500;
    }
    .system-status {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .source-info {
        background: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .loading-text {
        text-align: center;
        color: #3498db;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Cargar sistema RAG
@st.cache_resource
def load_medical_rag() -> Optional[OptimizedMedicalRAG]:
    """
    Carga el sistema RAG médico optimizado
    
    Returns:
        OptimizedMedicalRAG inicializado o None si hay error
    """
    try:
        with st.spinner("🔄 Inicializando sistema médico..."):
            # Configurar path del config (ajustar según tu estructura)
            config_path = "../config.yaml"  # Ajustar ruta
            mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"
            
            # Crear sistema RAG
            rag_system = OptimizedMedicalRAG(
                config_path=config_path,
                mode="embedding",
                mistral_model=mistral_model
            )
            
            # Inicializar
            if rag_system.initialize():
                st.success("✅ Sistema médico cargado correctamente")
                return rag_system
            else:
                st.error("❌ Error inicializando sistema médico")
                return None
                
    except Exception as e:
        st.error(f"❌ Error cargando sistema: {str(e)}")
        st.info("💡 Verifica que config.yaml esté en la ruta correcta y Mistral sea accesible")
        return None

def format_medical_response(consultation: MedicalConsultation) -> str:
    """
    Formatea la respuesta médica para mostrar en la interfaz
    
    Args:
        consultation: Resultado de la consulta médica
        
    Returns:
        Respuesta formateada con información adicional
    """
    if not consultation.success:
        return f"❌ {consultation.answer}"
    
    # Respuesta principal del médico
    formatted_response = f"👨‍⚕️ **RESPUESTA MÉDICA:**\n\n{consultation.answer}\n\n"
    
    # Información de la fuente
    chunk = consultation.best_chunk
    formatted_response += f"""
<div class="source-info">
    <strong>📖 Fuente consultada:</strong> {chunk.get('filename', 'Documento médico')}<br>
    <strong>📍 Sección:</strong> {chunk.get('chunk_position', 'No especificada')}<br>
    <strong>🏷️ Categoría:</strong> {chunk.get('categoria', 'Medicina general')}
</div>
"""
    
    # Estadísticas del pipeline (solo para debugging, comentar en producción)
    stats = consultation.pipeline_stats
    formatted_response += f"""
<details>
<summary>🔍 Detalles técnicos del análisis</summary>

**Pipeline Cross-Encoder Balanced:**
- 🔍 Candidatos BM25: {stats.get('bm25_candidates', 0)}
- 🧠 Candidatos Bi-Encoder: {stats.get('biencoder_candidates', 0)}  
- ⚖️ Pool balanceado: {stats.get('balanced_pool_size', 0)}
- 🎯 Ranking final: {stats.get('final_ranking_size', 0)}

</details>
"""
    
    return formatted_response

def display_medical_disclaimer():
    """Muestra el disclaimer médico importante"""
    st.markdown("""
    <div class="medical-warning">
        <strong>⚠️ AVISO MÉDICO IMPORTANTE:</strong><br><br>
        
        🔸 <strong>Este asistente proporciona información médica general</strong><br>
        🔸 <strong>NO sustituye la consulta con un profesional sanitario</strong><br>
        🔸 <strong>NO realiza diagnósticos ni prescribe tratamientos</strong><br>
        🔸 <strong>En emergencias llama al 112 o acude a urgencias</strong><br><br>
        
        <em>Siempre consulta con tu médico para diagnósticos y tratamientos específicos</em>
    </div>
    """, unsafe_allow_html=True)

def display_system_info(rag_system: OptimizedMedicalRAG):
    """Muestra información del sistema en el sidebar"""
    st.subheader("⚙️ Estado del Sistema")
    
    # Health check
    health = rag_system.system_health_check()
    
    if health["status"] == "healthy":
        st.markdown("""
        <div class="system-status">
            <strong>✅ Sistema Operativo</strong><br>
            🎯 Estrategia: Cross-Encoder Balanced<br>
            🤖 Generación: Mistral-7B<br>
            🔄 Pipeline optimizado activo
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Sistema con problemas")
    
    # Información de componentes
    retrieval_info = health["components"].get("retrieval", {})
    if "chunks_count" in retrieval_info:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 {retrieval_info['chunks_count']:,}</h3>
            <p>Chunks médicos disponibles</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Título principal
    st.markdown('<h1 class="main-title">🏥 Asistente Médico Virtual</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Consultas médicas inteligentes • Información confiable • Español y Euskera</p>', unsafe_allow_html=True)
    
    # Cargar sistema RAG
    rag_system = load_medical_rag()
    
    if not rag_system:
        st.error("❌ No se pudo cargar el sistema médico")
        st.info("💡 Verifica la configuración y vuelve a intentar")
        return
    
    # Disclaimer médico
    display_medical_disclaimer()
    
    # Inicializar historial de conversación
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensaje de bienvenida
        welcome_msg = """¡Hola! 👋 Soy tu asistente médico virtual.

Puedo ayudarte con información sobre:
• 🤒 Síntomas y enfermedades
• 💊 Tratamientos y medicamentos  
• 🛡️ Prevención y cuidados
• 🏥 Recomendaciones de salud

**¿En qué puedo ayudarte hoy?**"""
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Mostrar historial de chat
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
    
    # Input del usuario
    if prompt := st.chat_input("💬 Escribe tu consulta médica aquí..."):
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generar respuesta médica
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analizando tu consulta médica..."):
                try:
                    # CONSULTAR SISTEMA RAG MÉDICO
                    consultation = rag_system.consult_doctor(prompt)
                    
                    # Formatear respuesta
                    formatted_response = format_medical_response(consultation)
                    
                    # Mostrar respuesta
                    st.markdown(formatted_response, unsafe_allow_html=True)
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": formatted_response
                    })
                    
                except Exception as e:
                    error_msg = f"""❌ **Error procesando consulta médica**

Detalles técnicos: {str(e)}

💡 **Mientras tanto, te recomiendo:**
- Consultar con tu médico de cabecera
- Llamar al 112 si es urgente
- Reintentar la consulta más tarde"""
                    
                    st.error("Error en el sistema médico")
                    st.markdown(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Sidebar con información y herramientas
    with st.sidebar:
        st.header("📊 Panel de Control")
        
        # Información del sistema
        display_system_info(rag_system)
        
        st.markdown("---")
        
        # Estadísticas de uso
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        st.metric("🗨️ Consultas realizadas", len(user_messages))
        
        # Botón limpiar chat
        if st.button("🗑️ Limpiar Conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Consultas de ejemplo
        st.subheader("💡 Consultas de Ejemplo")
        
        medical_examples = [
            "¿Cuáles son los síntomas de la diabetes?",
            "¿Cómo prevenir enfermedades cardíacas?",
            "¿Qué hacer ante una herida sangrante?",
            "¿Cuándo debo preocuparme por dolor de cabeza?",
            "Síntomas de depresión y ansiedad",
            "Nola sendatu behar da zauria?"  # Euskera
        ]
        
        for example in medical_examples:
            # Mostrar solo primeras palabras para botones más cortos
            button_text = f"'{example[:25]}...'" if len(example) > 25 else f"'{example}'"
            
            if st.button(button_text, key=f"example_{example}", use_container_width=True):
                # Añadir consulta automáticamente
                st.session_state.messages.append({"role": "user", "content": example})
                
                # Procesar consulta
                try:
                    consultation = rag_system.consult_doctor(example)
                    response = format_medical_response(consultation)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ Error procesando ejemplo: {str(e)}"
                    })
                
                st.rerun()
        
        st.markdown("---")
        
        # Información adicional
        st.subheader("ℹ️ Información")
        
        st.markdown("""
        **🎯 Tecnología:**
        - RAG con Cross-Encoder Balanced
        - Base de conocimiento médica actualizada
        - Generación con Mistral-7B
        
        **📞 Emergencias:**
        - Urgencias: **112**
        - Toxicología: **91 562 04 20**
        - Salud Mental: **024**
        """)
        
      

if __name__ == "__main__":
    main()

# ============ INSTRUCCIONES DE INSTALACIÓN Y USO ============
"""
📋 INSTALACIÓN:

1. Instalar dependencias:
   pip install streamlit torch transformers sentence-transformers

2. Verificar estructura de archivos:
   project/
   ├── config.yaml
   ├── medical_rag_optimized.py  (nuestro sistema RAG)
   ├── medical_chatbot.py        (este archivo)
   └── retrieval/
       └── 2_bm25_model_chunk_bge.py

3. Ajustar rutas en load_medical_rag():
   - config_path: ruta a tu config.yaml
   - Verificar imports de OptimizedMedicalRAG

4. Ejecutar chatbot:
   streamlit run medical_chatbot.py

5. Abrir navegador en:
   http://localhost:8501

🎯 CARACTERÍSTICAS:

✅ Interfaz intuitiva para pacientes
✅ Integra OptimizedMedicalRAG completo
✅ Estrategia Cross-Encoder Balanced
✅ Respuestas médicas con Mistral
✅ Información de fuentes consultadas
✅ Sistema de feedback
✅ Ejemplos de consultas
✅ Disclaimer médico obligatorio
✅ Soporte español y euskera

🔧 PERSONALIZACIÓN:

- Cambiar modelo Mistral en load_medical_rag()
- Ajustar ejemplos de consultas
- Modificar estilos CSS
- Añadir más idiomas
- Integrar base de datos para feedback

💡 PRODUCCIÓN:

- Configurar HTTPS
- Implementar autenticación de usuarios
- Logs de consultas para análisis
- Monitoreo de rendimiento
- Backup de conversaciones
"""