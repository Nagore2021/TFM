import os
import sys
import fitz  # PyMuPDF, para leer PDFs
import requests
import logging
import json
from bs4 import BeautifulSoup
from keybert import KeyBERT

from sentence_transformers import SentenceTransformer

# Añadir ruta para módulos locales
sys.path.append(os.path.abspath('..'))
from preprocessing.preprocessing import preprocess_text              

# Configurar logging global
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extraer_entidades(texto: str, ner_pipeline, min_len=3) -> list:
    """
    Usa el pipeline de NER para extraer entidades relevantes y limpias.
    Devuelve una lista de entidades: [{"text": ..., "type": ...}, ...]
    """
    try:
        raw_ents = ner_pipeline(texto)
        entidades = []

        for ent in raw_ents:
            palabra = ent.get("word", "").strip()
            tipo = ent.get("entity_group", "")
            if len(palabra) >= min_len and palabra.isalpha():
                entidades.append({
                    "text": palabra.lower(),
                    "type": tipo
                })

        return entidades
    except Exception as e:
        logging.warning(f" Error extrayendo entidades: {e}")
        return []


# -----------------------------------------------------------------------------
# 1. Cargar PDFs y extraer texto limpio + palabras clave + entidades 
# -----------------------------------------------------------------------------
def load_pdfs_from_folder(folder_path, metadata_list,
                          use_lemmatization=False,
                          remove_stopwords=False,
                          use_keybert=True,
                          keybert_models=None,
                          n_keywords=5,
                          diversity=0.2,
                          ner_pipeline=None):
    """
    Carga y procesa documentos PDF. Limpia el texto y extrae keywords si se desea.
    
    Args:
        folder_path: ruta a la carpeta con PDFs
        metadata_list: lista de metadatos (con idioma, nombre, etc.)
        use_lemmatization: aplicar lematización (opcional)
        remove_stopwords: quitar stopwords (opcional)
        use_keybert: activar extracción de keywords
        keybert_models: diccionario de modelos por idioma
        n_keywords: número de palabras clave a extraer
        diversity: parámetro MMR para KeyBERT

    Returns:
        Lista de textos limpios y metadatos enriquecidos
    """
    texts = []
    metadatas = []

    #procesar cada documento PDF
    for doc in metadata_list:
        path = os.path.join(folder_path, doc["filename"])

        # Verificar si el archivo existe
        if not os.path.exists(path):
            logging.warning(f" No se encontró el archivo: {path}")
            continue

        try:

            
            text = ""
            with fitz.open(path) as pdf:  # Abre el PDF usando PyMuPDF
                for page in pdf:
                    text += page.get_text("text") + "\n"

            # Eliminar espacios en blanco y saltos de línea innecesarios . limpiar texto
            if text.strip():
                lang = doc.get("idioma", "es")  # idioma del documento
                logging.info(f"Procesando PDF: {doc['filename']} (idioma: {lang})")

                cleaned_text = preprocess_text(
                    text,
                    use_lemmatization=use_lemmatization,
                    remove_stopwords=remove_stopwords,
                    idioma=lang
                )

                texts.append(cleaned_text)

                # Extraer keywords si está activado
                if use_keybert and keybert_models:
                    # Obtener el modelo de KeyBERT según el idioma
                    modelo = keybert_models.get(lang)
                    if modelo:
                       # logging.info(f" KeyBERT → idioma: {lang} | clase: {modelo.__class__.__name__}")
                       # assert isinstance(modelo, SentenceTransformer), f"Modelo no válido para KeyBERT: {type(modelo)}"
                        kw_model = KeyBERT(model=modelo)
                        keywords = kw_model.extract_keywords(
                            cleaned_text,
                            keyphrase_ngram_range=(1, 2),
                            stop_words=None,
                            top_n=n_keywords,
                            use_mmr=True,
                            diversity=diversity
                        )
                        keywords = [kw for kw, _ in keywords]

                        existentes = set(map(str.lower, doc.get("palabras_clave", [])))
                        nuevas = set(map(str.lower, keywords))
                        doc["palabras_clave"] = list(existentes.union(nuevas))


                 # === EXTRAER ENTIDADES BIOMÉDICAS ===
              #  if ner_pipeline:
                   
              #      doc["entidades"] = extraer_entidades(cleaned_text, ner_pipeline)
             #   else:
             #       doc["entidades"] = []  
                metadatas.append(doc)

        except Exception as e:
            logging.error(f" Error al procesar {path}: {e}")

    return texts, metadatas


# -----------------------------------------------------------------------------
# 2. Extraer texto principal de una URL (limpiar + keywords + entidades)
# -----------------------------------------------------------------------------
def extract_main_content(url,
                         use_lemmatization=False,
                         remove_stopwords=False,
                         use_keybert=False,
                         keybert_models=None,
                         idioma="es",
                         n_keywords=5,
                         diversity=0.2, 
                         ner_pipeline=None):
    """
    Extrae el contenido útil de una página web (HTML), limpia y (opcionalmente) extrae keywords.
    """
    try:
        logging.info(f"Procesando Web: {url}")    
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Buscar el contenedor principal si existe
        editor_div = soup.find("div", class_="r01-editor")
        if editor_div:
            text = editor_div.get_text(separator="\n", strip=True)
        else:
            # Eliminar partes no informativas (scripts, headers, etc.)
            for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

    
       
        # Limpieza y normalización del texto
        cleaned_text = preprocess_text(
            text,
            use_lemmatization=use_lemmatization,
            remove_stopwords=remove_stopwords,
            idioma=idioma
        )

        # === EXTRAER ENTIDADES BIOMÉDICAS (siempre)
        entidades = []
        # if ner_pipeline:
        #     entidades =  extraer_entidades(cleaned_text, ner_pipeline)
        # else:
        #     entidades = []

        # logging.info(f"Entidades ({entidades})")

        # Extraer palabras clave (si se activa)
        keywords = []
        if use_keybert and keybert_models:
            modelo = keybert_models.get(idioma)
            if modelo:
                #logging.info(f" KeyBERT → idioma: {idioma} | clase: {modelo.__class__.__name__}")
                #assert isinstance(modelo, SentenceTransformer), f"Modelo no válido para KeyBERT: {type(modelo)}"
                kw_model = KeyBERT(model=modelo)
                keywords = kw_model.extract_keywords(
                    cleaned_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words=None,
                    top_n=n_keywords,
                    use_mmr=True,
                    diversity=diversity
                )

                keywords = [kw for kw, _ in keywords]


                return cleaned_text, keywords, entidades

        return cleaned_text, []

    except Exception as e:
        logging.warning(f"Error al procesar {url}: {e}")
        return "", []


# -----------------------------------------------------------------------------
# 3. Cargar múltiples páginas web desde metadatos
# -----------------------------------------------------------------------------
def load_web_pages(metadata_list,
                   use_lemmatization=False,
                   remove_stopwords=False,
                   use_keybert=False,
                   keybert_models=None,
                   n_keywords=5,
                   diversity=0.2,
                   ner_pipeline=None):
    """
    Carga contenido desde varias URLs, limpia y enriquece con palabras clave.
    
    Args:
        metadata_list: lista de diccionarios con metadatos por URL

    Returns:
        Lista de textos limpios y metadatos enriquecidos
    """
    texts = []
    metadatas = []

    for meta in metadata_list:
        url = meta.get("url")
        if not url:
            continue

        lang = meta.get("idioma", "es")  # idioma del documento
       

        text, keywords,entidades = extract_main_content(
            url,
            use_lemmatization=use_lemmatization,
            remove_stopwords=remove_stopwords,
            use_keybert=use_keybert,
            keybert_models=keybert_models,
            idioma=lang,
            n_keywords=n_keywords,
            diversity=diversity,
            ner_pipeline=ner_pipeline
        )

        if text:
            texts.append(text)

            if keywords:
                existentes = set(map(str.lower, meta.get("palabras_clave", [])))
                nuevas = set(map(str.lower, keywords))
                meta["palabras_clave"] = list(existentes.union(nuevas))

         #   meta["entidades"] = entidades
            metadatas.append(meta)

    return texts, metadatas
