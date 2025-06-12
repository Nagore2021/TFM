

import os
import logging
import yaml
from sentence_transformers import SentenceTransformer

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import AutoTokenizer, AutoModel
import torch
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain.embeddings.base import Embeddings
from typing import List
from typing import Tuple, Optional

# ------------------------------------------------------------------------------
# 1. Cargar configuración desde un archivo .yaml
# ------------------------------------------------------------------------------
def cargar_configuracion(ruta_config: str) -> dict:
    """
    Carga el archivo config.yaml y lo convierte en un diccionario.
    """
    try:
        with open(ruta_config, "r") as f:
            config = yaml.safe_load(f)
        logging.info("Configuración YAML cargada correctamente.")
        return config
    except Exception as e:
        logging.error(f"Error al cargar configuración: {e}")
        raise


# ------------------------------------------------------------------------
# 2. Descargar o reutilizar modelo local desde Hugging Face
# ------------------------------------------------------------------------


def descargar_o_usar_modelo_local(model_name: str, directorio_modelos: str, modo: str = "sentence-transformer") -> str:
    """
    Descarga o usa un modelo localmente según el tipo.

    Args:
        model_name: nombre Hugging Face o ruta relativa.
        directorio_modelos: ruta donde guardar modelos.
        modo: "sentence-transformer" o "sequence-classification".

    Returns:
        Ruta local al modelo.
    """
    # Evitar subcarpetas al guardar el modelo
    nombre_sanitizado = model_name.replace("/", "_")
    local_path = os.path.join(directorio_modelos, nombre_sanitizado)

    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
        logging.info(f"🔄 Descargando modelo desde Hugging Face: {model_name}")

        try:
            if modo == "sentence-transformer":
                model = SentenceTransformer(model_name)
                model.save(local_path)
                logging.info(f"Modelo ST guardado en: {local_path}")

            elif modo == "sequence-classification":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                tokenizer.save_pretrained(local_path)
                model.save_pretrained(local_path)
                logging.info(f"Modelo HF guardado en: {local_path}")

            else:
                raise ValueError(f"Modo desconocido: {modo}")

        except Exception as e:
            logging.error(f"Error al descargar modelo {model_name}: {e}")
            raise e

    else:
        logging.info(f" Modelo ya disponible en: {local_path}")

    return local_path


def descargar_o_usar_modelo_local1(model_name: str, directorio_modelos: str) -> str:
    local_path = os.path.join(directorio_modelos, model_name.replace("/", "_"))  # evitar carpetas con "/"

    if not os.path.exists(local_path):
        logging.info(f"Descargando modelo desde Hugging Face: {model_name}")

        try:
            # Intenta como SentenceTransformer primero
            model_st = SentenceTransformer(model_name)
            os.makedirs(local_path, exist_ok=True)
            model_st.save(local_path)
            logging.info(f"Modelo ST guardado en: {local_path} (SentenceTransformer)")

        except Exception as e_st:

            logging.warning(f"No es SentenceTransformer directo: {model_name}. Probando como HuggingFace.")

            # Fallback: descarga tokenizer y modelo como HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            os.makedirs(local_path, exist_ok=True)
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            logging.info(f"Modelo HuggingFace guardado en: {local_path}")
    else:
        logging.info(f"Modelo existe en: {local_path}")

    return local_path

# ------------------------------------------------------------------------
# 3. Clase compatible con ChromaDB: embeddings con mean pooling
# ------------------------------------------------------------------------


class FuncionEmbeddingsPersonalizada1(EmbeddingFunction):
    """
    Genera embeddings L2-normalizados usando BGE-M3 (o similar).
    Añade métodos embed_query y embed_documents para compatibilidad con Chroma/LangChain.


    
     Esta clase se encarga de generar los embeddings (vectores numéricos) 
     que se usarán para representar los textos dentro de ChromaDB.

     Qué hace paso a paso:
     ---------------------
     1. Carga un modelo de transformers (como BGE-M3) y su tokenizer.
     2. Cuando se llama con una lista de textos:
        - Los tokeniza (convierte el texto a números que entiende el modelo).
        - Los pasa por el modelo sin entrenar (sin actualizar pesos).
        - 'mean pooling' → calcula el promedio de los vectores de cada palabra del texto, teniendo en cuenta la máscara de atención.
     3. Normaliza los vectores (L2) → importante para que las búsquedas usen 
        distancia coseno (similaridad angular) y no se vean afectadas por la longitud del vector.
     4. Devuelve la lista de vectores ya listos para indexar o hacer búsquedas.

    BGE M3: (position_embeddings): https://bge-model.com/tutorial/1_Embedding/1.2.4.html?utm_source=chatgpt.com 
     Embedding(8194, 1024, padding_idx=1)
     El modelo puede trabajar con hasta 8192 posiciones (porque empieza a contar en 0 y una es padding).
     Cada posición (cada token en una frase) se representa con un vector de 1024 dimensiones.
     Si el texto tiene menos de 8192 tokens, el modelo rellena con tokens [PAD] (relleno) hasta completar la secuencia si es necesario.

    Esta clase es compatible con ChromaDB y LangChain.
    """

    def __init__(self, modelo_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        self.model     = AutoModel.from_pretrained(modelo_path)
        self.model.eval()
        self.max_length = self.model.config.max_position_embeddings
        logging.info(f"Modelo de embeddings cargado desde: {modelo_path} (max_length={self.max_length})")

    def __call__(self, textos: list[str]) -> list[list[float]]:
        if not textos:
            logging.warning("Lista de textos vacía al generar embeddings.")
            return []

        encoded = self.tokenizer(
            textos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        with torch.no_grad():  ## Pasar los textos por el modelo sin calcular gradientes, sin entrenar
            out = self.model(**encoded)

        # CLS pooling para BGE-M3
        if "bge-m3" in self.model.config.name_or_path:
            logging.debug("Usando vector CLS para embeddings (bge-m3).")
            pooled = out.last_hidden_state[:, 0]
        else:
            toks = out.last_hidden_state

            # La attention_mask marca con 1 los tokens reales y con 0 los de padding
            mask = encoded["attention_mask"].unsqueeze(-1).expand(toks.size()).float()
            pooled = (toks * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Normalizar (L2) para similitud coseno, de esta forma solo se mide el ángulo entre vectores, no su tamaño.
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.tolist()

    def embed_query(self, textos: list[str]) -> list[list[float]]:
        """
        Interfaz para generar embeddings de consultas.
        """
        return self(textos)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        """
        Interfaz para generar embeddings de documentos al indexar.
        """
        return self(textos)


from typing import List
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings

class FuncionEmbeddingsPersonalizada(Embeddings):
    """
    Clase que genera embeddings L2-normalizados y expone los métodos
    embed_query y embed_documents para compatibilidad con Chroma y LangChain.
    
    Soporta múltiples modelos:
    - BGE-M3: Modelo multilingüe general
    - bge_m3_bge_m3_epochs/epoch4_MRR0.9717: Modelo BGE-M3 fine-tuneado 


     Genera embeddings L2-normalizados usando BGE-M3 (o similar).
    Añade métodos embed_query y embed_documents para compatibilidad con Chroma/LangChain.


    
     Esta clase se encarga de generar los embeddings (vectores numéricos) 
     que se usarán para representar los textos dentro de ChromaDB.

     Qué hace paso a paso:
     ---------------------
     1. Carga un modelo de transformers (como BGE-M3) y su tokenizer.
     2. Cuando se llama con una lista de textos:
        - Los tokeniza (convierte el texto a números que entiende el modelo).
        - Los pasa por el modelo sin entrenar (sin actualizar pesos).
        - 'mean pooling' → calcula el promedio de los vectores de cada palabra del texto, teniendo en cuenta la máscara de atención.
     3. Normaliza los vectores (L2) → importante para que las búsquedas usen 
        distancia coseno (similaridad angular) y no se vean afectadas por la longitud del vector.
     4. Devuelve la lista de vectores ya listos para indexar o hacer búsquedas.

    BGE M3: (position_embeddings): https://bge-model.com/tutorial/1_Embedding/1.2.4.html?utm_source=chatgpt.com 
     Embedding(8194, 1024, padding_idx=1)
     El modelo puede trabajar con hasta 8192 posiciones (porque empieza a contar en 0 y una es padding).
     Cada posición (cada token en una frase) se representa con un vector de 1024 dimensiones.
     Si el texto tiene menos de 8192 tokens, el modelo rellena con tokens [PAD] (relleno) hasta completar la secuencia si es necesario.

    Esta clase es compatible con ChromaDB y LangChain.
    """

    def __init__(self, modelo_path: str):
        super().__init__()
        # Carga tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        self.model = AutoModel.from_pretrained(modelo_path)
        self.model.eval()
        self.max_length = self.model.config.max_position_embeddings
        self.modelo_path = modelo_path  # Guardar para detección de tipo
        
        # Detectar tipo de modelo
        self.model_type = self._detect_model_type(modelo_path)
        logging.info(f"Modelo de embeddings cargado desde: {modelo_path} (max_length={self.max_length}, type={self.model_type})")

    def _detect_model_type(self, modelo_path: str) -> str:
        """Detecta el tipo de modelo para aplicar pooling correcto."""
        modelo_path_lower = modelo_path.lower()
        
        if "bge-m3" in modelo_path_lower:
            return "bge-m3"
        elif "roberta" in modelo_path_lower or "bio_roberta" in modelo_path_lower:
            return "roberta"
        elif "bert" in modelo_path_lower:
            return "bert"
        else:
            # Para modelos desconocidos, usar mean pooling por defecto
            logging.warning(f"Tipo de modelo no reconocido: {modelo_path}. Usando mean pooling.")
            return "unknown"

    def _encode(self, textos: List[str]) -> torch.Tensor:
        # Validación de entrada
        if not textos:
            logging.warning("Lista de textos vacía al generar embeddings.")
            # Determinar dimensión esperada basada en el modelo
            expected_dim = getattr(self.model.config, 'hidden_size', 1024)
            return torch.empty(0, expected_dim)
            
        # Tokenización y paso por el modelo
        encoded = self.tokenizer(
            textos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            out = self.model(**encoded)

        # Pooling strategy basada en el tipo de modelo
        if self.model_type == "bge-m3":
            # BGE-M3: Usar CLS token
            logging.debug("Usando CLS pooling para BGE-M3")
            pooled = out.last_hidden_state[:, 0]
            
        elif self.model_type == "roberta":
            # RoBERTa (incluido Bio-RoBERTa): Usar CLS token también
            logging.debug("Usando CLS pooling para RoBERTa/Bio-RoBERTa")
            pooled = out.last_hidden_state[:, 0]
            
        else:
            # BERT y otros: Mean pooling con attention mask
            logging.debug(f"Usando mean pooling para {self.model_type}")
            toks = out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(toks.size()).float()
            pooled = (toks * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Normalización L2 para similitud coseno
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        
        # Log para debugging
        logging.debug(f"Embedding shape: {normalized.shape}, model: {self.model_type}")
        return normalized

    def embed_query(self, texto: str) -> List[float]:
        """
        Genera un embedding para una consulta (un único string) y devuelve
        un vector plano (List[float]).
        Requerido por LangChain Embeddings.
        """
        if not isinstance(texto, str):
            raise ValueError(f"embed_query espera un string, recibió: {type(texto)}")
            
        emb_tensor = self._encode([texto])    # Tensor de forma (1, dim)
        emb = emb_tensor[0]                   # Tensor de forma (dim,)
        return emb.tolist()                   # List[float]

    def embed_documents(self, textos: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de documentos y devuelve
        una lista de vectores (List[List[float]]).
        Requerido por LangChain Embeddings.
        """
        if not isinstance(textos, list):
            raise ValueError(f"embed_documents espera una lista, recibió: {type(textos)}")
            
        embs = self._encode(textos)           # Tensor de forma (n, dim)
        return embs.tolist()                  # List[List[float]]

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Para compatibilidad con ChromaDB cuando se usa como embedding_function.
        ChromaDB llama directamente a esta función.
        """
        return self.embed_documents(input)

    def get_model_info(self) -> dict:
        """Información útil para debugging y análisis."""
        return {
            "modelo_path": self.modelo_path,
            "model_type": self.model_type,
            "max_length": self.max_length,
            "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
            "vocab_size": getattr(self.model.config, 'vocab_size', 'unknown')
        }

# ------------------------------------------------------------------------
# 4. Cargar modelo + función de embeddings para ChromaDB
# ------------------------------------------------------------------------
def cargar_modelo_chromadb1(modelo_path: str):

    """
    Carga modelo base + función de embeddings personalizada (para ChromaDB).
    """
    try:
        logging.info("Cargando modelo de embeddings...")
        tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        model = AutoModel.from_pretrained(modelo_path)
        funcion_embeddings = FuncionEmbeddingsPersonalizada(modelo_path)
        logging.info("Modelo de embeddings cargado.")
        return model, funcion_embeddings
    except Exception as e:
        logging.error(f"Error al cargar modelo y tokenizer: {e}")
        raise



def cargar_modelo_chromadb(modelo_path: str) -> Tuple[AutoModel, FuncionEmbeddingsPersonalizada]:
    """
    Carga modelo base + función de embeddings personalizada (para ChromaDB).
    
    Optimizado para evitar cargar el modelo dos veces.
    Compatible con BGE-M3 y Bio-RoBERTa.
    """
    try:
        logging.info(f"Cargando modelo de embeddings desde: {modelo_path}")
        
        # Crear función de embeddings (que ya carga el modelo internamente)
        funcion_embeddings = FuncionEmbeddingsPersonalizada(modelo_path)
        
        # Reutilizar el modelo ya cargado
        model = funcion_embeddings.model
        
        # Log información del modelo
        model_info = funcion_embeddings.get_model_info()
        logging.info(f"Modelo cargado exitosamente: {model_info}")
        
        return model, funcion_embeddings
        
    except Exception as e:
        logging.error(f"Error al cargar modelo y embeddings desde {modelo_path}: {e}")
        raise


def cargar_modelo_chromadb1(modelo_path: str):
    """
    Devuelve una función de embedding compatible con ChromaDB >= 0.4.16
    """
    try:
        logging.info("Cargando modelo de embeddings (Chroma)...")
        funcion_embeddings = SentenceTransformerEmbeddingFunction(model_name=modelo_path)
        logging.info("Modelo cargado correctamente para Chroma.")
        return None, funcion_embeddings
    except Exception as e:
        logging.error(f"Error al cargar modelo de embeddings: {e}")
        raise




def cargar_modelo_keybert(modelo_path: str) -> SentenceTransformer:
    try:
        model = SentenceTransformer(modelo_path)
        logging.info(f"Modelo KeyBERT cargado correctamente desde: {modelo_path}")
        return model
    except Exception as e:
        logging.error(f"Error cargando modelo para KeyBERT: {e}")
        raise



