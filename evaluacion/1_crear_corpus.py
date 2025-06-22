import os
import json
import logging
import sys
from tqdm import tqdm
from typing import Dict, List
from urllib.parse import urlparse

# Configuración
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuración de rutas
top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(top)

# Importaciones específicas del proyecto
from retrieval.chroma_utils import extract_web_text, extract_pdf_text

class CorpusBuilder:
    def __init__(self, pdf_metadata_path: str, web_metadata_path: str, pdf_folder: str = "pdf_docs"):
        self.pdf_folder = pdf_folder
        self.pdf_metadata = self._load_metadata(pdf_metadata_path)
        self.web_metadata = self._load_metadata(web_metadata_path)
        self.corpus = {}
        self.metadata = {}

    def _load_metadata(self, path: str) -> List[Dict]:
        """Carga metadatos desde archivo JSON"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error cargando {path}: {str(e)}")
            return []

    def build_pdf_corpus(self):
        """Procesa todos los documentos PDF"""
        logging.info("Procesando PDFs...")
        
        for doc_meta in tqdm(self.pdf_metadata, desc="PDFs"):
            try:
                path = os.path.join(self.pdf_folder, doc_meta["filename"])
                if not os.path.exists(path):
                    logging.warning(f"No encontrado: {path}")
                    continue
                
                # Usar directamente la función optimizada de chroma_utils
                clean_text = extract_pdf_text(path)
                
                if not clean_text.strip():
                    logging.warning(f"PDF sin contenido: {doc_meta['filename']}")
                    continue
                
                # Generar ID único para el documento
                doc_id = f"pdf_{doc_meta['filename'].replace('.pdf', '')}"
                
                self.corpus[doc_id] = clean_text
                self.metadata[doc_id] = doc_meta
                
                logging.info(f"PDF procesado: {doc_id}")
                
            except Exception as e:
                logging.error(f"Error procesando PDF {doc_meta['filename']}: {str(e)}")

    def build_web_corpus(self):
        """Procesa todo el contenido web"""
        logging.info("Procesando contenido web...")
        
        for web_meta in tqdm(self.web_metadata, desc="Web"):
            try:
                url = web_meta["url"]
                
                # Usar directamente la función optimizada de chroma_utils
                # (que ya maneja traducción automática si es necesario)
                content = extract_web_text(url)
                
                if not content.strip():
                    logging.warning(f"Contenido web vacío: {url}")
                    continue

                # Generar ID único basado en la URL
                parsed = urlparse(url)
                slug = parsed.path.strip("/").split("/")[0] or "home"
                doc_id = f"web_{slug}"

                # Evitar duplicados
                if doc_id in self.corpus:
                    # Si hay duplicado, usar un sufijo numérico
                    counter = 1
                    original_id = doc_id
                    while doc_id in self.corpus:
                        doc_id = f"{original_id}_{counter}"
                        counter += 1
                    logging.warning(f"⚠️ ID duplicado resuelto: {original_id} → {doc_id}")

                self.corpus[doc_id] = content
                self.metadata[doc_id] = web_meta
                
                logging.info(f"Web procesada: {doc_id}")

            except Exception as e:
                logging.error(f"Error procesando {web_meta['url']}: {str(e)}")

    def save_corpus(self, output_file: str = "corpus_medico.json"):
        """Guarda el corpus final en formato JSON"""
        if not self.corpus:
            logging.error("No hay contenido en el corpus para guardar")
            return
        
        output = {
            "corpus": self.corpus,
            "metadata": self.metadata,
            "stats": {
                "total_documents": len(self.corpus),
                "pdf_documents": len([k for k in self.corpus.keys() if k.startswith("pdf_")]),
                "web_documents": len([k for k in self.corpus.keys() if k.startswith("web_")]),
                "total_characters": sum(len(text) for text in self.corpus.values())
            }
        }
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Corpus guardado en {output_file}")
            logging.info(f"Estadísticas: {output['stats']}")
            
        except Exception as e:
            logging.error(f"Error guardando corpus: {str(e)}")

    def validate_corpus(self):
        """Valida la integridad del corpus"""
        issues = []
        
        # Verificar documentos vacíos
        empty_docs = [doc_id for doc_id, text in self.corpus.items() if not text.strip()]
        if empty_docs:
            issues.append(f"Documentos vacíos: {empty_docs}")
        
        # Verificar metadatos faltantes
        missing_meta = [doc_id for doc_id in self.corpus.keys() if doc_id not in self.metadata]
        if missing_meta:
            issues.append(f"Metadatos faltantes: {missing_meta}")
        
        # Verificar IDs duplicados (no debería pasar, pero por seguridad)
        doc_ids = list(self.corpus.keys())
        if len(doc_ids) != len(set(doc_ids)):
            issues.append("Se detectaron IDs duplicados")
        
        if issues:
            logging.warning(f"Problemas detectados en el corpus: {'; '.join(issues)}")
        else:
            logging.info("Corpus validado correctamente")
        
        return len(issues) == 0


if __name__ == "__main__":
    # Configuración de rutas
    builder = CorpusBuilder(
        pdf_metadata_path="../data/metadata/pdf_metadata.json",
        web_metadata_path="../data/metadata/web_metadata.json", 
        pdf_folder="../data/pdf"
    )
    
    # Construcción del corpus
    builder.build_pdf_corpus()
    builder.build_web_corpus()
    
    # Validación antes de guardar
    if builder.validate_corpus():
        builder.save_corpus("corpus_medico.json")
    else:
        logging.error("Corpus inválido. Revisa los logs para más detalles.")