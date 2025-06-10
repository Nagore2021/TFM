import os
import json
import fitz  # PyMuPDF
import logging
import sys
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import Dict, List
from urllib.parse import urlparse
from functools import lru_cache
from googletrans import Translator

# Configuración
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(top)

translator = Translator()

from retrieval.chroma_utils import extract_web_text

class CorpusBuilder:
    def __init__(self, pdf_metadata_path: str, web_metadata_path: str, pdf_folder: str = "pdf_docs"):
        self.pdf_folder = pdf_folder
        self.pdf_metadata = self._load_metadata(pdf_metadata_path)
        self.web_metadata = self._load_metadata(web_metadata_path)
        self.corpus = {}
        self.metadata = {}

    def _load_metadata(self, path: str) -> List[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f" Error cargando {path}: {str(e)}")
            return []

    def _process_pdf_text(self, text: str) -> str:
        lines = []
        for line in text.split('\n'):
            clean_line = line.strip()
            if clean_line and not clean_line.isdigit():
                lines.append(clean_line)
        return '\n'.join(lines)

    def build_pdf_corpus(self):
        logging.info(" Procesando PDFs...")
        for doc_meta in tqdm(self.pdf_metadata, desc="PDFs"):
            try:
                path = os.path.join(self.pdf_folder, doc_meta["filename"])
                if not os.path.exists(path):
                    logging.warning(f" No encontrado: {path}")
                    continue
                text = ""
                with fitz.open(path) as pdf:
                    for page in pdf:
                        text += page.get_text("text") + "\n"
                clean_text = self._process_pdf_text(text)
                doc_id = f"pdf_{doc_meta['filename'].replace('.pdf', '')}"
                self.corpus[doc_id] = clean_text
                self.metadata[doc_id] = doc_meta
            except Exception as e:
                logging.error(f"Error procesando PDF {doc_meta['filename']}: {str(e)}")

    def _extract_web_content(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=15)
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.find('div', class_='r01-editor') or soup.find('main')
            return content.get_text(separator="\n", strip=True) if content else ""
        except Exception as e:
            logging.error(f"Error al obtener {url}: {str(e)}")
            return ""

    @lru_cache(maxsize=1000)
    def _translate_text(self, text: str) -> str:
        try:
            return translator.translate(text, src='eu', dest='es').text
        except:
            return text

    def build_web_corpus(self):
        logging.info(" Procesando contenido web...")
        for web_meta in tqdm(self.web_metadata, desc="Web"):
            try:
                url = web_meta["url"]
                content = extract_web_text(url)
                if not content:
                    continue

                # Traducir si está en euskera
                if web_meta.get("idioma") == "eu":
                    content = self._translate_text(content)

                parsed = urlparse(url)
                slug = parsed.path.strip("/").split("/")[0]  # ← el segmento principal
                doc_id = f"web_{slug}"

                logging.info(f"ID del Documento : {doc_id}")

                # Evitar sobrescribir documentos
                if doc_id in self.corpus:
                    logging.warning(f"Documento duplicado ignorado: {doc_id}")
                    continue

                self.corpus[doc_id] = content
                self.metadata[doc_id] = web_meta

            except Exception as e:
                logging.error(f"Error procesando {web_meta['url']}: {str(e)}")

    def save_corpus(self, output_file: str = "corpus_medico.json"):
        output = {
            "corpus": self.corpus,
            "metadata": self.metadata
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logging.info(f" Corpus guardado en {output_file}")


if __name__ == "__main__":
    builder = CorpusBuilder(
        pdf_metadata_path="../data/metadata/pdf_metadata.json",
        web_metadata_path="../data/metadata/web_metadata.json",
        pdf_folder="../data/pdf"
    )
    builder.build_pdf_corpus()
    builder.build_web_corpus()
    builder.save_corpus()
