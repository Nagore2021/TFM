#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline refinado de indexación para ChromaDB
- Extracción de texto de PDFs y webs
- Limpieza PRE-chunking
- Chunking robusto
- Extracción de keywords por chunk
- Indexación en ChromaDB
"""

import logging
import os
import sys
import json
from uuid import uuid4
from typing import List, Dict, Any
from urllib.parse import urlparse

# Rutas
top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(top)

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt


# Rutas
top = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(top)

# Utilidades
from retrieval.chroma_utils import (
    create_or_load_collection,
    enhance_chunk_metadata,
    extract_doc_id_from_url,
    clean_text,
    extract_web_text,
    extract_pdf_text,
    cargar_metadatos,
    analizar_chunks_por_longitud,
)
from embeddings.load_model import (
    cargar_configuracion,
    descargar_o_usar_modelo_local,
    cargar_modelo_chromadb,
    cargar_modelo_keybert
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from keybert import KeyBERT
from visualizar_umap import visualizar_umap
import re

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




def cargar_metadatos(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        logger.error(f"Metadatos no encontrados: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Metadatos cargados: {len(data)} desde {path}")
    return data






def analizar_chunks_por_longitud(collection, limit=None):
    """
    Extrae todos los chunks indexados, calcula estadísticas de longitud
    y dibuja un histograma para identificar extremos.
    """
    # 1) Recuperar todos los documentos (chunks)
    docs = collection.get(include=["documents"], limit=limit or collection.count())
    chunks = docs["documents"]
    # 2) Calcular longitudes en caracteres y tokens
    lengths_chars  = [len(c) for c in chunks]
    lengths_tokens = [len(c.split()) for c in chunks]

    # 3) Estadísticas
    stats = {
        "n_chunks": len(chunks),
        "chars_avg": np.mean(lengths_chars),
        "chars_min": np.min(lengths_chars),
        "chars_max": np.max(lengths_chars),
        "chars_p10": np.percentile(lengths_chars, 10),
        "chars_p90": np.percentile(lengths_chars, 90),
        "tokens_avg": np.mean(lengths_tokens),
        "tokens_min": np.min(lengths_tokens),
        "tokens_max": np.max(lengths_tokens)
    }
    print("❏ Estadísticas de chunking:", stats)

    # 4) Histograma
    plt.figure(figsize=(6,4))
    plt.hist(lengths_chars, bins=50, alpha=0.7)
    plt.title("Distribución de longitud en chars por chunk")
    plt.xlabel("Caracteres")
    plt.ylabel("Número de chunks")
    plt.tight_layout()
    plt.show()

    return stats




def indexar_en_chromadb(config_path: str):
    # 1) Configuración
    cfg = cargar_configuracion(config_path)
    paths = cfg["paths"]

    # 2) Inicializar Chroma
    emb_file = descargar_o_usar_modelo_local(cfg["model"]["name_embedding"], paths["model_path"])
    _, embedding_fn = cargar_modelo_chromadb(emb_file)
    collection = create_or_load_collection(paths["chroma_db_path"], cfg["collection"]["name"], embedding_fn)

    existing = collection.get()
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])
        logger.info("Colección limpiada")

    # 3) Preparar splitter
    chunk_size    = cfg["preprocessing"]["chunk_size"]
    chunk_overlap = cfg["preprocessing"]["chunk_overlap"]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "; ", "• ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 4) KeyBERT
    kb_models = {}
    if cfg["search"]["hybrid"]["use_keybert"]:
        ke = descargar_o_usar_modelo_local(cfg["model"]["name_es"], paths["model_path"])
        kb_models["es"] = cargar_modelo_keybert(ke)

    # 5) Indexar PDFs
    for meta in cargar_metadatos(paths["pdf_metadata"]):
        pdf_file = os.path.join(paths["pdf_folder"], meta["filename"])
        if not os.path.exists(pdf_file):
            logger.warning(f"PDF faltante: {pdf_file}")
            continue

        text = extract_pdf_text(pdf_file)

        raw_chunks = splitter.split_text(text)
        

        # 2) Filtrar chunks con menos de 5 tokens
        min_tokens = 5
        chunks = [c for c in raw_chunks if len(c.split()) >= min_tokens]
        total= len(chunks)
        logger.info(f"PDF '{meta['filename']}' → {total} chunks")

        filename = f"pdf_{os.path.splitext(meta['filename'])[0]}"

        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            # keywords por chunk
            kws = []
            if "es" in kb_models:
                kws = [kw for kw,_ in KeyBERT(model=kb_models["es"]).extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1,2),
                    use_mmr=True,
                    top_n=cfg["search"]["hybrid"]["n_keywords"],
                    diversity=cfg["search"]["hybrid"].get("diversity", 0.2)
                )]

            enriched = enhance_chunk_metadata(chunk, {**meta, "chunk_keywords": kws, "document_id": filename},idx, total)
            enriched["document_id"] = f"pdf_{os.path.splitext(meta['filename'])[0]}"
            ids.append(str(uuid4()))
            docs.append(chunk)
            metas.append(enriched)

        collection.add(ids=ids, documents=docs, metadatas=metas)

    # 6) Indexar Webs con extract_web_text
    for meta in cargar_metadatos(paths["web_metadata"]):
        url = meta.get("url")
        logger.info(f"Procesando Web: {url}")
        if not url:
            continue
        text = extract_web_text(url)
        if not text:
            continue

        # chunks = splitter.split_text(text)
        # total = len(chunks)
        # slug = url.strip("/").split("/")[-1]

        raw_chunks = splitter.split_text(text)
        

        # 2) Filtrar chunks con menos de 5 tokens
        min_tokens = 5
        chunks = [c for c in raw_chunks if len(c.split()) >= min_tokens]
        total= len(chunks)
        
        doc_id = extract_doc_id_from_url(url)
        if not doc_id:
            logger.warning(f"No se pudo extraer slug de URL: {url}")
            continue

 
  
        logger.info(f"Procesando Web: {doc_id }")
  

        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            kws = []
            if "es" in kb_models:
                kws = [kw for kw,_ in KeyBERT(model=kb_models["es"]).extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1,2),
                    use_mmr=True,
                    top_n=cfg["search"]["hybrid"]["n_keywords"],
                    diversity=cfg["search"]["hybrid"].get("diversity", 0.2)
                )]

        
            enriched = enhance_chunk_metadata(chunk, {**meta, "chunk_keywords": kws, "document_id": doc_id},idx, total)

           
            enriched["document_id"] = doc_id

            
            ids.append(str(uuid4()))
            docs.append(chunk)
            metas.append(enriched)

        collection.add(ids=ids, documents=docs, metadatas=metas)

    # 7) Resumen
    logger.info(f"Total chunks indexados: {collection.count()}")
    visualizar_umap(collection)
    return collection


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    collection= indexar_en_chromadb(cfg_path)
    # Uso:
    stats = analizar_chunks_por_longitud(collection)

    logger.info(f"Estadísticas post-indexado: {stats}")
