#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indexador para modelo fine-tuneado (ej: bio_roberta_epochs/epoch5_MRR0.6842)
- Usa embeddings personalizados con normalización L2
- Indexa PDFs y webs con metadatos enriquecidos
- Carga config.yaml y guarda en ChromaDB bajo 'documentos_finetuneado'
"""
import os
import sys
import logging
import json
from uuid import uuid4
import re
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
from keybert import KeyBERT
from visualizar_umap import visualizar_umap

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def indexar_en_chromadb_finetuneado(config_path: str):
    cfg = cargar_configuracion(config_path)
    paths = cfg["paths"]

    # 1. Cargar modelo fine-tuneado y función de embedding personalizada
    model_path = os.path.join(paths["model_path"], cfg["model"]["name_finetuning"])
    _, embedding_fn = cargar_modelo_chromadb(model_path)
    collection = create_or_load_collection(
        paths["chroma_db_path"],
        cfg["collection"]["name_finetuneado"],
        embedding_fn
    )

    existing = collection.get()
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])
        logger.info("Colección limpiada")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "; ", "• ", " ", ""],
        chunk_size=cfg["preprocessing"]["chunk_size"],
        chunk_overlap=cfg["preprocessing"]["chunk_overlap"]
    )

    # 2. KeyBERT para keywords
    kb_models = {}
    if cfg["search"]["hybrid"]["use_keybert"]:
        ke = descargar_o_usar_modelo_local(cfg["model"]["name_es"], paths["model_path"])
        kb_models["es"] = cargar_modelo_keybert(ke)

    # 3. PDFs
    for meta in cargar_metadatos(paths["pdf_metadata"]):
        pdf_path = os.path.join(paths["pdf_folder"], meta["filename"])
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF no encontrado: {pdf_path}")
            continue

        text = extract_pdf_text(pdf_path)
        raw_chunks = splitter.split_text(text)
        chunks = [c for c in raw_chunks if len(c.split()) >= 5]
        total = len(chunks)
        logger.info(f"PDF '{meta['filename']}' → {total} chunks")

        document_id=f"pdf_{os.path.splitext(meta['filename'])[0]}"

        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            kws = []
            if "es" in kb_models:
                kws = [kw for kw, _ in KeyBERT(model=kb_models["es"]).extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1, 2),
                    use_mmr=True,
                    top_n=cfg["search"]["hybrid"]["n_keywords"],
                    diversity=cfg["search"]["hybrid"].get("diversity", 0.2)
                )]

           
            enriched = enhance_chunk_metadata(chunk, {**meta, "chunk_keywords": kws, "document_id": document_id},idx, total)
            enriched["document_id"] = f"pdf_{os.path.splitext(meta['filename'])[0]}"
            ids.append(str(uuid4()))
            docs.append(chunk)
            metas.append(enriched)

        collection.add(ids=ids, documents=docs, metadatas=metas)

    # 4. Webs
    for meta in cargar_metadatos(paths["web_metadata"]):
        url = meta.get("url")
        if not url:
            continue
        text = extract_web_text(url)
        if not text:
            continue
        raw_chunks = splitter.split_text(text)
        chunks = [c for c in raw_chunks if len(c.split()) >= 5]
        total = len(chunks)

        doc_id = extract_doc_id_from_url(url)
        logger.info(f"Web '{url}' → {total} chunks")

        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            kws = []
            if "es" in kb_models:
                kws = [kw for kw, _ in KeyBERT(model=kb_models["es"]).extract_keywords(
                    chunk,
                    keyphrase_ngram_range=(1, 2),
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

    logger.info(f"Total chunks indexados: {collection.count()}")
    visualizar_umap(collection)
    return collection


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    collection = indexar_en_chromadb_finetuneado(cfg_path)
    stats = analizar_chunks_por_longitud(collection)
    logger.info(f"Estadísticas post-indexado: {stats}")
