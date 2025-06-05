# Este script sirve para ver cómo están distribuidos los textos (chunks) dentro del espacio semántico, 
# usando ula técnica llamada UMAP que reduce dimensiones (de vectores de 1024 a 2D) para poder visualizarlos en un gráfico.

import umap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import os
from datetime import datetime

def visualizar_umap(collection, n_components=2, color_por="categoria"):
    """
    Proyecta y visualiza embeddings desde una colección ChromaDB con UMAP.

    Args:
        collection: colección de ChromaDB cargada
        n_components: dimensiones UMAP (2D o 3D)
        color_por: metadato usado para colorear puntos (ej: "categoria", "fuente", "idioma")
    """
    logging.info("Aplicando UMAP para visualización...")

    # Obtener embeddings y metadatos
    results = collection.get(include=["embeddings", "metadatas"])
    embeddings = results["embeddings"]
    metadatas = results["metadatas"]

    # Entrenar modelo UMAP
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    umap_model.fit(embeddings)

    # Proyectar embeddings
    def project_embeddings(embs):

        """
        Esta función transforma los embeddings (vectores grandes) a un espacio de 2D o 3D usando UMA
        """
        "Uan matriz vacía para almacenar los embeddings proyectados, de tamaño (n_samples, n_components)"
        "n_samples: len(embs) → cuántos embeddings tenemos."
        umap_embs = np.empty((len(embs), n_components)) 

        # Recorro cada emebedding y si el original era un vector de 1024 dimensiones, esto lo transforma en 2 dimensiones.""
        for i, emb in enumerate(tqdm(embs, desc="Proyectando con UMAP")):
            umap_embs[i] = umap_model.transform([emb])

        "devuelvo la matriz de embeddings proyectados"
        return umap_embs

    projected = project_embeddings(embeddings)

    # Crear DataFrame con metadatos para visualización
    df = pd.DataFrame(projected, columns=["x", "y"][:n_components])
    df[color_por] = [meta.get(color_por, "desconocido") for meta in metadatas]

    # Gráfico
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue=color_por,
        style=color_por,
        palette="Set2",
        s=70
    )
    plt.title(f"Distribución semántica por {color_por} (UMAP)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Guardar figura
    os.makedirs("imagenes", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("imagenes", f"umap_{color_por}_{timestamp}.png")
    plt.savefig(filename)
    logging.info(f"Visualización UMAP guardada en: {filename}")
