from sentence_transformers import SentenceTransformer, models, util
import os
import numpy as np
import matplotlib.pyplot as plt

# -------- Función para calcular similitud --------
def calcular_similitud(modelo, frase1, frase2):
    """
    Calcula la similitud coseno entre dos frases usando un modelo dado
    
    Args:
        modelo: SentenceTransformer model
        frase1: Primera frase (str)
        frase2: Segunda frase (str)
    
    Returns:
        float: Similitud coseno entre 0 y 1
    """
    emb1 = modelo.encode(frase1, convert_to_tensor=True)
    emb2 = modelo.encode(frase2, convert_to_tensor=True)
    similitud = util.cos_sim(emb1, emb2).item()
    return similitud

# -------- Configurar modelos --------
print("🔄 Cargando modelos...")

try:
    modelo_base = SentenceTransformer("PlanTL-GOB-ES/bsc-bio-ehr-es")
    print("✅ Modelo base cargado")
except:
    print("⚠️ Creando modelo base con pooling manual...")
    word_embedding_model = models.Transformer("PlanTL-GOB-ES/bsc-bio-ehr-es", max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    modelo_base = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("✅ Modelo base creado")

# Cargar modelo fine-tuneado
ruta_modelo_finetuneado = "../models/bio_roberta_epochs/epoch4_MRR0.9726"


if not os.path.isdir(ruta_modelo_finetuneado):
    raise ValueError(f" Ruta '{ruta_modelo_finetuneado}' no encontrada")

modelo_finetuneado = SentenceTransformer(ruta_modelo_finetuneado)
print("✅ Modelo fine-tuneado cargado")

# -------- Pares semánticos para evaluar --------
pares_medicos = [
    ("diabetes", "azúcar alto"),
    ("cáncer colorrectal", "tumor maligno"),  
    ("cáncer", "tumor"),
    ("cáncer de colon", "colonoscopia"),
    ("recidiva del cáncer de mama", "reaparición del cáncer"),
     ("cáncer de mama", "Mamografía"),
    ("depresión", "tristeza"),
    ("HbA1c", "promedio de glucosa en sangre"),
    ("cáncer pulmon", "tos persistente"),
    ("hipoglucemia", "bajada de azúcar"),
    ("ictus", "interrupción repentina del flujo sanguíneo en el cerebro")
]

# Pares no relacionados (control negativo)
pares_control = [
    ("diabetes", "fractura"),
    ("hipertensión", "resfriado"),
    ("infarto", "alergia"),
    ("ictus", "gripe"),
    ("edema", "fiebre"),
     ("tumor de pulmón", "ataque al corazón"),
]

print(f"\n🧪 Evaluando {len(pares_medicos)} pares médicos y {len(pares_control)} pares control...")

# -------- Evaluación de pares médicos --------
resultados_medicos = []
print("\n PARES MÉDICOS (debería mejorar con fine-tuning):")
print("-" * 70)

for termino_tecnico, termino_coloquial in pares_medicos:
    sim_base = calcular_similitud(modelo_base, termino_tecnico, termino_coloquial)
    sim_ft = calcular_similitud(modelo_finetuneado, termino_tecnico, termino_coloquial)
    mejora = sim_ft - sim_base
    mejora_pct = (mejora / sim_base * 100) if sim_base > 0 else 0
    
    resultados_medicos.append({
        'par': f"{termino_tecnico} ↔ {termino_coloquial}",
        'base': sim_base,
        'finetuned': sim_ft,
        'mejora': mejora,
        'mejora_pct': mejora_pct
    })
    
    # Indicador visual de mejora
    if mejora > 0.1:
        indicador = " EXCELENTE"
    elif mejora > 0.05:
        indicador = " BUENA"
    elif mejora > 0.02:
        indicador = " LEVE"
    elif mejora > 0:
        indicador = " MÍNIMA"
    else:
        indicador = " SIN MEJORA"
    
    print(f"{termino_tecnico:12} ↔ {termino_coloquial:15} | "
          f"Base: {sim_base:.3f} | FT: {sim_ft:.3f} | "
          f"Δ: {mejora:+.3f} ({mejora_pct:+.1f}%) {indicador}")

# -------- Evaluación de pares control --------
resultados_control = []
print(f"\n PARES CONTROL (NO debería mejorar mucho):")
print("-" * 70)

for termino1, termino2 in pares_control:
    sim_base = calcular_similitud(modelo_base, termino1, termino2)
    sim_ft = calcular_similitud(modelo_finetuneado, termino1, termino2)
    mejora = sim_ft - sim_base
    mejora_pct = (mejora / sim_base * 100) if sim_base > 0 else 0
    
    resultados_control.append({
        'par': f"{termino1} ↔ {termino2}",
        'base': sim_base,
        'finetuned': sim_ft,
        'mejora': mejora,
        'mejora_pct': mejora_pct
    })
    
    print(f"{termino1:12} ↔ {termino2:15} | "
          f"Base: {sim_base:.3f} | FT: {sim_ft:.3f} | "
          f"Δ: {mejora:+.3f} ({mejora_pct:+.1f}%)")

# -------- Análisis estadístico --------
mejoras_medicos = [r['mejora'] for r in resultados_medicos]
mejoras_control = [r['mejora'] for r in resultados_control]

print(f"\n ANÁLISIS ESTADÍSTICO:")
print("=" * 50)
print(f"PARES MÉDICOS:")
print(f"  Mejora promedio: {np.mean(mejoras_medicos):+.4f}")
print(f"  Desviación estándar: {np.std(mejoras_medicos):.4f}")
print(f"  Pares con mejora: {sum(1 for m in mejoras_medicos if m > 0)}/{len(mejoras_medicos)} ({sum(1 for m in mejoras_medicos if m > 0)/len(mejoras_medicos)*100:.1f}%)")
print(f"  Mejora máxima: {max(mejoras_medicos):+.4f}")
print(f"  Mejora mínima: {min(mejoras_medicos):+.4f}")

print(f"\nPARES CONTROL:")
print(f"  Mejora promedio: {np.mean(mejoras_control):+.4f}")
print(f"  Desviación estándar: {np.std(mejoras_control):.4f}")
print(f"  Pares con mejora: {sum(1 for m in mejoras_control if m > 0)}/{len(mejoras_control)} ({sum(1 for m in mejoras_control if m > 0)/len(mejoras_control)*100:.1f}%)")

# -------- Interpretación --------
mejora_prom_medicos = np.mean(mejoras_medicos)
mejora_prom_control = np.mean(mejoras_control)

print(f"\n INTERPRETACIÓN:")
print("=" * 50)

if mejora_prom_medicos > 0.10:
    print(" EXCELENTE: Fine-tuning muy efectivo para pares médicos")
elif mejora_prom_medicos > 0.05:
    print(" BUENO: Fine-tuning efectivo para pares médicos")
elif mejora_prom_medicos > 0.02:
    print(" MODERADO: Fine-tuning con mejora leve")
else:
    print(" LIMITADO: Fine-tuning con impacto mínimo")

if abs(mejora_prom_control) < 0.02:
    print(" ESPECÍFICO: No afecta pares no relacionados (buen control)")
else:
    print(" INESPECÍFICO: También afecta pares no relacionados")

selectividad = mejora_prom_medicos - mejora_prom_control
print(f" Selectividad: {selectividad:+.4f} (diferencia médicos vs control)")

