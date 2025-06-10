from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import matplotlib.pyplot as plt

# -------- Función para calcular similitud --------
def calcular_similitud(modelo, frase1, frase2):
    """
    Calcula la similitud coseno entre dos frases usando un modelo dado
    """
    emb1 = modelo.encode(frase1, convert_to_tensor=True)
    emb2 = modelo.encode(frase2, convert_to_tensor=True)
    similitud = util.cos_sim(emb1, emb2).item()
    return similitud

# -------- Configurar modelos --------
print("🔄 Cargando modelos BGE-M3...")

# Modelo baseline (original)
modelo_base = SentenceTransformer("BAAI/bge-m3")
print("✅ BGE-M3 baseline cargado")

# Modelo fine-tuneado - AJUSTA ESTA RUTA
ruta_modelo_finetuneado = "D:/TFM/models/bge_m3_epochs/epoch4_MRR0.9717"

if not os.path.isdir(ruta_modelo_finetuneado):
    raise ValueError(f" Ruta '{ruta_modelo_finetuneado}' no encontrada")

modelo_finetuneado = SentenceTransformer(ruta_modelo_finetuneado)
print("✅ BGE-M3 fine-tuneado cargado")

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
    ("diabetes", "fractura ósea"),
    ("hipertensión", "resfriado común"),
    ("infarto", "alergia alimentaria"),
    ("ictus", "gripe e"),
    ("edema", "fiebre alta"),
    ("tumor pulmón", "dolor muelas"),
    ("depresión", "catarro"),
    ("epilepsia", "indigestión"),
    ("VIH", "caída pelo"),
    ("angina", "uña encarnada")
]

print(f"\n Evaluando {len(pares_medicos)} pares médicos y {len(pares_control)} pares control...")

# -------- Evaluación de pares médicos --------
resultados_medicos = []
print("\n PARES MÉDICOS (debería mejorar con fine-tuning):")
print("-" * 80)

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
    if mejora > 0.15:
        indicador = " EXCELENTE"
    elif mejora > 0.10:
        indicador = " MUY BUENA"
    elif mejora > 0.05:
        indicador = " BUENA"
    elif mejora > 0.02:
        indicador = " LEVE"
    else:
        indicador = " SIN MEJORA"
    
    print(f"{termino_tecnico:18} ↔ {termino_coloquial:20} | "
          f"Base: {sim_base:.3f} | FT: {sim_ft:.3f} | "
          f"Δ: {mejora:+.3f} ({mejora_pct:+.1f}%) {indicador}")

# -------- Evaluación de pares control --------
resultados_control = []
print(f"\n PARES CONTROL (NO debería mejorar mucho):")
print("-" * 80)

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
    
    print(f"{termino1:18} ↔ {termino2:20} | "
          f"Base: {sim_base:.3f} | FT: {sim_ft:.3f} | "
          f"Δ: {mejora:+.3f} ({mejora_pct:+.1f}%)")

# -------- Análisis estadístico --------
mejoras_medicos = [r['mejora'] for r in resultados_medicos]
mejoras_control = [r['mejora'] for r in resultados_control]

print(f"\n ANÁLISIS ESTADÍSTICO:")
print("=" * 60)
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

# -------- Interpretación específica para BGE-M3 --------
mejora_prom_medicos = np.mean(mejoras_medicos)
mejora_prom_control = np.mean(mejoras_control)

print(f"\n INTERPRETACIÓN BGE-M3 FINE-TUNING:")
print("=" * 60)

if mejora_prom_medicos > 0.15:
    print(" EXCELENTE: Fine-tuning extremadamente efectivo")
    print("   → BGE-M3 ha mejorado significativamente en comprensión médica")
elif mejora_prom_medicos > 0.10:
    print(" MUY BUENO: Fine-tuning muy efectivo")
    print("   → BGE-M3 muestra clara mejora en el dominio médico")
elif mejora_prom_medicos > 0.05:
    print(" BUENO: Fine-tuning efectivo")
    print("   → BGE-M3 ha ganado conocimiento médico específico")
elif mejora_prom_medicos > 0.02:
    print(" MODERADO: Fine-tuning con mejora leve")
    print("   → Algunas mejoras detectables")
else:
    print(" LIMITADO: Fine-tuning con impacto mínimo")
    print("   → El modelo ya era robusto o necesita más entrenamiento")

if abs(mejora_prom_control) < 0.03:
    print(" ESPECÍFICO: No afecta términos no relacionados")
    print("   → El fine-tuning es quirúrgico y preciso")
else:
    print(" INESPECÍFICO: También afecta términos no relacionados")
    print("   → Posible sobreajuste o cambio general en representaciones")

selectividad = mejora_prom_medicos - mejora_prom_control
print(f" Selectividad: {selectividad:+.4f}")

if selectividad > 0.10:
    print("   → EXCELENTE: Muy específico para dominio médico")
elif selectividad > 0.05:
    print("   → BUENO: Razonablemente específico")
else:
    print("   → MODERADO: Mejora general más que específica")

# -------- Ranking de mejores mejoras --------
print(f"\n TOP 5 MEJORES MEJORAS:")
print("-" * 50)
top_mejoras = sorted(resultados_medicos, key=lambda x: x['mejora'], reverse=True)[:5]
for i, resultado in enumerate(top_mejoras, 1):
    print(f"{i}. {resultado['par']}")
    print(f"   Mejora: {resultado['mejora']:+.4f} ({resultado['mejora_pct']:+.1f}%)")

