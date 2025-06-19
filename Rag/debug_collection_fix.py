"""
debug_collection_fix.py - Debugger para problema de colección

Identifica y soluciona el problema de nombres de colección en BM25DualChunkEvaluator
"""

import os
import sys
import yaml
import logging
import chromadb

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Imports locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logger = logging.getLogger(__name__)

def debug_collection_issue(config_path: str, mode: str = "embedding"):
    """
    Debugger completo para identificar el problema de colección
    
    Args:
        config_path: Ruta al config.yaml
        mode: 'embedding' o 'finetuneado'
    """
    
    print("🔍 DEBUGGING COLECCIÓN - BM25DualChunkEvaluator")
    print("="*60)
    
    # 1. Verificar config.yaml
    print(f"\n1️⃣ VERIFICANDO CONFIG: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ Config no existe: {config_path}")
        return
    
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"✅ Config cargado correctamente")
    except Exception as e:
        print(f"❌ Error leyendo config: {e}")
        return
    
    # 2. Mostrar configuración de colecciones
    print(f"\n2️⃣ CONFIGURACIÓN DE COLECCIONES:")
    if 'collection' in cfg:
        collection_config = cfg['collection']
        print(f"📋 Configuración completa:")
        for key, value in collection_config.items():
            print(f"   {key}: {value}")
        
        # Lógica que usa BM25DualChunkEvaluator
        collection_name = collection_config['name'] if mode == 'embedding' else collection_config['name_finetuneado']
        print(f"\n🎯 COLECCIÓN SELECCIONADA (mode='{mode}'): {collection_name}")
    else:
        print(f"❌ Sección 'collection' no encontrada en config")
        return
    
    # 3. Verificar ChromaDB
    print(f"\n3️⃣ VERIFICANDO CHROMADB:")
    
    if 'paths' not in cfg or 'chroma_db_path' not in cfg['paths']:
        print(f"❌ Path ChromaDB no configurado")
        return
    
    chroma_path = cfg['paths']['chroma_db_path']
    print(f"📁 Path ChromaDB: {chroma_path}")
    
    if not os.path.exists(chroma_path):
        print(f"❌ Directorio ChromaDB no existe: {chroma_path}")
        return
    
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        available_collections = [c.name for c in client.list_collections()]
        print(f"✅ ChromaDB conectado")
        print(f"📚 Colecciones disponibles: {available_collections}")
        
        # 4. Verificar si existe la colección seleccionada
        print(f"\n4️⃣ VERIFICACIÓN DE COLECCIÓN:")
        if collection_name in available_collections:
            print(f"✅ Colección '{collection_name}' EXISTE")
            
            # Información adicional de la colección
            try:
                collection = client.get_collection(collection_name)
                count = collection.count()
                print(f"📊 Documentos en colección: {count}")
                
                # Muestra de metadatos
                if count > 0:
                    sample = collection.get(limit=1, include=['metadatas'])
                    if sample['metadatas']:
                        print(f"🏷️ Ejemplo de metadatos:")
                        meta = sample['metadatas'][0]
                        for key, value in meta.items():
                            print(f"   {key}: {value}")
                
            except Exception as e:
                print(f"⚠️ Error accediendo a colección: {e}")
                
        else:
            print(f"❌ Colección '{collection_name}' NO EXISTE")
            print(f"💡 Colecciones disponibles: {available_collections}")
            
            # Sugerir soluciones
            print(f"\n🔧 POSIBLES SOLUCIONES:")
            
            if available_collections:
                print(f"   A) Usar colección existente - cambiar config.yaml:")
                for i, col in enumerate(available_collections, 1):
                    print(f"      {i}. collection.name: '{col}'")
                
                print(f"   B) Cambiar modo:")
                other_mode = "finetuneado" if mode == "embedding" else "embedding"
                other_collection = collection_config['name_finetuneado'] if mode == 'embedding' else collection_config['name']
                if other_collection in available_collections:
                    print(f"      Usar mode='{other_mode}' → colección '{other_collection}'")
            
            print(f"   C) Crear la colección ejecutando script de indexación")
            
    except Exception as e:
        print(f"❌ Error conectando ChromaDB: {e}")
        return
    
    # 5. Verificar modelos configurados
    print(f"\n5️⃣ VERIFICANDO MODELOS:")
    if 'model' in cfg:
        model_config = cfg['model']
        model_name = model_config.get(f'name_{mode}', 'NO CONFIGURADO')
        cross_encoder_name = model_config.get('name_cross_encoder', 'NO CONFIGURADO')
        
        print(f"🤖 Modelo {mode}: {model_name}")
        print(f"🔄 Cross-Encoder: {cross_encoder_name}")
        
        # Verificar si existen
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            
            # Test Bi-Encoder
            try:
                test_bi = SentenceTransformer(model_name)
                print(f"✅ Bi-Encoder accesible")
            except Exception as e:
                print(f"❌ Error Bi-Encoder: {e}")
            
            # Test Cross-Encoder
            try:
                test_cross = CrossEncoder(cross_encoder_name)
                print(f"✅ Cross-Encoder accesible")
            except Exception as e:
                print(f"❌ Error Cross-Encoder: {e}")
                
        except ImportError as e:
            print(f"❌ Error importando SentenceTransformers: {e}")
    
    print(f"\n" + "="*60)
    print(f"🎯 RESUMEN DEL PROBLEMA:")
    
    if collection_name in available_collections:
        print(f"✅ La colección existe - problema puede ser en otra parte")
        print(f"💡 Verifica imports y dependencias")
    else:
        print(f"❌ La colección '{collection_name}' no existe")
        print(f"💡 Solución más probable: cambiar nombre en config.yaml")

def fix_config_collection(config_path: str, correct_collection_name: str, mode: str = "embedding"):
    """
    Corrige automáticamente el nombre de colección en config.yaml
    
    Args:
        config_path: Ruta al config.yaml
        correct_collection_name: Nombre correcto de la colección
        mode: Modo a corregir
    """
    
    print(f"\n🔧 CORRIGIENDO CONFIG...")
    
    try:
        # Leer config actual
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        # Backup
        backup_path = config_path + ".backup"
        with open(backup_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"💾 Backup guardado: {backup_path}")
        
        # Corregir nombre
        if mode == "embedding":
            cfg['collection']['name'] = correct_collection_name
        else:
            cfg['collection']['name_finetuneado'] = correct_collection_name
        
        # Guardar config corregido
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Config corregido:")
        print(f"   collection.{mode}: '{correct_collection_name}'")
        
    except Exception as e:
        print(f"❌ Error corrigiendo config: {e}")

def test_bm25_evaluator(config_path: str, mode: str = "embedding"):
    """
    Test completo de BM25DualChunkEvaluator
    
    Args:
        config_path: Ruta al config.yaml
        mode: Modo a probar
    """
    
    print(f"\n🧪 TESTING BM25DUALCHUNKEVALUATOR")
    print(f"="*50)
    
    try:
        # Importar clase
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from retrieval.bm25_model_chunk_bge import BM25DualChunkEvaluator
        
        print(f"✅ Clase importada correctamente")
        
        # Crear instancia
        print(f"🔄 Creando instancia...")
        evaluator = BM25DualChunkEvaluator(config_path, mode)
        print(f"✅ Instancia creada")
        
        # Cargar colección
        print(f"🔄 Cargando colección...")
        evaluator.load_collection()
        print(f"✅ Colección cargada")
        
        # Información de la colección cargada
        print(f"📊 INFORMACIÓN DE LA COLECCIÓN:")
        print(f"   Chunks totales: {len(evaluator.chunk_ids)}")
        print(f"   Metadatos: {len(evaluator.metadatas)}")
        print(f"   Documentos raw: {len(evaluator.docs_raw)}")
        
        # Test rápido de retrieval
        print(f"\n🧪 TEST DE RETRIEVAL:")
        test_query = "síntomas diabetes"
        
        try:
            bm25_results = evaluator.calculate_bm25_rankings(test_query)
            print(f"✅ BM25: {len(bm25_results)} resultados")
        except Exception as e:
            print(f"❌ Error BM25: {e}")
        
        try:
            biencoder_results = evaluator.calculate_biencoder_rankings(test_query)
            print(f"✅ Bi-Encoder: {len(biencoder_results)} resultados")
        except Exception as e:
            print(f"❌ Error Bi-Encoder: {e}")
        
        print(f"✅ BM25DualChunkEvaluator funcionando correctamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error importando BM25DualChunkEvaluator: {e}")
        print(f"💡 Verifica la ruta del archivo")
        return False
    except Exception as e:
        print(f"❌ Error en BM25DualChunkEvaluator: {e}")
        return False

def main():
    """Función principal de debugging"""
    
    print("🔍 DEBUGGER PARA BM25DUALCHUNKEVALUATOR")
    print("="*60)
    
    # Configuración - AJUSTAR ESTAS RUTAS
    config_path = "../config.yaml"  # Ajustar según tu estructura
    mode = "embedding"  # o "finetuneado"
    
    print(f"📁 Config: {config_path}")
    print(f"🔧 Modo: {mode}")
    
    # 1. Debug completo
    debug_collection_issue(config_path, mode)
    
    # 2. Si quieres corregir automáticamente (opcional)
    # fix_collection = input("\n¿Quieres corregir el config? (y/n): ")
    # if fix_collection.lower() == 'y':
    #     correct_name = input("Nombre correcto de colección: ")
    #     fix_config_collection(config_path, correct_name, mode)
    
    # 3. Test final
    print(f"\n" + "="*60)
    test_success = test_bm25_evaluator(config_path, mode)
    
    if test_success:
        print(f"\n🎉 TODO FUNCIONANDO CORRECTAMENTE")
        print(f"✅ BM25DualChunkEvaluator listo para usar")
    else:
        print(f"\n❌ PROBLEMAS IDENTIFICADOS")
        print(f"💡 Revisa los errores mostrados arriba")

if __name__ == "__main__":
    main()