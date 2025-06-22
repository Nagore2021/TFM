import json
import os

# Paths
CORPUS_PATH = "corpus_medico.json"
QUERIES_PATH = "test_queries.json"
OUTPUT_PATH = "dataset_test.json"

def generar_dataset_test():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    dataset = {
        "queries": queries_data["queries"],
        "corpus": corpus_data["corpus"]
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset de evaluación creado: {OUTPUT_PATH} (queries + corpus)")

if __name__ == "__main__":
    generar_dataset_test()
