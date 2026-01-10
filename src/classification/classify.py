import sys
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from src.data_processing.data_prep import DataPrep

#1. Init DataPrep and Config
data_prep = DataPrep(config_path="./config.yml")
config = data_prep.get_config()

#MODEL_NAME = config["train"]["output_path"]
MODEL_NAME = "herbert_poc"
INDEX_FILE = config["index"]["index_file"]
META_FILE = config["index"]["meta_file"]
K_NEIGHBORS = config["search"]["k_neighbors"]

class DocumentClassifier:
    def __init__(self):
        print("Init classifier")

        if not os.path.exists(MODEL_NAME):
            raise FileExistsError(f"Model not found {MODEL_NAME}")
        if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
            raise FileExistsError("Index files not found")
        
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_FILE)

        with open(META_FILE, "rb") as f:
            self.metadata = pickle.load(f)

        print("System ready!")


    def classify(self, pdf_path):
        print(f"\n File analyzing: {os.path.basename(pdf_path)}")

        #1. Read and cut file
        full_text = data_prep.read_pdf(pdf_path)

        if len(full_text) < 50:
            return {"error": "File empty or too short"}
        
        chunks = []
        if len(full_text) <= data_prep.window_size:
            chunks.append(full_text)
        else:
            for i in range(0, len(full_text), data_prep.window_size - data_prep.overlap):
                chunk = full_text[i: i + data_prep.window_size]
                if len(chunk) > 50: chunks.append(chunk)

        if not chunks:
            return {"error": "No text to analyze"}
        
        #2. Change to vetors
        query_vectors = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        #3. Search in FAISS
        distances, indices = self.index.search(query_vectors, K_NEIGHBORS)

        #4. Weighted Voting
        class_scores = defaultdict(float)
        chunk_details = []

        total_chunks = len(chunks)
        print(f"{total_chunks} processed. Voting...")

        for i in range(total_chunks):
            for j in range(K_NEIGHBORS):
                idx = indices[i][j]
                score = distances[i][j]

                neighbor_meta = self.metadata[idx]
                category = neighbor_meta['category']

                #Points logic
                if score > 0.4:
                    class_scores[category] += score
        
        #5. Results
        if not class_scores:
            return []
        
        sorted_results = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)

        return sorted_results
    
#USER INTERFACE
if __name__ == "__main__":
    try:
        classifier = DocumentClassifier()
    except Exception as e:
        print(f"Critical error: {e}")
        exit()

    print("\n--- Inteligentny Klasyfikator Dokumentow IDK ---")
    print("Type file .pdf path")
    print("Type 'exit' or 'q' to close program\n")

    while True:
        user_input = input("PDF file path: ").strip()

        if user_input.lower() in ['exit', 'q', 'quit']:
            print("Bye! 👋")
            break

        user_input = user_input.strip('"').strip("'")

        if not os.path.exists(user_input):
            print("❌ File not exist")

        if not user_input.lower().endswith('.pdf'):
            print("⚠️ This is not PDF file!")

        results = classifier.classify(user_input)

        #Show results
        if isinstance(results, dict) and "error" in results:
            print(f"⚠️ {results['error']}")
        elif not results:
            print("❓ Category not found!")
        else:
            print("\n Classification")
            print("-" * 30)

            winner, score = results[0]

            print(f"Winner: {winner.upper()}")
            print(f"   Score: {score:.2f}")

            print("Other classes:")
            for category, score in results[1:]:
                print(f"   - {category}: {score:.2f}")
            print("-" * 30 + "\n")