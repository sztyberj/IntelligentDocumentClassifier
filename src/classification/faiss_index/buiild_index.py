import sys
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data_processing.data_prep import DataPrep

data_prep = DataPrep(config_path="./config.yml")
config = data_prep.get_config()

#MODEL_NAME = config["train"]["output_path"]
MODEL_NAME = "herbert_poc"
DATA_DIR = config["data"]["data_dir"]
INDEX_FILE = config["index"]["index_file"]
META_FILE = config["index"]["meta_file"]

def build_index():
    #1. Init
    print("1. Loading config and model")
    
    if not os.path.exists(MODEL_NAME):
        print("Model not found")
        return
    
    model = SentenceTransformer(MODEL_NAME)

    all_embeddings = []
    metadata = []

    #2. Docs processing
    print("2. Docs processing")

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    print(f"\n Classes found (dirs): {classes}")

    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(".pdf")]

        for file in files:
            file_path = os.path.join(class_path, file)
            full_text = data_prep.read_pdf(file_path)

            if len(full_text.strip()) < 50: continue

            chunks = []

            if len(full_text) <= data_prep.window_size:
                chunks.append(full_text)
            else:
                for i in range(0, len(full_text), data_prep.window_size - data_prep.overlap):
                    chunk = full_text[i : i + data_prep.window_size]
                    if len(chunk) > 50: chunks.append(chunk)

            if chunks:
                vecs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

                for vec in vecs:
                    all_embeddings.append(vec)
                    metadata.append({
                        "category": class_name,
                        "filename": file
                    })


    #3. Buiding FAISS
    print(f"3. Building indexes for {len(all_embeddings)} fragments...")

    embeddings_np = np.array(all_embeddings).astype('float32')

    #IndexFlatIP = Inner Product 
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    #4. Save
    print(f"Saving to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("Sucess!")

if __name__ == "__main__":
    build_index()