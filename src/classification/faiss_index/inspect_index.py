import faiss
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random

INDEX_FILE = "knn_index.bin"
META_FILE = "knn_metadata.pkl"

def inspect():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        print("❌ Brak plików bazy! Uruchom najpierw build_index.py")
        return

    print("1. Ładowanie bazy...")
    index = faiss.read_index(INDEX_FILE)
    
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    # --- CZĘŚĆ 1: STATYSTYKI ---
    total_vectors = index.ntotal
    dimension = index.d
    
    print("\n📊 STATYSTYKI BAZY WEKTOROWEJ:")
    print(f"   Liczba wektorów (fragmentów tekstu): {total_vectors}")
    print(f"   Wymiar pojedynczego wektora: {dimension}")
    print(f"   Liczba metadanych: {len(metadata)}")

    # --- CZĘŚĆ 2: PODGLĄD PRZYKŁADU ---
    print("\n🔍 LOSOWY PRZYKŁAD:")
    rand_idx = random.randint(0, total_vectors - 1)
    
    # Wyciągamy wektor z FAISS (reconstruct działa dla IndexFlat)
    vec = index.reconstruct(rand_idx)
    meta = metadata[rand_idx]
    
    print(f"   ID: {rand_idx}")
    print(f"   Kategoria: [{meta['category'].upper()}]")
    print(f"   Plik: {meta['filename']}")
    print(f"   Wektor (pierwsze 5 liczb): {vec[:5]} ...")

    # --- CZĘŚĆ 3: WIZUALIZACJA 2D (PCA) ---
    print("\n🎨 Generowanie mapy 2D (to może chwilę potrwać)...")
    
    # 1. Pobieramy wszystkie wektory z FAISS
    # Uwaga: Dla bardzo dużych baz (miliony) to by zapchało RAM, ale przy <100k jest OK.
    all_vectors = []
    for i in range(total_vectors):
        all_vectors.append(index.reconstruct(i))
    
    X = np.array(all_vectors)
    
    # 2. Pobieramy etykiety (kategorie)
    y = [m['category'] for m in metadata]
    categories = list(set(y))
    
    # 3. Redukcja wymiarów (768 -> 2)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # 4. Rysowanie wykresu
    plt.figure(figsize=(12, 8))
    
    # Mapa kolorów
    colors = plt.cm.get_cmap('tab10', len(categories))
    
    for i, category in enumerate(categories):
        # Wybieramy punkty tylko dla tej kategorii
        indices = [j for j, label in enumerate(y) if label == category]
        points = X_2d[indices]
        
        plt.scatter(points[:, 0], points[:, 1], label=category, alpha=0.6, s=15)

    plt.title("Mapa Twoich Dokumentów (PCA)", fontsize=16)
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "index_visualization.png"
    plt.savefig(output_img)
    print(f"✅ Wykres zapisano jako: {output_img}")
    print("   Otwórz ten plik, aby zobaczyć, jak model grupuje dokumenty!")
    # plt.show() # Odkomentuj, jeśli masz środowisko graficzne

if __name__ == "__main__":
    inspect()