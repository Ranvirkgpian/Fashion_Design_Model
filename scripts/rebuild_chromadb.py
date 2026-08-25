import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, json, torch
import pandas as pd
import numpy as np
import chromadb

print("=" * 60)
print("  Rebuilding ChromaDB with cosine distance metric")
print("=" * 60)

ROOT = r"E:\Fashion model"
EMB_DIR = os.path.join(ROOT, r"data\embeddings")
DB_DIR  = os.path.join(ROOT, r"data\chromadb")

embeddings = np.load(os.path.join(EMB_DIR, "clip_embeddings.npy"))
metadata   = pd.read_csv(os.path.join(EMB_DIR, "embedding_metadata.csv"))
metadata["id"] = metadata["id"].astype(str)

client = chromadb.PersistentClient(path=DB_DIR)
try:
    client.delete_collection("fashion_styles")
except Exception:
    pass

# Explicitly set cosine distance metric
collection = client.create_collection(
    name="fashion_styles",
    metadata={"description": "Fashion style embeddings", "hnsw:space": "cosine"}
)
print("  Collection created with cosine distance metric")

BATCH = 200
for i in range(0, len(metadata), BATCH):
    batch_meta = metadata.iloc[i:i+BATCH]
    batch_emb  = embeddings[i:i+BATCH]
    ids = batch_meta["id"].tolist()
    docs = batch_meta["search_description"].tolist()
    metas = [
        {"articleType": str(r["articleType"]), "baseColour": str(r["baseColour"]),
         "gender": str(r["gender"]), "trend_category": str(r["trend_category"]),
         "trend_score": float(r["trend_score"])}
        for _, r in batch_meta.iterrows()
    ]
    collection.add(ids=ids, embeddings=batch_emb.tolist(), documents=docs, metadatas=metas)
    if (i // BATCH) % 3 == 0:
        print(f"  Inserted {min(i+BATCH, len(metadata))}/{len(metadata)}")

print(f"\n  Collection rebuilt: {collection.count():,} items with cosine metric")
print("=" * 60)
