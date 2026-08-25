import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, json
import pandas as pd
import numpy as np
import chromadb

print("=" * 60)
print("  Week 5 Task 3 — ChromaDB Vector Storage & Retrieval")
print("=" * 60)

ROOT = r"E:\Fashion model"
EMB_DIR = os.path.join(ROOT, r"data\embeddings")
DB_DIR  = os.path.join(ROOT, r"data\chromadb")

# ── Load embeddings + metadata ─────────────────────────────
print("\n  Loading embeddings and metadata...")
embeddings = np.load(os.path.join(EMB_DIR, "clip_embeddings.npy"))
metadata   = pd.read_csv(os.path.join(EMB_DIR, "embedding_metadata.csv"))
metadata["id"] = metadata["id"].astype(str)

print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Metadata rows   : {len(metadata)}")

# ── Initialize ChromaDB ────────────────────────────────────
print("\n  Initializing ChromaDB...")
client = chromadb.PersistentClient(path=DB_DIR)

# Fresh collection each run
try:
    client.delete_collection("fashion_styles")
except Exception:
    pass

collection = client.create_collection(
    name="fashion_styles",
    metadata={"description": "Fashion style embeddings for recommendation"}
)
print("  Collection created: fashion_styles")

# ── Insert in batches ──────────────────────────────────────
print("\n  Inserting embeddings into ChromaDB...")
BATCH = 200

for i in range(0, len(metadata), BATCH):
    batch_meta = metadata.iloc[i:i+BATCH]
    batch_emb  = embeddings[i:i+BATCH]

    ids = batch_meta["id"].tolist()
    docs = batch_meta["search_description"].tolist()
    metas = [
        {
            "articleType": str(row["articleType"]),
            "baseColour": str(row["baseColour"]),
            "gender": str(row["gender"]),
            "trend_category": str(row["trend_category"]),
            "trend_score": float(row["trend_score"]),
        }
        for _, row in batch_meta.iterrows()
    ]

    collection.add(
        ids=ids,
        embeddings=batch_emb.tolist(),
        documents=docs,
        metadatas=metas,
    )

    if (i // BATCH) % 3 == 0:
        print(f"  Inserted {min(i+BATCH, len(metadata))}/{len(metadata)} items...")

print(f"\n  Total items in collection: {collection.count()}")

# ── Test retrieval ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  Testing Similarity Search")
print("=" * 60)

test_idx = 0
query_emb = embeddings[test_idx].tolist()
query_item = metadata.iloc[test_idx]

print(f"\n  Query item: {query_item['search_description']}")
print(f"  Trend category: {query_item['trend_category']}")

results = collection.query(
    query_embeddings=[query_emb],
    n_results=6,
)

print(f"\n  Top 5 similar items (excluding self):")
print("-" * 60)
count = 0
for doc, meta, dist, rid in zip(
    results["documents"][0], results["metadatas"][0],
    results["distances"][0], results["ids"][0]
):
    if rid == query_item["id"]:
        continue
    count += 1
    similarity = 1 - dist
    print(f"  {count}. [{similarity:.3f}] {doc}")
    if count >= 5:
        break

print("\n" + "=" * 60)
print(f"  ChromaDB setup complete!")
print(f"  {collection.count():,} items indexed and searchable")
print(f"  Database location: {DB_DIR}")
print("=" * 60)
