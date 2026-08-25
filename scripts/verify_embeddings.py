import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os
import numpy as np
import chromadb

ROOT = r"E:\Fashion model"
DB_DIR = os.path.join(ROOT, r"data\chromadb")
EMB_DIR = os.path.join(ROOT, r"data\embeddings")

# Check the .npy file on disk
local_emb = np.load(os.path.join(EMB_DIR, "clip_embeddings.npy"))
print("Local .npy file first 5 values of item 0:", local_emb[0][:5])

# Check what's actually stored in ChromaDB
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("fashion_styles")
result = collection.get(limit=1, include=["embeddings"])
print("\nChromaDB stored embedding first 5 values:", result["embeddings"][0][:5])

print("\nDo they match?", np.allclose(local_emb[0][:5], result["embeddings"][0][:5], atol=1e-4))
print("Collection count:", collection.count())
