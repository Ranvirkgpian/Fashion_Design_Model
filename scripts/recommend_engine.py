import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, json, torch
import pandas as pd
import numpy as np
import chromadb
from transformers import CLIPProcessor, CLIPModel

print("=" * 60)
print("  Week 5 Task 4 — Recommendation Engine (final fix)")
print("=" * 60)

ROOT = r"E:\Fashion model"
DB_DIR = os.path.join(ROOT, r"data\chromadb")
OUT_DIR = os.path.join(ROOT, r"outputs")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n  Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("fashion_styles")
print(f"  Loaded collection: {collection.count():,} items")

# Correct text encoding: text_model -> pooler_output -> text_projection
def encode_text(query_text):
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"])
        pooled = text_out.pooler_output
        projected = model.text_projection(pooled)
        projected = projected / projected.norm(dim=-1, keepdim=True)
    return projected.cpu().numpy().tolist()

def recommend_by_text(query_text, gender_filter=None, n=5):
    where_clause = {"gender": gender_filter} if gender_filter else None
    query_emb = encode_text(query_text)
    results = collection.query(query_embeddings=query_emb, n_results=n, where=where_clause)
    return results

def recommend_complete_look(item_id, embeddings_dict, metadata_df, n=4):
    if item_id not in embeddings_dict:
        return None
    query_emb = embeddings_dict[item_id]
    results = collection.query(query_embeddings=[query_emb], n_results=20)
    recs, seen = [], set()
    base = metadata_df[metadata_df["id"] == item_id]
    if base.empty: return None
    seen.add(base.iloc[0]["articleType"])
    for doc, meta, dist, rid in zip(results["documents"][0], results["metadatas"][0],
                                     results["distances"][0], results["ids"][0]):
        if rid == item_id or meta["articleType"] in seen:
            continue
        recs.append({"id": rid, "description": doc, "similarity": round(1-dist,3),
                     "trend_category": meta.get("trend_category","")})
        seen.add(meta["articleType"])
        if len(recs) >= n: break
    return recs

embeddings = np.load(os.path.join(ROOT, r"data\embeddings\clip_embeddings.npy"))
metadata = pd.read_csv(os.path.join(ROOT, r"data\embeddings\embedding_metadata.csv"))
metadata["id"] = metadata["id"].astype(str)
emb_dict = {row["id"]: embeddings[i].tolist() for i, row in metadata.iterrows()}

print("\n" + "=" * 60)
print("  Demo 1: Text-Based Style Search (final fix)")
print("=" * 60)

queries = [
    ("minimalist black outfit for casual wear", "Women"),
    ("formal navy blue business attire", "Men"),
    ("cozy winter layering pieces", None),
]

for query_text, gender in queries:
    print(f"\n  Query: \"{query_text}\"" + (f" (gender: {gender})" if gender else ""))
    results = recommend_by_text(query_text, gender, n=3)
    if results["documents"] and len(results["documents"][0]) > 0:
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"    {i+1}. [Score: {1-dist:.3f}] {doc}")
    else:
        print("    No matches found")

print("\n" + "=" * 60)
print("  Demo 2: Complete-the-Look Recommendations")
print("=" * 60)

sample_items = metadata.sample(3, random_state=7)
for _, item in sample_items.iterrows():
    print(f"\n  Base item: {item['search_description']}")
    recs = recommend_complete_look(item["id"], emb_dict, metadata, n=3)
    if recs:
        print(f"  Complete the look with:")
        for r in recs:
            print(f"    -> [{r['similarity']:.3f}] {r['description']}")
    else:
        print("    No complementary items found")

report = {
    "collection_size": collection.count(),
    "text_search_demo": [{"query": q, "gender_filter": g} for q, g in queries],
    "complete_look_demo": [{"base_item": item["search_description"]} for _, item in sample_items.iterrows()],
}
with open(os.path.join(OUT_DIR, "week5_recommendation_demo.json"), "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print(f"  Recommendation engine complete!")
print(f"  Report saved: outputs\\week5_recommendation_demo.json")
print("=" * 60)
