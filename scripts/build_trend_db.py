import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import pandas as pd
import json, os

ROOT = r"E:\Fashion model"
CSV  = os.path.join(ROOT, r"data\deepfashion\processed\fashion_metadata_clean.csv")
OUT_DIR = os.path.join(ROOT, r"data\trend_database")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Week 5 Task 1 — Fashion Trend Database")
print("=" * 60)

df = pd.read_csv(CSV)
print(f"\n  Loaded {len(df):,} items from Week 1 dataset")

trend_rules = {
    ("Summer", "Casual"):  {"trend": "Resort Casual",      "score": 0.92},
    ("Summer", "Formal"):  {"trend": "Summer Elegance",    "score": 0.85},
    ("Winter", "Casual"):  {"trend": "Cozy Layering",      "score": 0.88},
    ("Winter", "Formal"):  {"trend": "Winter Sophistication","score": 0.90},
    ("Fall",   "Casual"):  {"trend": "Transitional Layers","score": 0.86},
    ("Fall",   "Formal"):  {"trend": "Autumn Tailoring",   "score": 0.83},
    ("Spring", "Casual"):  {"trend": "Fresh Minimalism",   "score": 0.89},
    ("Spring", "Formal"):  {"trend": "Spring Refinement",  "score": 0.84},
}

def get_trend(row):
    key = (row["season"], row["usage"])
    info = trend_rules.get(key, {"trend": "Classic Staple", "score": 0.70})
    return info["trend"], info["score"]

print("\n  Computing trend scores...")
trends = df.apply(get_trend, axis=1)
df["trend_category"] = [t[0] for t in trends]
df["trend_score"]     = [t[1] for t in trends]

complement_map = {
    "Topwear": "Bottomwear", "Bottomwear": "Topwear",
    "Footwear": "Topwear", "Accessories": "Topwear",
    "Dress": "Footwear", "Innerwear": "Topwear",
}
df["pairs_with"] = df["subCategory"].map(complement_map).fillna("Accessories")

def build_description(row):
    return (
        f"{row['baseColour']} {row['articleType']} for {row['gender']}, "
        f"{row['trend_category']} style, {row['usage']} occasion, "
        f"{row['season']} season"
    )

df["search_description"] = df.apply(build_description, axis=1)

out_csv = os.path.join(OUT_DIR, "trend_database.csv")
df.to_csv(out_csv, index=False)

print("\n  Trend Category Distribution:")
print("-" * 60)
trend_counts = df["trend_category"].value_counts()
for trend, count in trend_counts.items():
    bar = "█" * int(count / len(df) * 50)
    pct = count / len(df) * 100
    print(f"  {trend:<24} {count:>6,} ({pct:>4.1f}%)  {bar}")

vocab = {
    "trend_categories": sorted(df["trend_category"].unique().tolist()),
    "total_items": len(df),
    "avg_trend_score": round(df["trend_score"].mean(), 3),
    "complement_mapping": complement_map,
}
with open(os.path.join(OUT_DIR, "trend_vocabulary.json"), "w") as f:
    json.dump(vocab, f, indent=2)

print(f"\n  Sample enriched records:")
print("-" * 60)
for _, row in df.sample(3, random_state=42).iterrows():
    print(f"  [{row['trend_category']}] {row['search_description']}")
    print(f"    Trend score: {row['trend_score']}  |  Pairs with: {row['pairs_with']}")
    print()

n_trends = len(vocab["trend_categories"])
print("=" * 60)
print(f"  Trend database built: {len(df):,} items")
print(f"  {n_trends} trend categories identified")
print(f"  Saved to: {out_csv}")
print(f"  Vocabulary: {OUT_DIR}\\trend_vocabulary.json")
print("=" * 60)
