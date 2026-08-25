import pandas as pd
import os, json

ROOT    = r"E:\Fashion model"
CSV     = os.path.join(ROOT, r"data\deepfashion\raw\styles.csv")
OUT_DIR = os.path.join(ROOT, r"data\deepfashion\processed")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Fashion Metadata Pipeline — Week 1 Task 3 & 4")
print("=" * 60)

# Step 1: Load
df = pd.read_csv(CSV, on_bad_lines='skip')
df['id'] = df['id'].astype(str)
print(f"\n  Step 1 ✓  Loaded {len(df):,} raw records")

# Step 2: Clean missing values (pandas 3.0 syntax)
df['baseColour']        = df['baseColour'].fillna('Unknown')
df['season']            = df['season'].fillna('Unknown')
df['usage']             = df['usage'].fillna('Unknown')
df['year']              = df['year'].fillna(df['year'].median())
df['productDisplayName']= df['productDisplayName'].fillna('No Name')
print(f"  Step 2 ✓  Missing values filled")

# Step 3: Normalize text
for col in ['gender','masterCategory','subCategory','articleType','baseColour','season','usage']:
    df[col] = df[col].astype(str).str.strip().str.title()
print(f"  Step 3 ✓  Text columns normalized")

# Step 4: Add image paths
df['image_path']   = df['id'].apply(lambda x: os.path.join(ROOT, r"data\deepfashion\raw\images", f"{x}.jpg"))
df['image_exists'] = df['image_path'].apply(os.path.exists)
found = df['image_exists'].sum()
print(f"  Step 4 ✓  Image paths added ({found:,} images found on disk)")

# Step 5: Build prompt templates (safe string conversion)
def build_prompt(row):
    return (f"A {str(row['gender']).lower()} {str(row['articleType']).lower()}, "
            f"{str(row['baseColour']).lower()} color, "
            f"{str(row['usage']).lower()} style, "
            f"for {str(row['season']).lower()} season")
df['prompt_template'] = df.apply(build_prompt, axis=1)
print(f"  Step 5 ✓  Prompt templates generated")

# Step 6: Encode categories
for col in ['masterCategory','subCategory','articleType','baseColour','gender','season','usage']:
    df[col + '_id'] = pd.Categorical(df[col]).codes
print(f"  Step 6 ✓  Numeric label encoding done")

# Step 7: Train/Val/Test split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
df['split'] = 'test'
df.loc[:train_end,        'split'] = 'train'
df.loc[train_end:val_end, 'split'] = 'val'
print(f"  Step 7 ✓  Dataset split:")
for split, count in df['split'].value_counts().items():
    bar = '█' * int(count / n * 25)
    print(f"            {split:<6} {count:>6,}  {bar}")

# Step 8: Save outputs
full_path  = os.path.join(OUT_DIR, 'fashion_metadata_clean.csv')
train_path = os.path.join(OUT_DIR, 'train.csv')
val_path   = os.path.join(OUT_DIR, 'val.csv')
vocab_path = os.path.join(OUT_DIR, 'attribute_vocabulary.json')

df.to_csv(full_path, index=False)
df[df['split']=='train'].to_csv(train_path, index=False)
df[df['split']=='val'].to_csv(val_path, index=False)

vocab = {}
for col in ['masterCategory','subCategory','articleType','baseColour','gender','season','usage']:
    vocab[col] = sorted(df[col].unique().tolist())
with open(vocab_path, 'w') as f:
    json.dump(vocab, f, indent=2)

print(f"\n  Step 8 ✓  fashion_metadata_clean.csv  ({len(df):,} rows)")
print(f"          ✓  train.csv  ({df['split'].value_counts()['train']:,} rows)")
print(f"          ✓  val.csv    ({df['split'].value_counts()['val']:,} rows)")
print(f"          ✓  attribute_vocabulary.json")

# Sample prompts
print(f"\n  Sample Prompt Templates (ready for Week 2 SDXL):")
print("─" * 60)
for p in df['prompt_template'].sample(5, random_state=1).values:
    print(f"  → {p}")

print(f"\n{'='*60}")
print(f"  ✓  Week 1 Complete!")
print(f"  ✓  {df['articleType'].nunique()} article types  |  {df['baseColour'].nunique()} colours")
print(f"  ✓  All files saved to: {OUT_DIR}")
print(f"{'='*60}")
