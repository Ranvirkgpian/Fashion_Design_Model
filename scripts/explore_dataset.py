import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────
ROOT     = r"E:\Fashion model"
CSV      = os.path.join(ROOT, r"data\deepfashion\raw\styles.csv")
IMG_DIR  = os.path.join(ROOT, r"data\deepfashion\raw\images")
OUT_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────
df = pd.read_csv(CSV, on_bad_lines='skip')
df['id'] = df['id'].astype(str)

print("=" * 60)
print("  Fashion Dataset — Exploration Report")
print("=" * 60)
print(f"\n  Total items     : {len(df):,}")
print(f"  Total columns   : {len(df.columns)}")
print(f"  Missing values  :\n")
print(df.isnull().sum().to_string())

# ── Chart 1: Master Category distribution ─────────────
print("\n\n→ Generating Chart 1: Category Distribution...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Fashion Dataset — Exploratory Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Master category
cat_counts = df['masterCategory'].value_counts()
colors1 = ['#7F77DD','#1D9E75','#D85A30','#D4537E','#C9A84C','#378ADD']
axes[0,0].bar(cat_counts.index, cat_counts.values, 
              color=colors1[:len(cat_counts)], edgecolor='white', linewidth=0.5)
axes[0,0].set_title('Master Categories', fontweight='bold', fontsize=13)
axes[0,0].set_xlabel('Category')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=20)
for i, v in enumerate(cat_counts.values):
    axes[0,0].text(i, v + 100, f'{v:,}', ha='center', fontsize=9, fontweight='bold')

# ── Chart 2: Top 15 article types ─────────────────────
print("→ Generating Chart 2: Article Types...")
art_counts = df['articleType'].value_counts().head(15)
axes[0,1].barh(art_counts.index[::-1], art_counts.values[::-1],
               color='#7F77DD', edgecolor='white', linewidth=0.5)
axes[0,1].set_title('Top 15 Article Types', fontweight='bold', fontsize=13)
axes[0,1].set_xlabel('Count')
for i, v in enumerate(art_counts.values[::-1]):
    axes[0,1].text(v + 20, i, str(v), va='center', fontsize=8)

# ── Chart 3: Gender split ──────────────────────────────
print("→ Generating Chart 3: Gender Distribution...")
gen_counts = df['gender'].value_counts()
colors3 = ['#378ADD','#D4537E','#1D9E75','#C9A84C','#D85A30']
wedges, texts, autotexts = axes[1,0].pie(
    gen_counts.values, labels=gen_counts.index,
    autopct='%1.1f%%', colors=colors3,
    startangle=90, pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
axes[1,0].set_title('Gender Distribution', fontweight='bold', fontsize=13)

# ── Chart 4: Top 12 colors ─────────────────────────────
print("→ Generating Chart 4: Base Colours...")
col_counts = df['baseColour'].value_counts().head(12)
color_map = {
    'Navy Blue':'#1A237E','Blue':'#1565C0','Black':'#212121',
    'Grey':'#9E9E9E','White':'#F5F5F5','Red':'#C62828',
    'Green':'#2E7D32','Purple':'#6A1B9A','Pink':'#E91E63',
    'Brown':'#4E342E','Yellow':'#F9A825','Orange':'#E65100',
    'Beige':'#D7CCC8','Maroon':'#880E4F','Off White':'#FAFAFA',
    'Teal':'#00695C','Copper':'#BF360C','Silver':'#757575',
    'Gold':'#F57F17','Coffee Brown':'#3E2723',
}
bar_colors = [color_map.get(c, '#7F77DD') for c in col_counts.index]
bars = axes[1,1].bar(col_counts.index, col_counts.values,
                      color=bar_colors, edgecolor='#cccccc', linewidth=0.5)
axes[1,1].set_title('Top 12 Base Colours', fontweight='bold', fontsize=13)
axes[1,1].set_xlabel('Colour')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
chart1_path = os.path.join(OUT_DIR, 'chart1_distribution.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {chart1_path}")

# ── Chart 5: Season & Usage heatmap ───────────────────
print("→ Generating Chart 5: Season × Usage heatmap...")
pivot = df.groupby(['season','usage']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
ax.set_yticklabels(pivot.index)
ax.set_title('Season × Usage Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(im, ax=ax, label='Item Count')
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if val > 0:
            ax.text(j, i, f'{val:,}', ha='center', va='center',
                    fontsize=8, color='black' if val < pivot.values.max()*0.6 else 'white')
plt.tight_layout()
chart2_path = os.path.join(OUT_DIR, 'chart2_season_usage.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {chart2_path}")

# ── Chart 6: Sample image grid ─────────────────────────
print("→ Generating Chart 6: Sample image grid...")
sample = df.sample(12, random_state=42).reset_index(drop=True)
fig, axes = plt.subplots(3, 4, figsize=(14, 11))
fig.suptitle('Sample Fashion Items — Dataset Preview', 
             fontsize=15, fontweight='bold', y=1.01)

for idx, (ax, (_, row)) in enumerate(zip(axes.flat, sample.iterrows())):
    img_path = os.path.join(IMG_DIR, f"{row['id']}.jpg")
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        title = f"{row['articleType']}\n{row['baseColour']} · {row['gender']}"
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    else:
        ax.text(0.5, 0.5, 'Image\nNot Found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        ax.set_facecolor('#f5f5f5')
    ax.axis('off')

plt.tight_layout()
chart3_path = os.path.join(OUT_DIR, 'chart3_sample_images.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {chart3_path}")

# ── Summary stats ──────────────────────────────────────
print("\n" + "=" * 60)
print("  Dataset Summary")
print("=" * 60)
print(f"  Total items       : {len(df):,}")
print(f"  Master categories : {df['masterCategory'].nunique()}")
print(f"  Article types     : {df['articleType'].nunique()}")
print(f"  Colours           : {df['baseColour'].nunique()}")
print(f"  Genders           : {df['gender'].unique().tolist()}")
print(f"  Seasons           : {df['season'].unique().tolist()}")
print(f"  Year range        : {int(df['year'].min())} – {int(df['year'].max())}")
print(f"\n  Output charts saved to: {OUT_DIR}")
print("=" * 60)
print("\n  ✓  Exploration complete! Open outputs/ to see your charts.")
print("=" * 60)