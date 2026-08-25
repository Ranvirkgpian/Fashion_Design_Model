import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")
import os, torch
import chromadb
from transformers import CLIPProcessor, CLIPModel

os.environ["HF_HOME"] = r"E:\Programe Files\huggingface"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
model.eval()

def encode_text(query_text):
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        projected = model.text_projection(text_out.pooler_output)
        projected = projected / projected.norm(dim=-1, keepdim=True)
    return projected.cpu().numpy().tolist()

ROOT = r"E:\Fashion model"
client = chromadb.PersistentClient(path=os.path.join(ROOT, r"data\chromadb"))
collection = client.get_collection("fashion_styles")

text_emb = encode_text("a black tshirt")
results = collection.query(query_embeddings=text_emb, n_results=3)

print("Query: a black tshirt")
print("With cosine distance metric, dist directly relates to 1-similarity:")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [dist={dist:.3f}, similarity={1-dist:.3f}] {doc}")
