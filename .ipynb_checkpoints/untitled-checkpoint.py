import sys
sys.path.insert(0, r"E:\Programe Files\PythonPackages")

import os, torch, warnings, json, random, gc
warnings.filterwarnings("ignore")
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"]  = r"E:\Programe Files\huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
print(f"CUDA available : {torch.cuda.is_available()}")
print(f"GPU            : {torch.cuda.get_device_name(0)}")
print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("Setup OK!")