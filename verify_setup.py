print("Testing all imports...\n")

try:
    import numpy as np
    print("✅ numpy:", np.__version__)
except Exception as e:
    print("❌ numpy:", e)

try:
    import faiss
    print("✅ faiss: installed")
except Exception as e:
    print("❌ faiss:", e)

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers: installed")
except Exception as e:
    print("❌ sentence-transformers:", e)

try:
    import openai
    print("✅ openai:", openai.__version__)
except Exception as e:
    print("❌ openai:", e)

try:
    import pydantic
    print("✅ pydantic:", pydantic.__version__)
except Exception as e:
    print("❌ pydantic:", e)

try:
    import fastapi
    print("✅ fastapi:", fastapi.__version__)
except Exception as e:
    print("❌ fastapi:", e)

try:
    import streamlit
    print("✅ streamlit:", streamlit.__version__)
except Exception as e:
    print("❌ streamlit:", e)

try:
    import sklearn
    print("✅ scikit-learn:", sklearn.__version__)
except Exception as e:
    print("❌ scikit-learn:", e)

try:
    import plotly
    print("✅ plotly:", plotly.__version__)
except Exception as e:
    print("❌ plotly:", e)

try:
    import jinja2
    print("✅ jinja2:", jinja2.__version__)
except Exception as e:
    print("❌ jinja2:", e)

try:
    import tenacity
    print("✅ tenacity: installed")
except Exception as e:
    print("❌ tenacity:", e)

try:
    import datasets
    print("✅ datasets: installed")
except Exception as e:
    print("❌ datasets:", e)

try:
    import httpx
    print("✅ httpx:", httpx.__version__)
except Exception as e:
    print("❌ httpx:", e)

try:
    import dotenv
    print("✅ python-dotenv: installed")
except Exception as e:
    print("❌ python-dotenv:", e)

print("\n--- Functional Tests ---\n")

try:
    import faiss
    import numpy as np
    index = faiss.IndexFlatL2(128)
    vectors = np.random.random((10, 128)).astype('float32')
    index.add(vectors)
    query = np.random.random((1, 128)).astype('float32')
    distances, indices = index.search(query, 3)
    print("✅ FAISS search working correctly")
except Exception as e:
    print("❌ FAISS functional test failed:", e)

try:
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embedding = model.encode(["test sentence"])
    print(f"✅ Sentence Transformers working on device: {device}")
    print(f"   Embedding shape: {embedding.shape}")
except Exception as e:
    print("❌ Sentence Transformers functional test failed:", e)

print("\n--- .env Check ---\n")

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if key and key != "your_openai_api_key_here":
        print("✅ OPENAI_API_KEY loaded from .env")
    elif key == "your_openai_api_key_here":
        print("⚠️  OPENAI_API_KEY is still placeholder — replace with real key before Phase 4")
    else:
        print("❌ OPENAI_API_KEY not found in .env")
except Exception as e:
    print("❌ .env loading failed:", e)

print("\n--- Git Check ---\n")

import subprocess
try:
    result = subprocess.run(['git', 'remote', '-v'], 
                          capture_output=True, text=True)
    if 'github.com' in result.stdout:
        print("✅ GitHub remote connected")
        print("  ", result.stdout.split('\n')[0])
    else:
        print("❌ GitHub remote not connected")
except Exception as e:
    print("❌ Git check failed:", e)

print("\n✅ Phase 0 verification complete!")
