# save as verify_setup.py and run it

print("Testing imports...")

import numpy as np
print("✅ numpy")

import faiss
print("✅ faiss")

from sentence_transformers import SentenceTransformer
print("✅ sentence-transformers")

import openai
print("✅ openai")

from pydantic import BaseModel
print("✅ pydantic")

import fastapi
print("✅ fastapi")

import streamlit
print("✅ streamlit")

import sklearn
print("✅ scikit-learn")

import plotly
print("✅ plotly")

import jinja2
print("✅ jinja2")

import tenacity
print("✅ tenacity")

import datasets
print("✅ datasets")

# Test FAISS actually works on M1
index = faiss.IndexFlatL2(128)
vectors = np.random.random((10, 128)).astype('float32')
index.add(vectors)
query = np.random.random((1, 128)).astype('float32')
distances, indices = index.search(query, 3)
print("✅ FAISS search working correctly")

# Test Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(["test sentence"])
print(f"✅ Sentence Transformers working - embedding shape: {embedding.shape}")

print("\n🎉 All dependencies working correctly on M1!")