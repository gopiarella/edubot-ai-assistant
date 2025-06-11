# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:34:31 2025

@author: gopia
"""

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load model, index, and metadata for retrieval
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("/content/drive/MyDrive/edubot_faiss_cosine.index")
with open("/content/drive/MyDrive/edubot_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def search_similar_chunks(query, top_k=5):
    query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(np.array([query_vector]).astype("float32"), top_k)
    return [metadata[i] for i in I[0]]

def build_prompt(query, chunks):
    context = "\n\n".join([chunk["content"] for chunk in chunks])
    prompt = f"""
Use the following syllabus-aligned content to answer the question.
Context:
{context}
Question:
{query}
Answer strictly using the above context. Do not guess or go outside the syllabus.
"""
    return prompt

# Example usage:
query = "What is the difference between speed and velocity?"
results = search_similar_chunks(query)
for res in results:
    print(f"\n📘 Class {res['class']} - {res['subject']} - {res['chapter']}")
    print(res['content'][:300])  # snippet

rag_prompt = build_prompt(query, results)
print("\n🧠 Prompt for LLM:\n")
print(rag_prompt[:1000])
