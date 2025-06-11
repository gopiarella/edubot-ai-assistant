# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:31:41 2025

@author: gopia
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_chapter_data(data, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunked_data = []
    for item in data:
        chunks = splitter.split_text(item["text"])
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "class": item["class"],
                "subject": item["subject"],
                "chapter": item["chapter"],
                "chunk_id": i,
                "content": chunk
            })
    return chunked_data

chunked_data = chunk_chapter_data(cleaned_data)

# Save chunked data
with open("/content/drive/MyDrive/edubot_chunked_data.json", "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=2)
print(f"✅ Chunked {len(chunked_data)} segments.")

# Install dependencies
!pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text_local(text):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

# Generate embeddings
embeddings = []
metadata = []
for doc in chunked_data:
    vector = embed_text_local(doc["content"])
    embeddings.append(vector)
    metadata.append({
        "class": doc["class"],
        "subject": doc["subject"],
        "chapter": doc["chapter"],
        "chunk_id": doc["chunk_id"],
        "content": doc["content"]
    })

# Save embeddings and metadata
embedding_array = np.array(embeddings).astype("float32")
np.save("/content/drive/MyDrive/edubot_embeddings.npy", embedding_array)

with open("/content/drive/MyDrive/edubot_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Embeddings and metadata saved.")
