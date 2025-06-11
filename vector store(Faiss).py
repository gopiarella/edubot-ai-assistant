# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:32:57 2025

@author: gopia
"""

import faiss
import numpy as np
import json

embedding_matrix = np.array(embeddings).astype("float32")
index = faiss.IndexFlatIP(embedding_matrix.shape[1])  # Inner product = cosine similarity (vectors normalized)
index.add(embedding_matrix)

# Save FAISS index and metadata
faiss.write_index(index, "/content/drive/MyDrive/edubot_faiss_cosine.index")

with open("/content/drive/MyDrive/edubot_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Saved FAISS index (cosine similarity) and metadata.")
