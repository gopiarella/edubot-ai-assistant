# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:28:20 2025

@author: gopia
"""

import os
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()

def collect_ncert_text_data(base_dir):
    """
    Traverse folder structure and extract text with metadata.
    Expected structure:
    /content/drive/MyDrive/edubot_dataset/Class_X/subject_name/file.pdf
    """
    collected_data = []
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for subject_folder in os.listdir(class_path):
            subject_path = os.path.join(class_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            for chapter_file in os.listdir(subject_path):
                if chapter_file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(subject_path, chapter_file)
                    chapter_name = os.path.splitext(chapter_file)[0]
                    print(f"📘 Reading: {pdf_path}")
                    raw_text = extract_text_from_pdf(pdf_path)
                    if raw_text:
                        collected_data.append({
                            "class": class_folder.replace("Class_", ""),
                            "subject": subject_folder.lower(),
                            "chapter": chapter_name,
                            "text": raw_text
                        })
    return collected_data

# Run the collection
base_dir = "/content/drive/MyDrive/edubot_dataset"
data = collect_ncert_text_data(base_dir)
print(f"\n✅ Extracted {len(data)} chapters.")
print("\nSample:\n", data[0]["class"], data[0]["subject"], data[0]["chapter"])
print(data[0]["text"][:500])
