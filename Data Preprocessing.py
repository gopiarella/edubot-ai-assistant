# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:30:59 2025

@author: gopia
"""

import re
import json

def clean_extracted_text(text):
    """
    Clean raw text extracted from PDFs by removing unwanted characters,
    page numbers, normalizing spacing, and fixing common OCR issues.
    """
    if not text:
        return ""
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Fix line breaks within paragraphs
    text = re.sub(r'\n{2,}', '\n\n', text)        # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)           # Remove excessive spaces/tabs
    text = re.sub(r'\bPage\s*[-–]?\s*\d+\b', '', text, flags=re.IGNORECASE)  # Remove page numbers
    text = re.sub(r'\b\d+\b', lambda m: '' if int(m.group()) < 100 else m.group(), text)  # Remove small numbers
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Normalize punctuation spacing
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
    text = text.strip()
    return text

def preprocess_all_chapters(data):
    for item in data:
        item["text"] = clean_extracted_text(item["text"])
    return data

# Clean data and save
cleaned_data = preprocess_all_chapters(data)
with open("/content/drive/MyDrive/edubot_cleaned_text.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
print("✅ Cleaned text saved.")
