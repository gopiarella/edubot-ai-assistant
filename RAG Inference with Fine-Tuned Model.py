# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:36:09 2025

@author: gopia
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load fine-tuned T5 model and tokenizer
model_dir = "/content/t5_ncert_final"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_answer_with_t5(question, context):
    input_text = f"explanation: {context} question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question_rag(question, top_k=10):
    relevant_chunks = search_similar_chunks(question, top_k=top_k)
    context = "\n".join([chunk['content'] for chunk in relevant_chunks])
    return generate_answer_with_t5(question, context)

# Example query
question = "What is time?"
answer = answer_question_rag(question)
print("🧠 Answer:", answer)
