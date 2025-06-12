# edubot-ai-assistant
AI-powered study assistant for Class 6-12 students using NCERT content
# EduBot – AI-Powered Study Assistant

EduBot is an intelligent AI chatbot designed to assist school students (Class 6–12) by providing syllabus-aligned answers and real-time explanations based on NCERT curriculum. It leverages Retrieval-Augmented Generation (RAG) with local large language models to ensure responses are accurate and constrained within the syllabus.

---

## Features

- ✅ Subject-aligned question answering  
- ✅ Syllabus-filtered responses (Class 6–12, NCERT-based)  
- ✅ Context-aware follow-up question handling  
- ✅ Real-time AI explanations with LaTeX formula support  
- ✅ Personal login for students with class and subject selection  
- ✅ Uses Retrieval-Augmented Generation (RAG) for accurate and relevant answers  

---

## Tech Stack

- **Frontend & Backend:** Streamlit  
- **Vector Search:** FAISS  
- **Embeddings:** Sentence Transformers  
- **Question Classification:** BART NLI  
- **Local LLMs:** Ollama + phi4-mini  
- **RAG:** Retrieval-Augmented Generation to fetch syllabus-aligned context  

---

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/gopiarella/edubot-ai-assistant.git
   cd edubot-ai-assistant

