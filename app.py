import streamlit as st  
import time
import os
import faiss 
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import requests
import threading 
from difflib import SequenceMatcher
import queue
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline(model="facebook/bart-large-mnli")

classifier = load_classifier()


# --- Page Configuration ---
st.set_page_config(page_title="EduBot - Study Assistant", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
/* Sticky Header */
.header { 
  position: fixed;
  top: 50px;
  left: 50px;
  background-color: #0d6efd;
  color: white;
  width: 100%;
  text-align: center;
  padding: 15px 30px;
  font-size: 28px;
  font-weight: bold;
  z-index: 10000;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.spacer {
  height: 80px;
}

/* Sticky Welcome Message */
.welcome-box {
  position: sticky;
  top: 80px;
  z-index: 900;
  background-color: #f0f4f8;
  padding: 18px 25px;
  border-left: 6px solid #4b6cb7;
  margin-bottom: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  font-size: 17px;
  color: #333333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.07);
}

/* Chat container with scroll */
.chat-container {
  max-height: 70vh;
  overflow-y: auto;
  padding-right: 10px;
}

/* User message styling */
.message.user {
  background: #d1e7ff;
  color: #054a91;
  padding: 14px 20px;
  border-radius: 20px 20px 0 20px;
  margin-bottom: 12px;
  max-width: 70%;
  align-self: flex-end;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  white-space: pre-wrap;
  box-shadow: 0 1px 4px rgba(0, 74, 145, 0.3);
}

/* Bot message styling: remove background, keep text color and spacing */
.message.bot {
  background: transparent !important;
  color: #1a1a1a;
  padding: 18px 0;
  border-radius: 0;
  margin-bottom: 12px;
  max-width: 75%;
  align-self: flex-start;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  white-space: pre-wrap;
  box-shadow: none !important;
}

/* Chat flex container */
.chat-flex {
  display: flex;
  flex-direction: column;
}

/* Sidebar title */
.sidebar-title {
  font-size: 22px;
  color: #2563EB;
  margin-bottom: 18px;
  font-weight: 700;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  text-align: center;
  padding-top: 20px;
}

/* Sidebar fixed bottom button */
.sidebar-bottom-button {
  position: fixed;
  bottom: 100px;
  left: 50px;
  width: 300px;  /* Adjust width to sidebar width */
  background-color: #dc3545;
  color: white;
  padding: 10px 16px;
  border-radius: 8px;
  font-weight: bold;
  text-align: center;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
  border: none;
}
.sidebar-bottom-button:hover {
  background-color: #b02a37;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
section[data-testid="stSidebar"] button[kind="secondary"] {
    width: 100% !important;
    background-color: #dc3545 !important;
    color: white !important;
    font-weight: bold !important;
    padding: 12px !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)
# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">📘 EduBot Dashboard</div>', unsafe_allow_html=True)

    class_subject_map = {
        'Class_6': ['maths', 'science', 'social'],
        'Class_7': ['maths', 'science', 'social'],
        'Class_8': ['physical science', 'social', 'maths', 'bio science'],
        'Class_9': ['maths', 'science', 'social'],
        'Class_10': ['maths', 'science', 'social'],
        'Class_11': ['accountancy', 'history', 'economics', 'chemistry', 'maths', 'physics', 'biology', 'political science'],
        'Class_12': ['physics', 'history', 'chemistry', 'economics', 'maths part 1', 'maths part 2', 'political science', 'biology']
    }

    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.markdown("### Student Login")
        roll_number = st.text_input("Roll Number")
        name = st.text_input("Name")
        selected_class = st.selectbox("Select Class", sorted(class_subject_map.keys()))
        subjects_for_class = class_subject_map.get(selected_class, [])
        selected_subject = st.selectbox("Select Subject", subjects_for_class)

        if st.button("Log In"):
            if roll_number.strip() and name.strip():
                st.session_state['logged_in'] = True
                st.session_state['roll_number'] = roll_number
                st.session_state['name'] = name
                st.session_state['class'] = selected_class
                st.session_state['subject'] = selected_subject
                st.success(f"Welcome {name} from {selected_class}, Subject: {selected_subject}")
                st.rerun()
            else:
                st.error("Please fill in all the login details.")
    else:
        st.markdown(f"""
        <div style="
            font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            color: #222;
            padding: 15px 10px 80px 10px;
            line-height: 1.5;
            min-height: 150px;
            box-sizing: border-box;
        ">
            <p style="margin: 8px 0; font-weight: 600;">Logged in as: {st.session_state.get('name')}</p>
            <p style="margin: 8px 0;"><strong>Roll no:</strong> {st.session_state.get('roll_number')}</p>
            <p style="margin: 8px 0;"><strong>Class:</strong> {st.session_state.get('class')}</p>
            <p style="margin: 8px 0;"><strong>Subject:</strong> {st.session_state.get('subject')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Log Out button fixed at bottom inside sidebar
        logout_button_style = """
            position: fixed;
            bottom: 30px;
            left: 15px;
            width: calc(100% - 30px);
            background-color: #dc3545;
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-weight: 700;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            text-align: center;
        """

        if st.button("🔒 Log Out", key="sidebar_logout"):
            for key in ['logged_in', 'roll_number', 'name', 'class', 'subject', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.markdown(f"""
        <style>
        button[kind="secondary"][data-testid="stSidebar"] {{
            {logout_button_style}
        }}
        </style>
        """, unsafe_allow_html=True)


# --- Header ---
st.markdown('<div class="header">📚 EDU Bot - Educational Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# --- Login Check ---
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.info("Please log in from the sidebar to start using EduBot.")
    st.stop()

# --- Load Resources ---
@st.cache_resource(show_spinner=False)
def load_resources():
    index = faiss.read_index("embeddings/index.faiss")
    with open("metadata/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embedder = SentenceTransformer('all-MiniLM-L12-v2')
    return index, metadata, embedder

index, metadata, embedder = load_resources()


# Add this helper function near the top of your script
def normalize_query_text(query: str) -> str:
    # Lowercase and remove special characters
    query = query.lower()
    query = re.sub(r'[^a-z0-9\s]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()

    # Replace 'class 10' with 'class_10' for consistency
    query = re.sub(r'class\s*(\d+)', r'class_\1', query)

    return query

 
def normalize_embeddings(embeddings):
    # Normalize using FAISS's built-in function
    faiss.normalize_L2(embeddings)
    return embeddings
def extract_chapter_number(query: str):
    match = re.search(r'\bchapter\s*(\d+)', query.lower())
    if match:
        return match.group(1)
    return None


def extract_chapter_info(query: str):
    """
    Extract chapter number and chapter name keywords from query.
    Returns (chapter_number: str or None, chapter_name: str or None)
    """
    chapter_number = None
    chapter_name = None

    # Extract chapter number if present, e.g. "chapter 3"
    match = re.search(r'\bchapter\s*(\d+)', query.lower())
    if match:
        chapter_number = match.group(1)

    # Extract chapter name after "chapter" or "explain"
    chapter_name_match = re.search(r'(?:chapter\s*\d*\s*|explain\s+)([\w\s]+)', query.lower())
    if chapter_name_match:
        chapter_name = chapter_name_match.group(1).strip()
        # Limit to first 3 words to avoid overly long names
        chapter_name = " ".join(chapter_name.split()[:3])

    # If chapter_name is purely numeric, discard it (since it's a number, not a name)
    if chapter_name and chapter_name.isdigit():
        chapter_name = None

    return chapter_number, chapter_name



@st.cache_data(show_spinner=False, max_entries=100)
def cached_search(query, top_k=3):
    normalized_query = normalize_query_text(query)
    query_embedding = embedder.encode([normalized_query], convert_to_numpy=True)
    query_embedding = normalize_embeddings(query_embedding)

    _, indices = index.search(query_embedding, top_k * 5)  # Overfetch for better filtering

    filtered_results = []

    selected_class = st.session_state.get("class", "").strip().lower().replace("class_", "")
    selected_subject = normalize_subject(st.session_state.get("subject", ""))

    top_label, top_score, is_off_syllabus = classify_syllabus(query)
    normalized_top_label = normalize_subject(top_label)
    if top_score > 0.9 and normalized_top_label != selected_subject:
        print(f"[INFO] Overriding session subject '{selected_subject}' → '{normalized_top_label}'")
        selected_subject = normalized_top_label

    chapter_number, chapter_name = extract_chapter_info(query)

    print(f"[DEBUG] Starting document filtering for class='{selected_class}', subject='{selected_subject}'")
    print(f"[DEBUG] Looking for chapter number: {chapter_number}, name: {chapter_name}")

    for idx in indices[0]:
        if idx == -1:
            continue

        meta = metadata[idx]
        meta_data = meta.get("metadata", {})

        meta_class = meta_data.get("class", meta.get("class", "")).strip().lower().replace("class_", "")
        meta_subject = normalize_subject(meta_data.get("subject", meta.get("subject", "")))
        meta_chapter_number = str(meta_data.get("chapter_number", "")).strip()
        meta_chapter_title = meta_data.get("chapter_title", "").lower()

        print(f"[DEBUG] Checking doc idx={idx}, class='{meta_class}', subject='{meta_subject}', chapter='{meta_chapter_title}'")

        # Filter by class and subject
        if meta_class != selected_class or meta_subject != selected_subject:
            print(f"[SKIP] Class/subject mismatch. Wanted class='{selected_class}', subject='{selected_subject}'")
            continue

        # --- Updated Chapter Filtering Logic ---
        if chapter_number:
            if meta_chapter_number != chapter_number:
                print(f"[SKIP] Chapter number mismatch: found {meta_chapter_number}, expected {chapter_number}")
                continue
            if chapter_name and not is_similar(chapter_name, meta_chapter_title):
                print(f"[INFO] Chapter number matched but name '{chapter_name}' not similar to '{meta_chapter_title}' – proceeding anyway.")
                # Proceed anyway if number matches
        else:
            if chapter_name and not is_similar(chapter_name, meta_chapter_title):
                print(f"[SKIP] Chapter name '{chapter_name}' not similar to '{meta_chapter_title}'")
                continue

        if not is_formula_only(meta.get("text", "")):
            filtered_results.append(meta)

        if len(filtered_results) >= top_k:
            break

    print(f"[DEBUG] Filtered {len(filtered_results)} document(s) out of {len(indices[0])}")
    return filtered_results


# --- Utility Functions ---
def normalize_subject(subject):
    subject = subject.strip().lower()
    replacements = {
        "maths part 1": "maths",
        "maths part 2": "maths",
        "bio science": "biology",
        "physical science": "science"
    }
    return replacements.get(subject, subject) 

def is_similar(a: str, b: str, threshold=0.6) -> bool:
    """
    Returns True if similarity ratio between a and b exceeds threshold.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def classify_syllabus(query):
    selected_class = st.session_state.get("class", "")
    valid_subjects = class_subject_map.get(selected_class, [])

    # Normalize subjects and add "off topic"
    normalized_labels = sorted(set(normalize_subject(sub) for sub in valid_subjects))
    candidate_labels = normalized_labels + ["off topic"]

    result = classifier(query, candidate_labels=candidate_labels)
    top_label = result["labels"][0].lower()
    top_score = result["scores"][0]

    is_off_syllabus = top_label == "off topic"
    return top_label, top_score, is_off_syllabus

def is_formula_only(text):
    return sum(char.isalpha() for char in text) < 0.4 * len(text)

def fix_formula_text(text):
    text = text.replace("–", "-")
    text = re.sub(r"\bvx\s*=\s*vox\b", "Vₓ = V₀ₓ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvy\s*=\s*voy\s*-\s*g\s*t\b", "Vᵧ = V₀ᵧ - g × t", text, flags=re.IGNORECASE)
    return text.replace("=", " = ")

def check_ollama_model_ready(model="phi4-mini"): 
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": "ping", "stream":False},
            timeout=10
        )
        return res.status_code == 200
    except Exception as e:
        print(f"Error checking Ollama model readiness: {e}")
        return False

    

def ask_ollama_stream(query, context, model="phi4-mini"):
    prompt = f"""
You are an NCERT-based study assistant. Your task is to answer the student's question clearly and thoroughly using the study material below.

**Instructions:**

- Use the study material provided to support your answer.
- If the material contains relevant information but is incomplete, you may supplement with your general knowledge to provide a full and clear response.
- Include **reasoning**, detailed explanations, and relevant **formulas** using proper mathematical notation (LaTeX style).
- Structure your answer with **headings**, **bold** key terms, and write in a clear, engaging style like GPT.
- Never refuse to answer if some material is available.
- Use only the material and your general knowledge. Do not fabricate unrelated information.

---

**Study Material:**

{context.strip()}

---

**Student Question:**

{query.strip()}

---

**Answer:**
"""



   
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    print("🔵 Raw stream line:", decoded_line)  # Debug print
                    try:
                        data = json.loads(decoded_line)
                        chunk = data.get("response", "")
                        print("🔷 Parsed chunk:", chunk)  # Debug print
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        print("⚠️ JSON decode error for line:", decoded_line)
                        continue
    except Exception as e:
        yield f"❌ Error contacting Ollama model: {e}"


# --- Dynamic follow-up message generator based on user query ---

def generate_followup_message(user_question: str) -> str:
    q = user_question.lower()
    suggestions = []

    if any(keyword in q for keyword in ["cache", "caching", "cache_data", "cache_resource"]):
        suggestions.append("I can help you optimize caching strategies for faster load times.")
    if any(keyword in q for keyword in ["thread", "threading", "background", "concurrent", "async"]):
        suggestions.append("I can assist with concurrency improvements like threading or async.")
    if any(keyword in q for keyword in ["rerender", "rerun", "performance", "ui slowdown"]):
        suggestions.append("We can look into minimizing UI rerenders for smoother interactions.")
    if any(keyword in q for keyword in ["faiss", "vector search", "embedding"]):
        suggestions.append("I can guide you on speeding up FAISS search or embedding optimizations.")

    if suggestions:
        message = "💡 " + " ".join(suggestions)
    else:
        # Professional and friendly fallback message
        message = (
            "💡 If you'd like more detailed insights or additional assistance on this topic, "
            "please type **'Yes'** and I'll be happy to provide further help."
        )

    return message


# --- Initialize messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Welcome to **EduBot** – Your Study Assistant!"
    })

# --- Welcome Box ---
st.markdown(f'<div class="welcome-box">{st.session_state.messages[0]["content"]}</div>', unsafe_allow_html=True)

# --- Chat Container ---
def display_chat():
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container chat-flex">', unsafe_allow_html=True)
        messages_to_show = st.session_state.messages[-20:]  # Limit to last 20 messages
        for msg in messages_to_show:
            role_class = "user" if msg["role"] == "user" else "bot"
            content_html = msg["content"].replace("\n", "<br>")
            st.markdown(f'<div class="message {role_class}">{content_html}</div>', unsafe_allow_html=True)

display_chat()

# --- Input and streaming logic ---
# --- Chat Input ---
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = False

query = st.chat_input("Ask your question from Syllabus...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.user_query = query
    st.session_state.trigger_query = True  # Set trigger for query

# --- Query Execution ---
if st.session_state.get("trigger_query") and st.session_state.get("user_query"):

    user_query = st.session_state.user_query
    st.session_state.trigger_query = False  # Reset after handling

    # 🔍 Step 0: Check if question is syllabus-relevant
    top_label, top_score, is_off_syllabus = classify_syllabus(user_query)
    st.info(f"📌 Detected Subject: **{top_label.capitalize()}**, Confidence: {top_score:.2f}")

    if is_off_syllabus:
        st.warning("🚫 This question appears to be off-topic or outside your selected class and subject.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❗ Your question doesn't match your class syllabus. Please try asking something from your selected subject."
        })
        st.stop()

    # 🔎 Step 1: Get FAISS documents filtered by class and subject
    docs = cached_search(user_query, top_k=3)

    if not docs:
        st.warning("⚠️ No study material found matching your class and subject for this query.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❗ Your question doesn't match any content in the study material for your class and subject. Please rephrase or try another."
        })
        st.stop()

    # ✅ Syllabus-Valid: Continue
    with st.spinner("Thinking... please wait while I prepare the answer."):

        if not check_ollama_model_ready():
            st.error("Ollama model is not running or reachable. Please start it and try again.")
            st.stop()

        context_text = "\n\n".join([doc.get("text", "") for doc in docs])

        # Optional: Show document info
        st.markdown("### 📄 Fetched Documents:")
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            doc_class = meta.get("class", "N/A")
            doc_subject = meta.get("subject", "N/A")
            doc_chapter = meta.get("chapter_title", "N/A")
            st.markdown(f"- **Document {i}:** Class: {doc_class}, Subject: {doc_subject}, Chapter: {doc_chapter}")

        # 🔄 Step 2: Stream response from LLM
        response_placeholder = st.empty()
        full_response = ""

        for chunk in ask_ollama_stream(user_query, context_text):
            full_response += chunk
            response_placeholder.markdown(f"**EduBot:** {full_response}")

        cleaned_response = fix_formula_text(full_response)
        st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
        st.session_state.messages.append({
            "role": "assistant",
            "content": "✅ This answer was generated strictly using your class syllabus material."
        })

