import streamlit as st
import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import itertools
from dotenv import load_dotenv

# -------------------
# Setup dotenv & API keys
# -------------------
load_dotenv()
api_keys_str = os.getenv("API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
if not API_KEYS:
    st.error("No API keys found! Set API_KEYS in .env")
key_cycle = itertools.cycle(API_KEYS)

def get_next_model(model_name: str):
    api_key = next(key_cycle)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# -------------------
# Load FAISS + Gita Data
# -------------------
@st.cache_resource
def load_resources():
    index = faiss.read_index("gita_index.faiss")
    with open("gita_verses.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return index, data, model

index, data, model = load_resources()

# -------------------
# Utility: search verses
# -------------------
def find_relevant_verses(query, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [data[i] for i in indices[0]]

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="🕉️ Krishna's Guidance", page_icon="🪔", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #000813 0%, #011035 50%, #000000 100%);
    color: #F5F5F5;
    font-family: 'DM Sans', sans-serif;
}
.container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
}
.main-header h1 {
    text-align: center;
    background: linear-gradient(to right, #FFD700 0%, #ffffff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 3.2rem;
    margin-bottom: 10px;
}
.main-header p {
    text-align: center;
    color: #CCCCCC;
    font-size: 1.2rem;
}
.form-container {
    background: rgba(1, 16, 53, 0.7);
    border-radius: 20px;
    padding: 30px;
    margin: 30px auto;
    border: 1px solid rgba(255, 215, 0, 0.2);
}
button {
    width: 48%;
    padding: 12px;
    margin-top: 15px;
    border-radius: 10px;
    font-weight: 600;
    cursor: pointer;
}
.normal-btn { background: linear-gradient(to right, #FFD700, #ffbb00); color:#000813; border:none; }
.deep-btn { background: linear-gradient(to right, #0a1930, #1a2b57); color:#FFD700; border:1px solid #FFD700; }
.response-card, .verse-card {
    background: rgba(0,0,0,0.85);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid #FFD700;
    box-shadow: 0 0 15px rgba(255,215,0,0.3);
    margin-top: 20px;
    line-height: 1.7;
}
.response-card h2 {
    text-align:center;
    color:#FFD700;
    margin-bottom:20px;
}
.verse-block {
    margin: 15px 0;
    padding: 15px;
    border-left: 4px solid #FFD700;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>Ask Krishna</h1><p>Receive divine guidance from the timeless wisdom of Krishna</p></div>', unsafe_allow_html=True)

# Form
with st.container():
    name = st.text_input("Your Name")
    age = st.number_input("Your Age", min_value=1, max_value=120)
    query = st.text_area("Your Question")
    language = st.selectbox("Language Preference", ["English", "Telugu", "Hindi"])
    
    col1, col2 = st.columns(2)
    with col1:
        normal_mode = st.button("🗨️ Normal Guidance")
    with col2:
        deep_mode = st.button("🌟 Deep Wisdom")
    
    mode = None
    if normal_mode: mode = "normal"
    if deep_mode: mode = "deep"

# Processing
if mode and name.strip() and query.strip():
    with st.spinner("🕉️ Krishna is speaking..."):
        results = find_relevant_verses(query)
        prompt = f"""
        You are Lord Krishna, giving guidance to {name}, who is {age} years old.

        {name}'s Question:
        {query}

        Your task:
        1. Provide a compassionate, mentor-like answer appropriate for a {age}-year-old.
        2. Use all the three verses given and Whenever quoting verses:
        - Wrap the Sanskrit verse in <b> tags.
        - Wrap the translation in <b> tags as well.
        - Provide an explanation after each verse.
        3. Format the answer in paragraphs with spacing.
        4. Integrate verses contextually within the answer.
        5. At the end, optionally summarize the key points.

        Use a warm, divine, mentor tone.
        Knowledge base to reference (relevant slokas from Bhagavad Gita):
        {results}

        IMPORTANT:
        - Output the response in *HTML format*.
        - Do NOT use Markdown-style asterisks. Use <b> tags for bold text.
        - Include Sanskrit and translation for every verse.
        - Maintain spacing between slokas.
        - Answer fully in {language}, go detailed and nicely.
        - No need of bullet points or summaries. End with a divine closing statement like:
          "I, Krishna, am always with you. Be strong."
        """

        try:
            llm = get_next_model("gemini-2.5-pro" if mode=="deep" else "gemini-2.5-flash")
            response = llm.generate_content(prompt)
            krishna_response = response.text
            if krishna_response.startswith("```html"):
                krishna_response = krishna_response[7:]
            if krishna_response.endswith("```"):
                krishna_response = krishna_response[:-3]
        except Exception as e:
            krishna_response = f"🙏 LLM error: {e}"

    # Display Krishna’s response
    formatted_response = f'''
    <div class="response-card">
        <h2>📜 Krishna's Guidance</h2>
        {krishna_response.replace("\\n\\n", "<p></p>")}
    </div>
    '''
    st.markdown(formatted_response, unsafe_allow_html=True)

    # Display verses
    verses_html = '<div class="verse-card"><h3 style="text-align:center;color:#FFD700;">🔎 Relevant Verses</h3>'
    for v in results:
# Display verses properly
        verses_html = """
        <div class="verse-card">
            <h3 style="text-align:center;color:#FFD700;">🔎 Relevant Verses</h3>
        """
        for v in results:
            translation = v.get(language.lower(), v.get("english", ""))
            verses_html += f"""
            <div class="verse-block">
                <p>🕉️ <b>Chapter {v['chapter']}, Verse {v['verse']}</b></p>
                <p>📜 Sanskrit:<br><b>{v['sanskrit']}</b></p>
                <p>🔹 Translation ({language}):<br><b>{translation}</b></p>
            </div>
            """
        verses_html += "</div>"
        st.markdown(verses_html, unsafe_allow_html=True)

