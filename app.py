
"""
Streamlit frontend for Lawyers Assistant (uses local Ollama via backend module)
Run: streamlit run app.py
"""
import streamlit as st
from backend import cohere_utils as llm
import requests
import os

st.set_page_config(page_title="Lawyers Assistant", layout="wide")

st.title("Lawyers Assistant")
st.write("Upload a legal case file (text or pdf) and choose an action. This runs locally and uses Ollama (you must have Ollama running).")

uploaded = st.file_uploader("Upload a case file (txt or pdf)", type=["txt","pdf"])
text_content = ""

if uploaded is not None:
    fname = uploaded.name
    if fname.lower().endswith(".txt"):
        text_content = uploaded.read().decode("utf-8")
    else:
        try:
            # Try PyPDF2 if available
            from io import BytesIO
            from PyPDF2 import PdfReader
            reader = PdfReader(BytesIO(uploaded.read()))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text())
            text_content = "\n\n".join(pages)
        except Exception as e:
            st.error("Could not parse PDF. Install PyPDF2 or upload a TXT file. Error: " + str(e))

if text_content:
    st.subheader("File content (preview)")
    st.text_area("Content preview", value=text_content[:5000], height=300)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                out = llm.summarization(text_content)
                st.subheader("Summary")
                st.text_area("Summary", value=out, height=300)
    with col2:
        if st.button("Argument mining"):
            with st.spinner("Mining arguments..."):
                out = llm.argument_mining(text_content)
                st.subheader("Arguments")
                st.text_area("Arguments", value=out, height=300)
    with col3:
        if st.button("Strategy suggestions"):
            with st.spinner("Generating suggestions..."):
                out = llm.strategy_suggestions(text_content)
                st.subheader("Strategy Suggestions")
                st.text_area("Strategy", value=out, height=300)

    col4, col5 = st.columns(2)
    with col4:
        if st.button("Risk prediction"):
            with st.spinner("Assessing risk..."):
                out = llm.risk_prediction(text_content)
                st.subheader("Risk Prediction")
                st.text_area("Risk", value=out, height=300)
    with col5:
        if st.button("Future steps"):
            with st.spinner("Listing next steps..."):
                out = llm.future_steps(text_content)
                st.subheader("Next Steps")
                st.text_area("Next Steps", value=out, height=300)
# --- RAG Q&A Section ---
    st.markdown("---") # Add a horizontal line for separation
    st.subheader("Ask a Question About Your Legal Corpus")

    user_question = st.text_input("Enter your question:")

    if st.button("Ask with RAG"):
        if user_question:
            with st.spinner("Searching knowledge base and generating answer..."):
                answer = llm.ask_question_with_rag(user_question)
                st.success("Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")

else:
    st.info("Upload a file to get started.")

