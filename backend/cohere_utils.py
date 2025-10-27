"""
Compatibility module: provides the same function names your project expected (cohere_generate etc.)
but routes calls to a local Ollama instance (HTTP API) or the ollama python package if available.

Requirements:
 - Ollama running locally (default: http://localhost:11434)
 - Python packages: requests (and optional 'ollama' client)

This module exposes:
 - ollama_chat / ollama_generate
 - argument_mining(text), summarization(text), strategy_suggestions(text), risk_prediction(text), future_steps(text)
 - cohere_generate(prompt, task)  <-- compatibility wrapper used by existing code
"""

import os
import requests
from typing import Optional, Dict, Any, List

# Prefer using the official ollama python client if installed.
try:
    import ollama
    _HAS_OLLAMA_PY = True
except Exception:
    _HAS_OLLAMA_PY = False

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2") 


# ----------------- Low-level Ollama calls -----------------

def _call_ollama_chat(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """Call Ollama's /api/chat endpoint or ollama python client."""
    if _HAS_OLLAMA_PY:
        try:
            resp = ollama.chat(model, messages=messages)
            if isinstance(resp, dict) and "message" in resp:
                return resp["message"].get("content", "")
        except Exception as e:
            print("Ollama python client failed, falling back to HTTP:", e)

    url = f"{OLLAMA_URL}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        choices = data.get("choices")
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "")
        if "message" in data:
            return data["message"].get("content", "")
    return str(data)


def _call_ollama_generate(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Call Ollama's /api/generate endpoint or ollama python client."""
    if _HAS_OLLAMA_PY:
        try:
            resp = ollama.generate(model, prompt=prompt, stream=False, options={"temperature": temperature})
            if isinstance(resp, dict) and "response" in resp:
                return resp["response"]
        except Exception as e:
            print("Ollama python client failed, falling back to HTTP:", e)

    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "response" in data:
        return data["response"]
    if "text" in data:
        return data["text"]
    if "output" in data:
        return data["output"]

    return str(data)


# ----------------- High-level wrappers -----------------

def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 512, temperature: float = 0.2) -> str:
    return _call_ollama_generate(prompt, model=model, max_tokens=max_tokens, temperature=temperature)


def ollama_chat_system(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return _call_ollama_chat(messages=messages, model=model)


# ----------------- Task-specific functions -----------------

def argument_mining(text: str) -> str:
    prompt = f"""You are an assistant that mines legal arguments from a case file. 
Extract key arguments, supporting facts, parties, dates, and applicable legal issues. 
Return a concise structured summary with bullet points and short labels.

Case file:
{text}

Output:"""
    return ollama_generate(prompt, max_tokens=600)


def summarization(text: str) -> str:
    prompt = f"""You are an expert legal summarizer. 
Produce a clear, concise executive summary (150-300 words) of the following case file. 
Focus on facts, claims, procedural posture, and outcome if present.

Case file:
{text}

Summary:"""
    return ollama_generate(prompt, max_tokens=350)


def strategy_suggestions(text: str) -> str:
    prompt = f"""You are a litigation strategy assistant. 
Based on the following case, suggest defense and plaintiff strategies, potential motions, key witnesses, 
and evidence to obtain. Provide prioritized steps (short, medium, long-term).

Case file:
{text}

Suggestions:"""
    return ollama_generate(prompt, max_tokens=600)


def risk_prediction(text: str) -> str:
    prompt = f"""You are an assistant that predicts legal risks and likely outcomes. 
Given the following case file, identify high-risk claims, likely defenses, potential damages, 
and probability estimates (low/medium/high) with short explanation for each.

Case file:
{text}

Risk assessment:"""
    return ollama_generate(prompt, max_tokens=400)


def future_steps(text: str) -> str:
    prompt = f"""You are an action-planning assistant for legal cases. 
Given the case file below, list the next 10 practical steps (each 1-2 sentences) for counsel to take, 
prioritized and numbered.

Case file:
{text}

Next steps:"""
    return ollama_generate(prompt, max_tokens=400)


# ----------------- Compatibility wrapper -----------------

def cohere_generate(prompt: str, task: Optional[str] = None) -> str:
    """Compatibility wrapper that routes old cohere_generate calls to Ollama."""
    if task:
        t = task.lower()
        if "summar" in t:
            return summarization(prompt)
        if "arg" in t:
            return argument_mining(prompt)
        if "strategy" in t or "suggest" in t:
            return strategy_suggestions(prompt)
        if "risk" in t:
            return risk_prediction(prompt)
        if "future" in t or "step" in t:
            return future_steps(prompt)

    return ollama_generate(prompt)


# Add these imports at the top of your cohere_utils.py file
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- RAG IMPLEMENTATION ---

# Define the path to the persistent vector database
PERSIST_DIRECTORY = 'db'

def ask_question_with_rag(question: str) -> str:
    """
    Answers a question by retrieving relevant context from the vector database.
    """
    try:
        # 1. Initialize models
        # This is the same embedding model used in ingest.py
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        llm = Ollama(model="llama3.2") # Make sure this matches your Ollama model

        # 2. Load the existing vector database
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        # 3. Create a retriever
        # This component is responsible for fetching relevant documents
        retriever = db.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 most relevant chunks

        # 4. Create a prompt template
        # This tells the LLM how to use the retrieved context
        template = """
        You are a helpful legal assistant. Answer the following question based only on the provided context.
        If the context does not contain the answer, state that you cannot answer the question.
        Provide citations from the source document if possible.

        Context:
        {context}

        Question:
        {input}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. Create the RAG chain
        # This chain ties together the retriever, prompt, and LLM
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # 6. Invoke the chain and get the response
        response = retrieval_chain.invoke({"input": question})
        
        return response.get("answer", "No answer could be generated.")

    except Exception as e:
        return f"An error occurred during the RAG process: {e}"