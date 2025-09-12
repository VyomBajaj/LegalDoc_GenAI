# modules/retriever.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import Optional

# UPDATED: Import only Pinecone-related items
from modules.embedding_store import USE_PINECONE, query_pinecone, PINECONE_INDEX_NAME

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROMPT = """You are an experienced lawyer. 
Give only the direct answer in 2–3 sentences maximum. 
Do not use phrases like "Based on the context" or "As a lawyer". 
Extract only the essential points from the context. 
If the context is missing, give short, practical legal advice. 
Always reply in the same language as the question, if user explicitly mentions language, then give response in that language.


Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT)

# --- Cache LLM instance ---
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        _llm_instance = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
    return _llm_instance

def answer_query(query: str, top_k: int = 4, index_path: str = None, conversation_id: Optional[str] = None) -> str:
    results = []

    if USE_PINECONE:
        if conversation_id:
            results = query_pinecone(query, top_k=top_k, index_name=PINECONE_INDEX_NAME, namespace=conversation_id)
        else:
            raise ValueError("Pinecone query requires a conversation_id.")
    else:
        # Since FAISS is removed, this block is an error.
        raise RuntimeError("FAISS is no longer supported.")

    context = "\n\n---\n\n".join([d.page_content for d in results]) if results else "No specific document content available."

    language_hint = "Respond in the same language as the question if possible."

    full_context = f"{context}\n\n{language_hint}"

    llm = get_llm()
    rag_chain = prompt | llm

    resp = rag_chain.invoke({"context": full_context, "question": query}).content

    return resp