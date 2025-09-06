# modules/retriever.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import Optional

from modules.embedding_store import load_faiss, INDEX_DIR, USE_PINECONE, query_pinecone, PINECONE_INDEX_NAME

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROMPT = """You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not contained in the context, reply exactly: Sorry it is not present in knowledge base, use google to get answer of general query.

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT)

def get_llm():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in environment.")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )
    return llm

def answer_query(query: str, top_k: int = 4, index_path: str = str(INDEX_DIR / "faiss_index"), conversation_id: Optional[str] = None) -> str:
    """
    Returns generated answer or fallback message. If Pinecone is enabled, queries Pinecone using the conversation_id as namespace.
    """
    # Use Pinecone if configured
    if USE_PINECONE:
        if not conversation_id:
            # If user forgot to provide conversation id, we cannot scope the search - return helpful error
            return "Sorry it is not present in knowledge base, use google to get answer of general query"
        results = query_pinecone(query, top_k=top_k, index_name=PINECONE_INDEX_NAME, namespace=conversation_id)
    else:
        # fallback to local FAISS index
        try:
            db = load_faiss(index_path=index_path)
        except FileNotFoundError:
            return "Sorry it is not present in knowledge base, use google to get answer of general query"
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)

    # simple heuristic: if no results or results too short -> fallback message
    if not results or len(" ".join([d.page_content for d in results]).strip()) < 50:
        return "Sorry it is not present in knowledge base, use google to get answer of general query"

    context = "\n\n---\n\n".join([d.page_content for d in results])

    llm = get_llm()
    rag_chain = prompt | llm
    resp = rag_chain.invoke({"context": context, "question": query}).content

    if "sorry" in resp.lower() and "knowledge base" in resp.lower():
        return "Sorry it is not present in knowledge base, use google to get answer of general query"

    return resp
