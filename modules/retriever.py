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

PROMPT = """You are an experienced and responsible lawyer. Use ONLY the provided context to answer the question.
If the answer can be derived from the context, provide it clearly and precisely.
If the context is insufficient but the question is general, give advice as a lawyer would, based on the idea of the documents.
Do not hallucinate or make up answers if you don't have specific knowledge or action to advise.
If the user asks in another language, respond in that language; otherwise respond in English.

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
        temperature=0.2,  # slight creativity for advice, but not hallucinations
        google_api_key=GOOGLE_API_KEY
    )
    return llm

def answer_query(query: str, top_k: int = 4, index_path: str = str(INDEX_DIR / "faiss_index"), conversation_id: Optional[str] = None) -> str:
    """
    Returns generated answer using context if available, or advice as a lawyer.
    Avoids generic fallback messages and responds intelligently.
    """
    # Retrieve relevant documents
    results = []

    if USE_PINECONE:
        if conversation_id:
            results = query_pinecone(query, top_k=top_k, index_name=PINECONE_INDEX_NAME, namespace=conversation_id)
    else:
        try:
            db = load_faiss(index_path=index_path)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            results = retriever.get_relevant_documents(query)
        except FileNotFoundError:
            results = []

    # Prepare context if available
    context = "\n\n---\n\n".join([d.page_content for d in results]) if results else "No specific document content available."

    # Get the language preference by inspecting the query (basic heuristic)
    language_hint = "Respond in the same language as the question if possible."

    full_context = f"{context}\n\n{language_hint}"

    llm = get_llm()
    rag_chain = prompt | llm

    resp = rag_chain.invoke({"context": full_context, "question": query}).content

    return resp
