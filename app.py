# app.py
import os
import uuid
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")

from summarizer import generate_document_summary, extract_last_date
from models import SummaryResponse, LegalDocSummary, LastDateResponse
from utils import extract_text_from_pdf
from modules.chunking import file_to_chunks
from modules.embedding_store import (
    build_faiss_from_chunks,
    INDEX_DIR,
    USE_PINECONE,
    upsert_chunks_to_pinecone,
    PINECONE_INDEX_NAME,
    load_faiss
)
from modules.retriever import answer_query
from modules.chatbot import init_chat, add_user_message, add_bot_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_session: Dict[str, dict] = {}

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str

# --- Cache FAISS index ---
cached_faiss = None
if not USE_PINECONE:
    try:
        cached_faiss = load_faiss(index_path=str(INDEX_DIR))
    except FileNotFoundError:
        cached_faiss = None

@app.post("/upload-and-build/")
async def upload_and_build_db(file: UploadFile = File(...)):
    global db_session
    
    filepath = Path("data") / file.filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = file_to_chunks(str(filepath))
        if not chunks:
            raise HTTPException(status_code=400, detail="The document is empty or could not be processed.")
        
        metadatas = [{"source": file.filename, "chunk_id": str(uuid.uuid4())} for _ in range(len(chunks))]

        conversation_id = str(uuid.uuid4())

        if USE_PINECONE:
            upsert_chunks_to_pinecone(chunks, metadatas=metadatas, index_name=PINECONE_INDEX_NAME, namespace=conversation_id)
        else:
            db = build_faiss_from_chunks(chunks, metadatas=metadatas, index_path=str(INDEX_DIR))
            global cached_faiss
            cached_faiss = db  # update cache
            db_session[conversation_id] = {
                "faiss_db": db,
                "chat_history": init_chat()
            }

        if USE_PINECONE:
            db_session[conversation_id] = {
                "chat_history": init_chat()
            }

        return {
            "message": f"Successfully processed {len(chunks)} chunks and built the vector DB.",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    finally:
        os.remove(filepath)

@app.post("/chat/")
async def chat_with_docs(request: ChatRequest):
    conversation_id = request.conversation_id
    query = request.query

    if not conversation_id:
        raise HTTPException(status_code=400, detail="Missing conversation ID.")

    if conversation_id not in db_session:
        db_session[conversation_id] = {"chat_history": init_chat()}

    session_data = db_session[conversation_id]
    add_user_message(session_data["chat_history"], query)

    try:
        if USE_PINECONE:
            answer = answer_query(
                query,
                index_path=str(INDEX_DIR),
                conversation_id=conversation_id
            )
        else:
            global cached_faiss
            if cached_faiss:
                retriever = cached_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                results = retriever.get_relevant_documents(query)
                context = "\n\n---\n\n".join([d.page_content for d in results]) if results else "No specific document content available."
                language_hint = "Respond in the same language as the question if possible."
                full_context = f"{context}\n\n{language_hint}"
                from modules.retriever import get_llm, prompt
                llm = get_llm()
                rag_chain = prompt | llm
                resp = rag_chain.invoke({"context": full_context, "question": query}).content
                answer = resp
            else:
                answer = "Document index not available."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    add_bot_message(session_data["chat_history"], answer)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "chat_history": session_data["chat_history"]
    }

# --- Summarize and extract date endpoints remain unchanged ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
