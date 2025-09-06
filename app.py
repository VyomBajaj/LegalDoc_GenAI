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

# Load environment variables FIRST
load_dotenv()

# Get the API key right after loading the environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the key is None and raise an error early
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")

# Now, import your modules that rely on the key
from summarizer import generate_document_summary, extract_last_date
from models import SummaryResponse, LegalDocSummary, LastDateResponse
from utils import extract_text_from_pdf
from modules.chunking import file_to_chunks
# app.py (imports near the top)
from modules.embedding_store import (
    build_faiss_from_chunks,
    INDEX_DIR,
    USE_PINECONE,
    upsert_chunks_to_pinecone,
    PINECONE_INDEX_NAME
)
from modules.retriever import answer_query

from modules.chatbot import init_chat, add_user_message, add_bot_message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for the application
db_session: Dict[str, dict] = {}

# Initialize FastAPI
app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the chat request body
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str

# --- UPLOAD AND CHUNK Endpoints ---

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
        
        # create metadata list for each chunk (we will store chunk_id and source and index it)
        metadatas = [{"source": file.filename, "chunk_id": str(uuid.uuid4())} for _ in range(len(chunks))]

        conversation_id = str(uuid.uuid4())

        if USE_PINECONE:
            # upsert into Pinecone under namespace = conversation_id
            upsert_chunks_to_pinecone(chunks, metadatas=metadatas, index_name=PINECONE_INDEX_NAME, namespace=conversation_id)
        else:
            # Fallback: build local FAISS index and persist
            db = build_faiss_from_chunks(chunks, metadatas=metadatas, index_path=str(INDEX_DIR))
            # optionally still save something in db_session if you rely on in-memory for immediate queries
            db_session[conversation_id] = {
                "faiss_db": db,
                "chat_history": init_chat()
            }

        # For Pinecone mode we don't store faiss_db in memory; only chat history
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


# --- CHAT Endpoints ---

@app.post("/chat/")
async def chat_with_docs(request: ChatRequest):
    conversation_id = request.conversation_id
    query = request.query

    if not conversation_id:
        raise HTTPException(status_code=400, detail="Missing conversation ID.")

    # If chat history is missing (e.g., after restart), create a fresh one
    if conversation_id not in db_session:
        db_session[conversation_id] = {"chat_history": init_chat()}

    session_data = db_session[conversation_id]
    add_user_message(session_data["chat_history"], query)

    try:
        # 🔑 always query Pinecone using namespace = conversation_id
        answer = answer_query(
            query,
            index_path=str(INDEX_DIR),  # only used for FAISS fallback
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    add_bot_message(session_data["chat_history"], answer)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "chat_history": session_data["chat_history"]
    }



# --- Summarizer Endpoints ---

@app.post("/summarize/", response_model=SummaryResponse)
async def summarize_document(file: UploadFile = File(...), language: str = Form(...)):
    # ... (no change, existing code for summarization) ...
    file_location = f"temp/{file.filename}"
    Path(file_location).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        document_content = extract_text_from_pdf(file_location)
        if not document_content:
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        summary_result: LegalDocSummary = generate_document_summary(document_content, language, GOOGLE_API_KEY)
        
        logger.info("--- Document Summary ---")
        logger.info(summary_result.model_dump_json(indent=2))
        
        return SummaryResponse(
            summary=summary_result,
            is_summarized=False
        )
    except Exception as e:
        logger.error(f"An internal error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


# --- New Endpoint for Date Extraction ---
@app.post("/extract-last-date/", response_model=LastDateResponse)
async def extract_date_from_document(file: UploadFile = File(...)):
    """
    Extracts the last date from a legal document.
    """
    file_location = f"temp/{file.filename}"
    Path(file_location).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        document_content = extract_text_from_pdf(file_location)
        if not document_content:
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        last_date = extract_last_date(document_content, GOOGLE_API_KEY)
        
        return LastDateResponse(last_date=last_date)
    except Exception as e:
        logger.error(f"An error occurred during date extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during date extraction.")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)