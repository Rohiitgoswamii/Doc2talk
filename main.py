
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.preprocessing import extract_document_text
from services.rag_pipeline import ingest_document

import logging

from pydantic import BaseModel, Field
from services.embeddings import generate_embeddings
from services.vector_store import search_chunks
from services.llm import generate_answer


# 1. Application configuration

app = FastAPI(
    title="Talk2Doc",
    description="Upload documents and prepare them for RAG-based question answering.",
    version="0.1.0",
)
class ChatRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=4000)


# 2. CORS configuration

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# 3. Upload configuration

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


# 4. Health endpoint

@app.get("/health")
def health():
    return {"status": "healthy"}


# 5. Homepage endpoint

@app.get("/")
def root():
    return {"message": "Welcome to the Talk2Doc API"}


# 6. Upload and RAG ingestion endpoint

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    # Validate the file type.
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and DOCX files are allowed.",
        )

    try:
        # Read at most 10 MB plus one byte.
        file_bytes = await file.read(MAX_FILE_SIZE + 1)

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB limit.",
            )

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Extract the document's text.
        text = extract_document_text(
            file_bytes=file_bytes,
            extension=extension,
        )

        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was found. "
                    "The document may be empty or scanned."
                ),
            )

        # Give this upload its own unique ID.
        document_id = str(uuid4())

        # Chunk, embed, and store the extracted text.
        chunk_count = ingest_document(
            text=text,
            document_id=document_id,
            filename=filename,
        )

        # Return the ingestion result.
        return {
            "document_id": document_id,
            "filename": filename,
            "file_type": extension.lstrip("."),
            "character_count": len(text),
            "chunk_count": chunk_count,
            "text_preview": text[:1000],
            "message": (
                "Document uploaded, processed, "
                "embedded, and stored successfully."
            ),
        }

    finally:
        await file.close()

@app.post("/chat")
def chat_with_document(request: ChatRequest):
    question = request.question.strip()
    document_id = request.document_id.strip()

    if not question or not document_id:
        raise HTTPException(
            status_code=422,
            detail="Document ID and question cannot be empty.",
        )

    # 1. Convert the question into an embedding.
    query_embedding = generate_embeddings([question])[0]

    # 2. Retrieve the most relevant chunks from this document.
    results = search_chunks(
        query_embedding=query_embedding,
        document_id=document_id,
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No chunks found for this document. Upload it again if necessary.",
        )

    # 3. Build context for Gemini from retrieved chunks only.
    context_parts = []
    sources = []

    for index, chunk in enumerate(documents):
        metadata = metadatas[index]

        context_parts.append(
            f"[Source {index + 1}]\n{chunk}"
        )

        sources.append({
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"],
            "distance": distances[index],
        })

    context = "\n\n".join(context_parts)

    # 4. Ask Gemini to answer the question using this context.
    try:
        answer = generate_answer(
            question=question,
            context=context,
        )
    except Exception as exc:
        logging.exception("Gemini answer generation failed")
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed. Check the Gemini API key, model availability, and API quota.",
        ) from exc

    # 5. Return the answer and its retrieved source references.
    return {
        "document_id": document_id,
        "question": question,
        "answer": answer,
        "sources": sources,
    }
