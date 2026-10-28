
from services.preprocessing import chunk_text
from services.embeddings import generate_embeddings
from services.vector_store import store_chunks


def ingest_document(
    text: str,
    document_id: str,
    filename: str,
) -> int:

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "No text chunks were generated from the document."
        )

    embeddings = generate_embeddings(chunks)

    stored_count = store_chunks(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
    )

    return stored_count
