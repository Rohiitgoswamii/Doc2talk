
from pathlib import Path

import chromadb


DB_PATH = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "talk2doc_chunks"


def get_collection():
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def store_chunks(
    document_id: str,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:

    if not chunks:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Each chunk must have exactly one embedding."
        )

    collection = get_collection()

    ids = [
        f"{document_id}_{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": index,
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)

def search_chunks(
    query_embedding: list[float],
    document_id: str,
    n_results: int = 3,
    ) -> dict:
    collection = get_collection()

    if collection.count() == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )

    return results

def delete_document(document_id: str) -> None:
    collection = get_collection()

    collection.delete(
        where={"document_id": document_id}
    )