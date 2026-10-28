
from pathlib import Path
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Create persistent ChromaDB storage
db_path = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(
    path=str(db_path)
)


# 3. Create or open a collection
collection = client.get_or_create_collection(
    name="talk2doc_test",
    metadata={"hnsw:space": "cosine"},
)


# 4. Prepare sample document chunks
chunks = [
    "Machine learning learns patterns from data.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks.",
]


# 5. Generate embeddings
embeddings = model.encode(chunks)


# 6. Store chunks, vectors, and metadata
document_id = str(uuid4())

collection.add(
    ids=[
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "document_id": document_id,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ],
)


print("Chunks stored:", len(chunks))
print("Total records in database:", collection.count())


# 7. Search for relevant chunks
question = "What is deep learning?"

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    include=["documents", "metadatas", "distances"],
)


# 8. Display search results
for i, chunk in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}")
    print("Text:", chunk)
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])
