
from services.embeddings import generate_embeddings
from services.vector_store import search_chunks


def main():
    document_id = input(
        "Enter the document_id returned by upload: "
    ).strip()

    question = input(
        "Ask a question about your document: "
    ).strip()

    if not document_id or not question:
        raise ValueError(
            "Both document_id and question are required."
        )

    # Convert the question into an embedding.
    query_embedding = generate_embeddings([question])[0]

    # Retrieve the three most relevant chunks.
    results = search_chunks(
        query_embedding=query_embedding,
        document_id=document_id,
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print(
            "No chunks found. Check the document_id "
            "and confirm ingestion succeeded."
        )
        return

    for index, chunk in enumerate(documents):
        print(f"\n--- Result {index + 1} ---")
        print(f"Filename: {metadatas[index]['filename']}")
        print(f"Chunk index: {metadatas[index]['chunk_index']}")
        print(f"Distance: {distances[index]:.4f}")
        print(f"Text:\n{chunk}")


if __name__ == "__main__":
    main()