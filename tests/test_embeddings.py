from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = [
    "Machine learning learns patterns from data.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks."
]

embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

# Display every embedding
for index, (chunk, embedding) in enumerate(
    zip(chunks, embeddings),
    start=1
):
    print(f"\n{'=' * 60}")
    print(f"CHUNK {index}: {chunk}")
    print(f"EMBEDDING DIMENSIONS: {len(embedding)}")
    print("EMBEDDING VALUES:")
    print(embedding.tolist())

