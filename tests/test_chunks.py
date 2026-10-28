from main import chunk_text

sample_text = (
    "Machine learning learns patterns from data. "
    * 100
)

chunks = chunk_text(sample_text)

print("Total characters:", len(sample_text))
print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk[:100])