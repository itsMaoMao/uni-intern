import chromadb
# chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="./")

collection = chroma_client.get_collection(name="test_collection")

# collection.add(
#     documents=["This is a document about engineer", "This is a document about steak"],
#     metadatas=[{"source": "doc1"}, {"source": "doc2"}],
#     ids=["id1", "id2"]
# )

results = collection.query(
    query_texts=["I want to get my cooker."],
    n_results=1
)

print(results)
