from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
url = "http://localhost:6333"

client= QdrantClient(
    url= url,
    prefer_grpc=False,
)
print(client)
print("Qdrant Client Successfully Created!")

db= Qdrant(client= client,embeddings= embeddings,collection_name= "vector_db")

print(db)
print("Qdrant DB Successfully Created!")

query= "what is the cause of cancer?"

docs= db.similarity_search(query=query,k=1)

for doc in docs:
    print({
        "page": doc.page_content,
        "metadata": doc.metadata
    })


