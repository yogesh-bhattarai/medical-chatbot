import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
# from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings= SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

data_path = os.path.join(os.path.dirname(__file__), "Data")


loader= DirectoryLoader(data_path, glob='*.pdf',show_progress=True,loader_cls=PyPDFLoader)
documents= loader.load()

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
texts= text_splitter.split_documents(documents)
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")