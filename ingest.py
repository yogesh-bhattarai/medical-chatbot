import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings= SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
path='./Data'
loader= DirectoryLoader(path ,glob='*.pdf',show_progress=True,loader_cls=PyPDFLoader)
documents= loader.load()

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
texts= text_splitter.split_documents(documents)
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
index_name= "medical-chatbot"
docsearch = PineconeVectorStore.from_documents(
    documents=texts,
    index_name='medical-chatbot',
    embedding=embeddings,
)