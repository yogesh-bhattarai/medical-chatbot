import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Qdrant

embeddings= SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
path='./Data'
loader= DirectoryLoader(path,glob='*.pdf',show_progress=True,loader_cls=PyPDFLoader)
documents= loader.load()

print(len(documents))
