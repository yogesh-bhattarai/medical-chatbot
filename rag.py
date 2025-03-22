from langchain_core.prompts import PromptTemplate
from langchain_community.llms import ctransformers
from langchain.chains import retrieval_qa
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request,Form,Response