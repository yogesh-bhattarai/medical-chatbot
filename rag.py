from langchain_core.prompts import PromptTemplate
from langchain_community.llms import ctransformers
from langchain.chains import retrieval_qa
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request,Form,Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json

app= FastAPI()
templates=  Jinja2Templates(directory= "templates")

app.mount("/static", StaticFiles(directory="static"),name="static")
