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

#app.mount("/static", StaticFiles(directory="static"),name="static")

# config= {
#     "max_new_token":1024,
#     "context_lenght":2048,
#     "repetition_penalty":1.1,
#     "temperature": 0.1,
#     "top_k":50,
#     "top_p":0.9,
#     "stream": True,
#     "threads": int(os.cpu_count()/2) 
# }
# model= AutoModelForCausalLM.from_pretrained(
#     "mradermacher/Meditron3-Gemma2-2B-i1-GGUF",
#     model_type="llama",
#     lib="avx2",
# )
from transformers import AutoModelForCausalLM,AutoTokenizer
model_name= "openai-community/gpt2"
tokenizer= AutoTokenizer.from_pretrained(model_name,trust_remote_code= True)
model= AutoModelForCausalLM.from_pretrained(model_name)

print("LLM Initialized....")

prompt_template= """
Use the following pieces of information to answer  the user's question 
If you donot know the answer, just say that  you don't know don't try to make up an answer.

Context:{context}
Question:{question}

only return the helpful answer below  and nothing else
Helpful answer:

"""

embeddings= SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url= "http://localhost:6333"
client= QdrantClient(
    url=url,
     prefer_grpc=False)
db= Qdrant(client=client,embeddings=embeddings,collection_name="vector_db")
prompt= PromptTemplate(template=prompt_template,input_variables=["context","question"])
retrieval= db.as_retriever(search_kwargs= {"k":1})

@app.get("/",response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

#@app.post("/get_response")

 


