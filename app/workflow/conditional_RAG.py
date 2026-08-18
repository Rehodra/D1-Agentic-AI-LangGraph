import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph , START , END 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path : str ):

    #load the pdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    #split pdf into docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    #create chunks
    chunks = text_splitter.split_documents(documents)
    # storing into vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    ## Retrieve the most semantically similar chunks for a user query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

academic_retriever = build_retriver("app/workflow/academic_paper.pdf")
fees_retriever = build_retriver("app/workflow/fees.pdf")