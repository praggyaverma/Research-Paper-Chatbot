import streamlit as st
import os
import subprocess
import platform
import openai
from openai import OpenAI
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil 

try:
    os.mkdir("pdfs")
    os.mkdir("embeddings")
except:
    pass

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with open("pdfs/uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF file uploaded successfully.")
else:
    st.warning("Please upload a PDF file.")

