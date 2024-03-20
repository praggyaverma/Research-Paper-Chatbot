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

st.title("Research Paper Chatbot")
st.write("Chat with any research paper uploaded in PDF format.")
api_key = st.text_input("Enter your OpenAI API key", type = "password")
client = OpenAI(api_key = api_key)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    with open("pdfs/uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF file uploaded successfully.")
else:
    st.warning("Please upload a PDF file.")

pdf = uploaded_file


if (api_key == ""):
    pass
elif st.button("Research this paper!"):
    pdf = os.listdir("pdfs")[0]
    pdf = f"pdfs/{pdf}"
    with st.spinner("Embedding document... This may take a while"):
        loader = UnstructuredPDFLoader(f"{pdf}")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectorstore = FAISS.from_texts(texts=[t.page_content for t in texts], embedding=embeddings)
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your query"):
        docs = vectorstore.similarity_search(query, 4)
        system_content = f"""
        You are a helpful assistant performing Retrieval-augmented generation (RAG).
        You will be given a user query and some text. 
        Analyse the text and answer the user query. 
        If the answer is not present within the text, say that the given question cannot be answered, 
        dont make up stuff on your own
        """
        user_content = f"""
            Question: {prompt}
            Do not return the question in the response, please. 
            ======
            Supporting texts:
            1. {docs[0].page_content}
            2. {docs[1].page_content}
            3. {docs[2].page_content}
            4. {docs[3].page_content}
            ======
            """
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                
                messages = [{"role": "system", "content": f"{system_content}"},
                {"role": "assistant", "content": f"{user_content}"}],
                
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})