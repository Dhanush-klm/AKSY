import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
import tempfile


OPENAI_API=st.secrets['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"] = OPENAI_API # Just to verify that it loads correctly; remove or replace in production code



def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa,db

class ChatBot:
    def __init__(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_file = None  # Initially no file is loaded
        self.qa = None  # Initially no QA chain is loaded
        self.db = None  # Store the database instance

    def load_db(self, file_input=None):
        if file_input is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_input.getvalue())
                tmp_path = tmp.name

            self.loaded_file = tmp_path
            self.qa, self.db = load_db(self.loaded_file, "stuff", 4)
            st.markdown(f"Loaded File: {self.loaded_file}")
        self.clear_history()

    def add_document(self, file_input):
        if file_input is not None and self.db is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_input.getvalue())
                tmp_path = tmp.name

            # Load additional documents
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            # Update the database with new documents
            self.db.add_documents(docs)
            st.markdown(f"Added document: {file_input.name}")

    def convchain(self, query):
        if not query:
            st.write("User: ", "")
        else:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.extend([(query, result["answer"])])
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result['answer']
            for q, a in self.chat_history:
                st.write("User: ", q)
                st.write("ChatBot: ", a)

    def clear_history(self, count=0):
        self.chat_history = []


import datetime

current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"

cb = ChatBot()

# Streamlit interface handling for file upload
file_input = st.file_uploader("Upload File", type=['pdf'])
button_load = st.button("Load File")
inp = st.text_input("Enter your question:")


if button_load and file_input:
    cb.load_db(file_input)

if inp:
    cb.convchain(inp)
