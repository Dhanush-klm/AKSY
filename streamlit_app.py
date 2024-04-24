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

os.environ["OPENAI_API_KEY"] = "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"

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
    return qa

class ChatBot:
    def __init__(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_file = "2024_NCR_GUIDE_NH_VOYIX_HR.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def load_db(self, count):
        if count == 0 or file_input is None:  # init or no file specified :
            st.markdown(f"Loaded File: {self.loaded_file}")
        else:
            self.loaded_file = file_input.name
            self.qa = load_db(self.loaded_file, "stuff", 4)
        self.clear_history()

    def convchain(self, query):
        if not query:
            st.write("User: ", "")
        else:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.extend([(query, result["answer"])])
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result['answer']
            st.write("User: ", query)
            st.write("ChatBot: ", self.answer)

    def clear_history(self, count=0):
        self.chat_history = []

import datetime

current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"

cb = ChatBot()

file_input = st.file_uploader("Upload File", type=['pdf'])
button_load = st.button("Load File")
inp = st.text_input("Enter your question:")

if button_load:
    cb.load_db(1)

if inp:
    cb.convchain(inp)
