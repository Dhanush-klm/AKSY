import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import datetime
import tempfile


# Access your API key
api_key = st.secrets["OPENAI_API_KEY"]  # Just to verify that it loads correctly; remove or replace in production code


def load_db(file_input, chain_type, k, llm_name):
    # Check if file_input is not None and is a file-like object
    if file_input is not None:
        # Use a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the content of the file-like object to the temporary file
            tmp_file.write(file_input.read())
            file_path = tmp_file.name
    else:
        raise ValueError("No file uploaded")

    # Proceed with loading and processing the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

class ChatBot:
    def __init__(self, default_file=None):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_file = default_file
        self.llm_name = "gpt-3.5-turbo-0301" if datetime.datetime.now().date() < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"
        self.qa = None  # Initialize as None
        if self.loaded_file:
            self.load_db(self.loaded_file)  # Ensure this is a valid call

    def load_db(self, file_input):
        if file_input:
            self.qa = load_db(file_input, "stuff", 4, self.llm_name)
            st.markdown(f"Loaded File: {file_input.name}")
            self.clear_history()
        else:
            st.error("No file provided. Please upload a file.")
            self.qa = None  # Ensure qa is None if no file is loaded

    def convchain(self, query):
        if self.qa is None:
            st.error("No QA model loaded. Please upload a file and load the model.")
            return
        if query:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.append((query, result["answer"]))
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result['answer']
            for q, a in self.chat_history:
                st.write("User: ", q)
                st.write("ChatBot: ", a)

    def clear_history(self):
        self.chat_history = []
# Streamlit UI

# Assuming the rest of your necessary imports and ChatBot class definition are here

# Initialize the chatbot
cb = ChatBot()

# File uploader
file_input = st.file_uploader("Upload File", type=['pdf'])

# Automatically load the file once it's uploaded
if file_input is not None:
    cb.load_db(file_input)

# Text input for the question
inp = st.text_input("Enter your question:")

# Process the question automatically when it changes
if inp:
    cb.convchain(inp)
