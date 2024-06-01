import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import datetime
import tempfile
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Access your API key
api_key = st.secrets["OPENAI_API_KEY"]  # Just to verify that it loads correctly; remove or replace in production code
def load_db(file_inputs, chain_type, k, llm_name):
    qas = []
    for file_input in file_inputs:
        # Check if file_input is not None and is a file-like object
        if file_input is not None:
            # Use a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                # Write the content of the file-like object to the temporary file
                tmp_file.write(file_input.read())
                file_path = tmp_file.name
        else:
            raise ValueError("No file uploaded")
        # Determine the loader based on file extension
        if file_input.type == 'application/pdf':
            loader = PyPDFLoader(file_path)
        elif file_input.type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        # Proceed with loading and processing the document
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
        qas.append(qa)
    return qas
class ChatBot:
    def __init__(self, default_files=None):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_files = default_files
        self.llm_name = "gpt-3.5-turbo-0301" if datetime.datetime.now().date() < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"
        self.qas = []  # Initialize as empty list
        if self.loaded_files:
            self.load_db(self.loaded_files)  # Ensure this is a valid call
    def load_db(self, file_inputs):
        if file_inputs:
            self.qas = load_db(file_inputs, "stuff", 4, self.llm_name)
            for file_input in file_inputs:
                st.markdown(f"Loaded File: {file_input.name}")
            self.clear_history()
        else:
            st.error("No files provided. Please upload files.")
            self.qas = []  # Ensure qas is empty if no files are loaded
    def convchain(self, query):
        if not self.qas:
            st.error("No QA models loaded. Please upload files and load the models.")
            return
        if query:
            for qa in self.qas:
                result = qa({"question": query, "chat_history": self.chat_history})
                self.chat_history.append((query, result["answer"]))
                self.db_query = result["generated_question"]
                self.db_response = result["source_documents"]
                self.answer = result['answer']
            self.display_history()
    def display_history(self):
        # Clear previous display
        st.empty()
        # Display all history
        if len(self.chat_history) > 0:  # Check if chat history is not empty
            q, a = self.chat_history[-1]  # Get the latest question and answer
            st.write("User: ", q)
            st.write("ChatBot: ", a)
    def clear_history(self):
        self.chat_history = []
# Initialize the chatbot
cb = ChatBot()
# File uploader for multiple files
file_inputs = st.file_uploader("Upload Files", type=['pdf', 'doc', 'docx'], accept_multiple_files=True)
# Automatically load the files once they are uploaded
if file_inputs is not None:
    cb.load_db(file_inputs)
# Text input for the question
inp = st.text_input("Enter your question:")
# Process the question automatically when it changes
if inp:
    cb.convchain(inp)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "ASKY"
