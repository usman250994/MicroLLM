import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load .env file
load_dotenv()

# Read key
api_key = os.getenv("OPENAI_API_KEY")

# Wrap strings as Documents
documents = [
    Document(page_content="FastAPI is a Python web framework for building APIs quickly."),
    Document(page_content="Python is a programming language used for web, ML, and scripting.")
]

# Create embeddings + vector DB
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma.from_documents(documents, embeddings)

# Turn into retriever
retriever = vectordb.as_retriever()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

# Setup Retrieval QA pipeline
qa = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Run a query
answer = qa.run("What is FastAPI?")
print(answer)
