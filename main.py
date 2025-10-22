from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

docs = []
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir("docs"):
    if filename.endswith(".txt"):
        with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = text_splitter.split_text(text)
            docs.extend(chunks)

load_dotenv() # Loads API key from .env
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Initialize the embeddings (using the API key already loaded)
embeddings = OpenAIEmbeddings()

# Create the FAISS Vector Store from your document chunks
vectorstore = FAISS.from_texts(
    texts=docs, 
    embedding=embeddings
)
print("FAISS Index created and stored in 'vectorstore' variable.")

# Convert the vector store into a retriever object
retriever = vectorstore.as_retriever()
print("Vector Store converted to a Retriever.")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# Define the Prompt Template (System Prompt)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following context to answer the user's question:\n\n{context}"),
    ("user", "{input}"),
])

# Create the document combining chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the full retrieval chain (Retriever + Document Chain)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Ask for input
input_question = input("How can I help you?\n\n")

# Run a query against your indexed documents
response = retrieval_chain.invoke({"input": input_question})

# Print the final answer
print("\n" + response["answer"])