import os
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback



# 1. Caching Function (The "Heavy Lifting")
@st.cache_resource
def load_and_index_data():
    """
    Loads documents, splits them, creates embeddings, and builds the FAISS index.
    This function runs only once due to st.cache_resource.
    """
    load_dotenv() 
    
    # Data Loading and Splitting
    docs = []
    # Note: Use RecursiveCharacterTextSplitter for better results usually.
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    
    for filename in os.listdir("docs"):
        if filename.endswith(".txt"):
            with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = text_splitter.split_text(text)
                docs.extend(chunks)

    # Embedding and Indexing
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)
    
    return vectorstore

# 2. Main Retrieval Function
def get_answer(vectorstore, question):
    """
    Sets up the RAG chain and executes the query.
    """
    # 1. Setup Retriever
    retriever = vectorstore.as_retriever()

    # 2. Setup LLM
    llm = ChatOpenAI(model="gpt-3.5-nano", temperature=0)

    # 3. Define Prompt and Chains
    prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     You are a helpful assistant for question-answering tasks.
     Answer the user's question **ONLY** based on the following context.
     Do not use any external or general knowledge.
     
     **If the answer is not present in the provided context, you must respond with:**
     'Sorry, I cannot answer your question. Please ask a different question or give us a call.'
     
     Context:
     {context}
     """
    ),
    ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 4. Invoke and Return
    response = retrieval_chain.invoke({"input": question})
    
    # The following commented code can check the costs per prompt in the terminal
    # with get_openai_callback() as cb:
    #    response = retrieval_chain.invoke({"input": question})
    #    print(cb) # Uncomment to see cost in console
    return response["answer"]



