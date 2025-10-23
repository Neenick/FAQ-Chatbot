import os
import random
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever


REFUSAL_MESSAGES = [
    "Sorry, I cannot answer your question. Please ask a different question or give us a call.",
    "I'm sorry, I couldn't find the information you need. Please ask a different question or call our support line.",
    "I don't have enough knowledge to answer that question. Could you please try asking in a different way?",
    "Apologies, but I cannot answer your question based on my current knowledge. Please ask me about another topic."
]

@st.cache_resource # To cache the result
def load_and_index_data():
    """
    Loads documents, splits them, creates embeddings, and builds the FAISS index.
    This function runs only once due to Streamlit's cache mechanism.
    """
    # Load environment variables
    load_dotenv()
    
    # 1. Data Loading and Splitting
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50) 
    
    # Ensure the 'docs' folder exists
    doc_path = "docs"
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The directory '{doc_path}' was not found. Please create it and add your .txt files.")
    
    for filename in os.listdir(doc_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(doc_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = text_splitter.split_text(text)
                    docs.extend(chunks)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if not docs:
        raise ValueError("No text chunks were created. Check if your .txt files have content.")

    # 2. Embedding and Indexing
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)
    
    return vectorstore

# 2. Main Retrieval Function
def get_answer(vectorstore, question, chat_history):
    """
    Sets up the RAG chain and executes the query.
    """
    # 1. Setup Retriever
    retriever = vectorstore.as_retriever()

    # 2. Setup LLM
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
    )

    # 3. Define Prompt and Chains
    prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    You are a helpful, friendly assistant for customer support about PixelPad.

    You have two modes of response:
    1. **Knowledge Mode** — When the user's question is about the product or something factual,
       answer **only** using the provided context or chat history. 
       If the answer cannot be found there, respond with:
       '""" + random.choice(REFUSAL_MESSAGES) + """'

    2. **Conversational Mode** — If the user says something conversational (like greetings,
       self-introductions such as "My name is Leon", or small talk like "How are you?"),
       respond politely and naturally as a human would, but do NOT provide factual or external information.

    Context:
    {context}
    """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # 4. Invoke and Return
    return retrieval_chain.stream({"input": question,
                                   "chat_history": chat_history})
