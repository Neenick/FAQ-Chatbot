import streamlit as st

from rag_engine import load_and_index_data, get_answer

## Streamlit UI (Running the App)


st.title("PixelPad FAQ Chatbot")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Write the first question form the chatbot
    st.session_state.messages.append({"role": "assistant", "content": "Hey! I am your virtual PixelPad assistant. How can I help you?"})


# Display existing messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Load the data using the cached function
# This runs once on app startup.
try:
    vectorstore = load_and_index_data()
except Exception as e:
    st.error(f"Error loading data or embeddings. Check the API key and 'docs' folder. Error: {e}")
    st.stop()


# Accept user input using the chat interface
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # 1. Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI response using the RAG function
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            
            # The vectorstore from the cached function
            vectorstore = load_and_index_data() 
            
            # The RAG chain logic from the `get_answer`
            response_text = get_answer(vectorstore, prompt) 
            
            st.markdown(response_text)

    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})