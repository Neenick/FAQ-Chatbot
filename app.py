import streamlit as st

from rag_engine import load_and_index_data, get_answer

## Streamlit UI (Running the App)

# --- Helper Function for Stream Processing ---

# This generator function extracts only the 'answer' text from the LangChain stream
# and yields it back to st.write_stream.
def stream_rag_response(stream):
    """Yields text chunks from the LangChain stream."""
    for chunk in stream:
        # LangChain stream yields dictionaries. We only want the text from the 'answer' key.
        if "answer" in chunk:
            yield chunk["answer"]

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Document Chatbot (RAG)")
st.title("📚 PixelPad FAQ Chatbot")
st.caption("Answers are restricted to the contents of the files in the 'docs/' folder.")


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
        
        # 1. Get the raw stream generator from the RAG engine
        raw_stream = get_answer(vectorstore, prompt)
        
        # 2. Process the stream using the helper function
        text_stream = stream_rag_response(raw_stream)
        
        # 3. Use st.write_stream to display the text incrementally.
        # It automatically returns the final accumulated string.
        full_response = st.write_stream(text_stream)

    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})