import streamlit as st

from rag_engine import load_and_index_data, get_answer

## Streamlit UI (Running the App)

st.title("Document Q&A Chatbot")

# Load the data using the cached function
# This runs once on app startup.
try:
    vectorstore = load_and_index_data()
except Exception as e:
    st.error(f"Error loading data or embeddings. Check your API key and 'docs' folder. Error: {e}")
    st.stop()


# User Input
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    # Get the answer and display it
    with st.spinner("Searching and generating answer..."):
        answer = get_answer(vectorstore, user_input)
        st.write("### Answer:")
        st.write(answer)