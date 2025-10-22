from langchain_text_splitters import CharacterTextSplitter
import os

docs = []
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir("docs"):
    if filename.endswith(".txt"):
        with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = text_splitter.split_text(text)
            docs.extend(chunks)

