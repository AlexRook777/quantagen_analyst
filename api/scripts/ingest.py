from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "../../data/chroma_data")

def ingest_10k(ticker: str, pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata = {"ticker": ticker}
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    vector_db.persist()
    print(f"Ingested {len(chunks)} chunks for {ticker} into ChromaDB.")

if __name__ == "__main__":
    ingest_10k("AAPL", "./AAPL_10K.pdf")
