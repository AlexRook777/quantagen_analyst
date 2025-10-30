from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "../../data/chroma_data")

embeddings = OpenAIEmbeddings()
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
llm = OpenAI(temperature=0)

def answer_question(ticker: str, question: str):
    # Filter by ticker in ChromaDB metadata
    retriever = vector_db.as_retriever(search_kwargs={"k": 5, "filter": {"ticker": ticker}})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    result = qa_chain.run(question)
    return {"answer": result}
