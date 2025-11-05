from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.getcwd(), "data", "chroma_data"))

embeddings: Optional[HuggingFaceEmbeddings] = None
vector_db: Optional[Chroma] = None
llm: Optional[ChatGoogleGenerativeAI] = None

async def initialize_rag_service():
    global embeddings, vector_db, llm
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

async def shutdown_rag_service():
    global vector_db
    if vector_db:
        # Attempt to persist before closing, if applicable
        # For Chroma, persist is usually handled during writes, but a final explicit call might be good.
        # However, Chroma's close() method is not directly exposed or typically needed for simple shutdowns.
        # If there were an explicit close/dispose method, it would go here.
        vector_db = None # Dereference to allow garbage collection

async def answer_question(ticker: str, question: str):
    """
    Enhanced RAG system with section-aware retrieval for better context selection.
    """
    global vector_db, llm, embeddings

    # The lifespan event in main.py will ensure these are initialized.
    # No need for a redundant check here.
    assert vector_db is not None, "vector_db not initialized"
    assert llm is not None, "llm not initialized"
    assert embeddings is not None, "embeddings not initialized"

    # Explicitly cast to non-Optional types for Pylance
    current_vector_db: Chroma = vector_db
    current_llm: ChatGoogleGenerativeAI = llm

    # Prompt that guides the model to provide specific, detailed answers
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst. Based EXCLUSIVELY on the provided context...
        **After each specific piece of information or claim, you MUST cite the source using the [Source N] tag it came from.**
        
        Context: {context}"""),
        ("user", "{input}")
    ])


    def format_docs(docs):
        formatted_docs = []
        for i, doc in enumerate(docs):

            source_id = f"[Source {i+1}]"

            source_name = doc.metadata.get("source", "N/A")
            page_num = doc.metadata.get("page", "N/A")

            formatted_doc = f"{source_id}\nSource: {source_name}, Page: {page_num}\nContent: {doc.page_content}"
            formatted_docs.append(formatted_doc)
            
        return "\n\n".join(formatted_docs)

    # Multi-stage retrieval strategy
    def retrieval(query):
        all_docs = []
        seen_content = set()

        # Primary retrieval with increased k for narrative content
        primary_docs = current_vector_db.as_retriever(
            search_kwargs={"k": 8, "filter": {"ticker": ticker}} # This is a single condition, so it's fine
        ).invoke(query)
        all_docs.extend(primary_docs)

        # If question is about risks, boost risk factor chunks
        if any(keyword in query.lower() for keyword in ["risk", "risks", "uncertainties"]):
            risk_docs = current_vector_db.as_retriever(
                search_kwargs={"k": 10, "filter": {"$and": [{"ticker": ticker}, {"section": "risk_factors"}]}}
            ).invoke(query)
            all_docs.extend(risk_docs)
        
        # If question is about MD&A, boost MD&A chunks
        if any(keyword in query.lower() for keyword in ["management discussion", "md&a", "operations", "results of operations", "financial condition"]):
            mdna_docs = current_vector_db.as_retriever(
                search_kwargs={"k": 10, "filter": {"$and": [{"ticker": ticker}, {"section": "mdna"}]}}
            ).invoke(query)
            all_docs.extend(mdna_docs)

        # If question is about financial statements, boost financial statement chunks
        if any(keyword in query.lower() for keyword in ["financial statement", "balance sheet", "income statement", "cash flow", "earnings", "revenue", "profit", "loss"]):
            financial_docs = current_vector_db.as_retriever(
                search_kwargs={"k": 10, "filter": {"$and": [{"ticker": ticker}, {"section": "financial_statements"}]}}
            ).invoke(query)
            all_docs.extend(financial_docs)

        # Deduplicate and limit final results
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:10]  # Limit final results

    # Build enhanced RAG chain
    chain = {
        "context": retrieval | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
    } | prompt | current_llm | StrOutputParser()

    # Invoke with enhanced retrieval
    answer_text = chain.invoke(question)
    
    return {
        "ticker": ticker,
        "question": question,
        "answer": answer_text,
        "retrieval_count": len(retrieval(question))
    }



if __name__ == "__main__":
    pass
    import asyncio

    async def main():
        await initialize_rag_service()
        ticker = "AAPL"
        
        # Test question
        question = "What are the main risks mentioned in the latest 10-K filing?"
        print(f"\n=== TESTING QUESTION: {question} ===")
        answer = await answer_question(ticker, question)
        print(answer)
        await shutdown_rag_service()

    asyncio.run(main())
