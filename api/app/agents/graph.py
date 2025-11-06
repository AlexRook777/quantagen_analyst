# LangGraph orchestration stub
# In production, use LangGraph to route requests to agents

from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

def classify_intent(query: str) -> str:
    prompt = (
        "Classify the user intent for the following query as either 'forecast' (for price prediction) or 'qa' (for document Q&A):\n"
        f"Query: {query}\nIntent: "
    )
    intent = llm(prompt).strip().lower()
    if "forecast" in intent:
        return "forecast"
    return "qa"

def analyze(query: str):
    intent = classify_intent(query)
    if intent == "forecast":
        from api.app.services.forecast_service import ForecastService
        # Extract ticker if possible (simple demo)
        import re
        match = re.search(r"([A-Z]{1,5})", query)
        ticker = match.group(1) if match else "AAPL"
        forecast_service = ForecastService()
        return forecast_service.get_forecast(ticker)
    else:
        from api.app.services.rag_service import RAGService
        import asyncio
        # Extract ticker and question
        import re
        match = re.search(r"([A-Z]{1,5})", query)
        ticker = match.group(1) if match else "AAPL"
        
        # Create and run async RAG service
        rag_service = RAGService()
        
        async def run_rag():
            await rag_service.initialize()
            try:
                response = await rag_service.answer_question(ticker, query)
                return {
                    "ticker": response.ticker,
                    "question": response.question,
                    "answer": response.answer,
                    "retrieval_count": response.retrieval_count
                }
            finally:
                await rag_service.shutdown()
        
        return asyncio.run(run_rag())
