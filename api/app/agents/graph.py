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
        from api.app.services.forecast_service import get_forecast
        # Extract ticker if possible (simple demo)
        import re
        match = re.search(r"([A-Z]{1,5})", query)
        ticker = match.group(1) if match else "AAPL"
        return get_forecast(ticker)
    else:
        from api.app.services.rag_service import answer_question
        # Extract ticker and question
        import re
        match = re.search(r"([A-Z]{1,5})", query)
        ticker = match.group(1) if match else "AAPL"
        return answer_question(ticker, query)
