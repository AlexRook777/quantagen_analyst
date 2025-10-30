# LangGraph orchestration stub
# In production, use LangGraph to route requests to agents

def analyze(query: str):
    # Dummy intent classification
    if "forecast" in query:
        from api.app.services.forecast_service import get_forecast
        return get_forecast("AAPL")
    else:
        from api.app.services.rag_service import answer_question
        return answer_question("AAPL", query)
