from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.app.services.forecast_service import get_forecast
from api.app.services.rag_service import answer_question
from api.app.db.session import SessionLocal
from api.app.db.models import UserRequest
from api.app.agents.graph import analyze

app = FastAPI()

class ForecastRequest(BaseModel):
    ticker: str

class FilingRequest(BaseModel):
    ticker: str
    question: str

class AnalyzeRequest(BaseModel):
    query: str

@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        result = get_forecast(req.ticker)
        # Log request
        db = SessionLocal()
        user_req = UserRequest(agent_type="forecast", question=req.ticker, response=str(result))
        db.add(user_req)
        db.commit()
        db.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/ask_filing")
def ask_filing(req: FilingRequest):
    result = answer_question(req.ticker, req.question)
    # Log request
    db = SessionLocal()
    user_req = UserRequest(agent_type="rag", question=req.question, response=str(result))
    db.add(user_req)
    db.commit()
    db.close()
    return result

@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    result = analyze(req.query)
    # Optionally log to SQL here
    return result

@app.get("/")
def root():
    return {"message": "QuantaGen Analyst API is running."}
