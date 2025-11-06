from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from api.app.services.forecast_service import get_forecast
from api.app.services.rag_service import answer_question, initialize_rag_service, shutdown_rag_service
#from api.app.db.session import SessionLocal
#from api.app.db.models import UserRequest
#from api.app.agents.graph import analyze
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = None
    try:
        service = await initialize_rag_service()
        yield
    finally:
        if service:
            await shutdown_rag_service(service)

app = FastAPI(lifespan=lifespan)

class ForecastRequest(BaseModel):
    ticker: str

class FilingRequest(BaseModel):
    ticker: str
    question: str

class AnalyzeRequest(BaseModel):
    query: str


@app.post("/forecast")
async def forecast(req: ForecastRequest):
    try:
        ticker = req.ticker.upper()
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        result = await get_forecast(ticker)
        return result
    except ValidationError as e:
        logger.warning(f"Validation error for ticker {req.ticker}: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except ValueError as e:
        logger.error(f"Value error for ticker {req.ticker}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred for ticker {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ask_filing")
async def ask_filing(req: FilingRequest):
    try:
        ticker = req.ticker.upper()
        question = req.question
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        result = await answer_question(ticker, question)
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# @app.post("/analyze")
# def analyze_endpoint(req: AnalyzeRequest):
#     result = analyze(req.query)
#     # Optionally log to SQL here
#     return result

@app.get("/")
def root():
    return {"message": "QuantaGen Analyst API is running."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "QuantaGen Analyst API",
        "version": "1.0.0",
        "features": ["forecast", "ask_filing"]
    }
