from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from api.app.services.forecast_service import get_forecast_async
from api.app.services.rag_service import get_async_rag_service, shutdown_async_rag_service
import asyncio
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with async service initialization."""
    rag_service = None
    try:
        # Initialize the async RAG service
        rag_service = await get_async_rag_service()
        await rag_service.initialize_once()
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup resources
        if rag_service:
            await shutdown_async_rag_service()
        logger.info("Application shutdown completed")

app = FastAPI(lifespan=lifespan)

class ForecastRequest(BaseModel):
    ticker: str

class FilingRequest(BaseModel):
    ticker: str
    question: str

class ComprehensiveAnalysisRequest(BaseModel):
    ticker: str
    question: str


@app.post("/forecast")
async def forecast(req: ForecastRequest):
    """Generate stock price forecast using async ML service."""
    try:
        ticker = req.ticker.upper()
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        
        # Use async forecast service
        result = await get_forecast_async(ticker)
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
    """Answer questions about company filings using async RAG service."""
    try:
        ticker = req.ticker.upper()
        question = req.question
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Use async RAG service
        rag_service = await get_async_rag_service()
        result = await rag_service.answer_question(ticker, question)
        return result
    except ValidationError as e:
        logger.warning(f"Validation error for {req.ticker}: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        logger.error(f"Error in ask_filing for {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/comprehensive_analysis")
async def comprehensive_analysis(req: ComprehensiveAnalysisRequest):
    """
    Perform comprehensive analysis by running forecast and RAG analysis concurrently.
    
    This endpoint demonstrates the power of async execution by running both
    the ML forecasting and document analysis in parallel, reducing total
    response time significantly.
    """
    try:
        ticker = req.ticker.upper()
        question = req.question
        
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Starting comprehensive analysis for {ticker}")
        
        # Run both services concurrently
        forecast_task = get_forecast_async(ticker)
        rag_service = await get_async_rag_service()
        rag_task = rag_service.answer_question(ticker, question)
        
        # Execute both tasks in parallel
        forecast_result, rag_result = await asyncio.gather(
            forecast_task, 
            rag_task,
            return_exceptions=True
        )
        
        # Handle potential errors
        response_data = {
            "ticker": ticker,
            "question": question
        }
        
        if isinstance(forecast_result, Exception):
            logger.error(f"Forecast generation failed: {forecast_result}")
            response_data["forecast"] = {"error": str(forecast_result)}
        else:
            response_data["forecast"] = forecast_result
        
        if isinstance(rag_result, Exception):
            logger.error(f"RAG analysis failed: {rag_result}")
            response_data["rag_analysis"] = {"error": str(rag_result)}
        else:
            response_data["rag_analysis"] = rag_result
        
        logger.info(f"Completed comprehensive analysis for {ticker}")
        return response_data
        
    except ValidationError as e:
        logger.warning(f"Validation error for {req.ticker}: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        logger.error(f"Error in comprehensive analysis for {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# @app.post("/analyze")
# def analyze_endpoint(req: AnalyzeRequest):
#     result = analyze(req.query)
#     # Optionally log to SQL here
#     return result

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "QuantaGen Analyst API is running.",
        "version": "2.0.0",
        "features": ["forecast", "ask_filing", "comprehensive_analysis"],
        "async_enhanced": True
    }

@app.get("/health")
def health_check():
    """Health check endpoint with service status."""
    return {
        "status": "healthy",
        "service": "QuantaGen Analyst API",
        "version": "2.0.0",
        "features": ["forecast", "ask_filing", "comprehensive_analysis"],
        "async_improvements": {
            "forecast_service": "async_wrapper_enabled",
            "rag_service": "singleton_pattern_enabled", 
            "concurrent_execution": "comprehensive_analysis_endpoint"
        }
    }
