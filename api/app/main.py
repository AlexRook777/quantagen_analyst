from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from api.app.services.forecast_service import get_forecast_async, _forecast_service_singleton
from api.app.services.rag_service import get_rag_response_async
import asyncio
import re
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with async service initialization."""
    logger.info("Application startup...")
    try:
        # 1. Check that ForecastService (singleton) loaded on import
        if _forecast_service_singleton is None:
            logger.critical("ForecastService singleton failed to initialize. Aborting startup.")
            raise RuntimeError("ForecastService singleton failed to initialize.")
        else:
            logger.info("ForecastService singleton initialized successfully.")
            
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Application shutdown completed")

app = FastAPI(lifespan=lifespan)

# Middleware for time logging 
@app.middleware("http")
async def log_processing_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Request processed: "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status_code={response.status_code} "
        f"processing_time_ms={processing_time_ms:.2f}"
    )
    return response

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
async def ask_filing(request: Request, req: FilingRequest): 
    """Answer questions about company filings using async RAG service."""
    try:
        ticker = req.ticker.upper()
        question = req.question
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = await get_rag_response_async(ticker, question)
        return result
    except ValidationError as e:
        logger.warning(f"Validation error for {req.ticker}: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        logger.error(f"Error in ask_filing for {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/comprehensive_analysis")
async def comprehensive_analysis(request: Request, req: ComprehensiveAnalysisRequest): 
    """
    Perform comprehensive analysis by running forecast and RAG analysis concurrently.
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
        
        # Create tasks for parallel execution
        # 1. Call cached forecast
        forecast_task = get_forecast_async(ticker)
        # 2. Call cached RAG
        rag_task = get_rag_response_async(ticker, question)
        
        # Execute both tasks in parallel
        forecast_result, rag_result = await asyncio.gather(
            forecast_task, 
            rag_task,
            return_exceptions=True # Don't "crash" the whole request if one task fails
        )
        
        # Handle potential errors
        response_data: dict = {
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
        "async_improvements": True
    }
