from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
# --- ИЗМЕНЕНЫЙ ИМПОРТ ---
from api.app.services.forecast_service import get_forecast_async, _forecast_service_singleton
from api.app.services.rag_service import AsyncRAGService
import asyncio
import re
import logging
import time # <-- ДОБАВЛЕНО для middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with async service initialization."""
    logger.info("Application startup...")
    rag_service = None
    try:
        # --- ИЗМЕНЕНО ---
        # 1. Проверяем, что ForecastService (синглтон) загрузился при импорте
        if _forecast_service_singleton is None:
            logger.critical("ForecastService singleton failed to initialize. Aborting startup.")
            raise RuntimeError("ForecastService singleton failed to initialize.")
        else:
            logger.info("ForecastService singleton initialized successfully.")
            
        # 2. Инициализируем RAG service
        rag_service = AsyncRAGService()
        await rag_service.initialize_once()
        app.state.rag_service = rag_service # Сохраняем сервис в state, чтобы эндпоинты его "видели"
        
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup resources
        if rag_service:
            await rag_service.shutdown()
        logger.info("Application shutdown completed")

app = FastAPI(lifespan=lifespan)

# --- ДОБАВЛЕНО: Middleware для логирования времени ---
@app.middleware("http")
async def log_processing_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request processed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time_ms": f"{processing_time_ms:.2f}"
        }
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
        
        # --- ИЗМЕНЕНО ---
        # Вызываем асинхронную, кэшированную функцию
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
async def ask_filing(request: Request, req: FilingRequest): # <-- ДОБАВЛЕН 'request: Request'
    """Answer questions about company filings using async RAG service."""
    try:
        ticker = req.ticker.upper()
        question = req.question
        # Validate input
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker format")
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # --- ИЗМЕНЕНО ---
        # Получаем синглтон RAG-сервиса из 'state' (который мы положили туда в lifespan)
        rag_service: AsyncRAGService = request.app.state.rag_service
        result = await rag_service.answer_question(ticker, question)
        return result
    except ValidationError as e:
        logger.warning(f"Validation error for {req.ticker}: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        logger.error(f"Error in ask_filing for {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/comprehensive_analysis")
async def comprehensive_analysis(request: Request, req: ComprehensiveAnalysisRequest): # <-- ДОБАВЛЕН 'request: Request'
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
        
        # --- ИЗМЕНЕНО ---
        # Получаем синглтон RAG-сервиса
        rag_service: AsyncRAGService = request.app.state.rag_service
        
        # Создаем задачи для параллельного выполнения
        # 1. Вызываем кэшированный forecast
        forecast_task = get_forecast_async(ticker)
        # 2. Вызываем кэшированный RAG
        rag_task = rag_service.answer_question(ticker, question)
        
        # Execute both tasks in parallel
        forecast_result, rag_result = await asyncio.gather(
            forecast_task, 
            rag_task,
            return_exceptions=True # Не "роняем" весь запрос, если одна из задач упала
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
        "async_improvements": {
            # --- ОБНОВЛЕНО ---
            "forecast_service": "singleton_cached_async_wrapper",
            "rag_service": "singleton_cached_async_wrapper", 
            "concurrent_execution": "comprehensive_analysis_endpoint"
        }
    }
