"""
Forecast service for stock price prediction using LSTM neural networks.

This module provides functionality to load financial data, perform feature engineering,
and generate forecasts using a pre-trained model.
"""

import logging
import datetime
from dataclasses import dataclass
import os
from typing import List, Optional
import asyncio  
import cachetools.func  
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from api.app.models.forecaster import ForecastModel

logger = logging.getLogger(__name__)

@dataclass # <-- 2. Add decorator
class ForecastConfig:
    """Configuration class for forecast service parameters."""
    
    # Model parameters
    lookback_window: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    forecast_horizon: int = 7
    data_period_days: int = 150
    
    # Feature engineering parameters
    rsi_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_short: int = 7
    sma_long: int = 30
    
    # File paths (for model and scaler)
    base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    scaler_path: str = os.path.join(base_dir, "saved_models", "scaler.joblib")
    model_path: str = os.path.join(base_dir, "saved_models", "quant_model.pth")

@dataclass
class ForecastResponse:
    ticker: str
    forecast_days: int
    forecast: List[float]
    generated_at: str 
    model_version: str = "1.0.0"

class ForecastService:
    """
    Service for generating stock price forecasts using machine learning.
    
    This class encapsulates the entire forecasting pipeline including data loading,
    feature engineering, model inference, and post-processing.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        """
        Initialize the forecast service.
        
        Args:
            config: Configuration object. If None, default config is used.
        """
        self.config = config or ForecastConfig()
        self.model: Optional[ForecastModel] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_size: Optional[int] = None
        self.target_column_index: Optional[int] = None
        self._initialize_artifacts()
    
    def _initialize_artifacts(self):
        """Load and initialize the ML model and scaler artifacts."""
        try:
            # Verify artifact files exist
            if not os.path.exists(self.config.scaler_path):
                raise FileNotFoundError(f"Scaler not found at {self.config.scaler_path}")
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
            
            # Load scaler
            self.scaler = joblib.load(self.config.scaler_path)
            if not hasattr(self.scaler, 'n_features_in_') or not hasattr(self.scaler, 'feature_names_in_'):
                raise ValueError("Scaler object missing required attributes")
            
            assert self.scaler is not None, "Scaler should be initialized at this point"
            self.input_size = self.scaler.n_features_in_
            feature_names = self.scaler.feature_names_in_
            if 'Close' not in feature_names:
                raise ValueError("'Close' feature not found in scaler feature names")
            self.target_column_index = feature_names.tolist().index('Close')
            
            # Load model
            self.model = ForecastModel(
                input_size=self.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                output_size=self.config.forecast_horizon
            )
            # Load model on CPU, which is safe for inference in FastAPI
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            
            logger.info("Successfully initialized forecast service artifacts")
            
        except Exception as e:
            logger.error(f"Failed to initialize artifacts: {e}")
            raise
    
    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            
        Returns:
            DataFrame with stock data
            
        Raises:
            ValueError: If no data is downloaded
        """
        try:
            data = yf.download(ticker, period=f"{self.config.data_period_days}d", auto_adjust=True)
            
            if data is None or data.empty:
                raise ValueError(f"No data downloaded for ticker '{ticker}'")
            
            logger.info(f"Loaded {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            raise ValueError(f"Failed to download data for ticker '{ticker}': {e}")
    
    def _standardize_columns(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Standardize column names to handle MultiIndex and ticker-specific naming.
        
        Args:
            data: Raw data from yfinance
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with standardized column names
        """
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            
            # Rename ticker-specific Close column to generic 'Close'
            close_col_name = f'Close_{ticker.upper()}'
            if close_col_name in data.columns:
                data.rename(columns={close_col_name: 'Close'}, inplace=True)
        
        return data
    
    def _create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical analysis features from price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical features
        """
        # RSI
        data['RSI'] = ta.rsi(data['Close'], length=self.config.rsi_length)
        
        # MACD
        macd = ta.macd(
            data['Close'], 
            fast=self.config.macd_fast, 
            slow=self.config.macd_slow, 
            signal=self.config.macd_signal
        )
        data = pd.concat([data, macd], axis=1)
        
        # Simple Moving Averages
        data['SMA_7'] = ta.sma(data['Close'], length=self.config.sma_short)
        data['SMA_30'] = ta.sma(data['Close'], length=self.config.sma_long)
        
        # Remove NaN values after feature calculation
        data = data.dropna()
        
        return data
    
    def _prepare_input_data(self, ticker: str) -> pd.DataFrame:
        """
        Load and prepare data for forecasting.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with processed features
        """
        # Load raw data
        data = self.load_stock_data(ticker)
        
        # Standardize column names
        data = self._standardize_columns(data, ticker)
        
        # Create technical features
        data = self._create_technical_features(data)
        
        if data.empty:
            raise ValueError("Data became empty after feature engineering")
        
        # Get the required lookback window
        last_days = data.tail(self.config.lookback_window)
        
        if len(last_days) < self.config.lookback_window:
            raise ValueError(
                f"Not enough data: need {self.config.lookback_window} days, "
                f"got {len(last_days)} days"
            )
        
        return last_days
    
    def _normalize_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Normalize data using the pre-loaded scaler.
        
        Args:
            data: DataFrame with feature data
            
        Returns:
            Normalized numpy array
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not initialized")
        
        try:
            # Ensure column order matches training data
            ordered_data = data[self.scaler.feature_names_in_]
            return self.scaler.transform(ordered_data)
        except KeyError:
            available_cols = set(data.columns)
            expected_cols = set(self.scaler.feature_names_in_)
            missing_cols = expected_cols - available_cols
            
            raise ValueError(
                f"Feature mismatch. Missing features: {missing_cols}. "
                f"Expected: {self.scaler.feature_names_in_}, "
                f"Available: {data.columns.tolist()}"
            )
    
    def _predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Generate prediction using the loaded model.
        
        Args:
            input_data: Normalized input data
            
        Returns:
            Scaled predictions array
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Convert to tensor with batch dimension and ensure shape is correct
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            prediction_scaled = self.model(input_tensor).numpy().flatten()
        
        return prediction_scaled
    
    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> List[float]:
        """
        Convert scaled predictions back to original price scale.
        
        Args:
            prediction_scaled: Scaled predictions from model
            
        Returns:
            List of price predictions
        """
        if self.scaler is None or self.input_size is None or self.target_column_index is None:
            raise RuntimeError("Scaler not properly initialized")
        
        # Create helper array for inverse transform
        helper = np.zeros((self.config.forecast_horizon, self.input_size))
        helper[:, self.target_column_index] = prediction_scaled
        
        # Inverse transform and extract target column
        full_prediction = self.scaler.inverse_transform(helper)
        target_prediction = full_prediction[:, self.target_column_index]
        
        return target_prediction.tolist()
    
    def get_forecast(self, ticker: str) -> ForecastResponse:
        """
        Generate a stock price forecast for the given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            
        Returns:
            List of predicted prices for the next 7 days
            
        Raises:
            ValueError: If ticker is invalid or insufficient data
        """
        try:
            logger.info(f"Generating forecast for ticker: {ticker}")
            
            # Validate ticker
            if not ticker or not isinstance(ticker, str):
                raise ValueError("Ticker must be a non-empty string")
            
            # Prepare input data
            prepared_data = self._prepare_input_data(ticker)
            
            # Normalize data
            normalized_data = self._normalize_data(prepared_data)
            
            # Generate prediction
            prediction_scaled = self._predict(normalized_data)
            
            # Convert back to original scale
            final_forecast = self._inverse_transform_prediction(prediction_scaled)
            
            logger.info(f"Successfully generated forecast for {ticker}")
            return ForecastResponse(
                ticker=ticker,
                forecast_days=self.config.forecast_horizon,
                forecast=final_forecast,
                generated_at=datetime.datetime.now(datetime.UTC).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate forecast for {ticker}: {e}")
            raise

# --- 3. CREATE "SINGLETON" (SINGLE INSTANCE) ---
# This code will run once on import, loading the model and scaler into memory.
try:
    _forecast_service_singleton = ForecastService()
except Exception as e:
    logger.critical(f"Failed to initialize global ForecastService singleton: {e}")
    _forecast_service_singleton = None

# --- 4. DEFINE CACHED BLOCKING FUNCTION ---
# This function will be a "wrapper" for calling our singleton
# Cache for 100 tickers, "time to live" - 900 seconds (15 minutes)
@cachetools.func.ttl_cache(maxsize=100, ttl=900)
def _get_forecast_blocking_cached(ticker: str) -> ForecastResponse:
    """
    Internal cacheable function that performs the heavy work.
    Calls the get_forecast() method of our global singleton.
    """
    if _forecast_service_singleton is None:
        logger.error("ForecastService singleton is not initialized.")
        raise RuntimeError("ForecastService is not available due to initialization failure.")
        
    logger.info(f"Cache miss. Generating new forecast for {ticker}")
    return _forecast_service_singleton.get_forecast(ticker)

# --- 5. CREATE ASYNC "WRAPPER" THAT FASTAPI CALLS ---
async def get_forecast_async(ticker: str) -> ForecastResponse:
    """
    Generate a stock price forecast using async execution and caching.
    
    This function wraps the synchronous, cached forecast generation in an 
    async context to prevent blocking the event loop.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of predicted prices
    """
    try:
        # Call the cached function in a separate thread
        return await asyncio.to_thread(_get_forecast_blocking_cached, ticker)
    except Exception as e:
        logger.error(f"Error in async forecast for {ticker}: {e}")
        # Pass error forward so FastAPI can catch it
        raise
