import torch
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
from api.app.models.forecaster import ForecastModel
import os # Добавлено для пути

# --- 1. Определение констант ---
LOOKBACK_WINDOW = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FORECAST_HORIZON = 7
# INPUT_SIZE будет определен из scaler'а
SCALER_PATH = "api/saved_models/scaler.joblib"
MODEL_PATH = "api/saved_models/quant_model.pth"

# --- 2. Загрузка артефактов ---
# Проверяем, существуют ли файлы
if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model or scaler not found. Please run the training script first.")

scaler = joblib.load(SCALER_PATH)
# Определяем INPUT_SIZE из самого scaler'а
INPUT_SIZE = scaler.n_features_in_
target_column_index = scaler.feature_names_in_.tolist().index('Close')

model = ForecastModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=FORECAST_HORIZON)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # ВАЖНО: перевести в режим инференса

def get_forecast(ticker: str):
    # 1. Загружаем ПОСЛЕДНИЕ данные
    # Нам нужно 60 (lookback) + 30 (SMA_30) = 90 дней
    # Загрузим 150 *календарных* дней (~105 торговых дней), чтобы был запас
    data = yf.download(ticker, period="150d") # <-- ИЗМЕНЕНО: 100d -> 150d
    
    if data is None or data.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker or period.")
    
    # --- 2. Делаем АБСОЛЮТНО тот же Feature Engineering, что и в train.py ---
    
    # ИСПРАВЛЕНИЕ: yfinance.download() возвращает РАЗНЫЕ столбцы с/без auto_adjust.
    # Мы должны сделать ИДЕНТИЧНО как в train.py
    
    # Шаг 2a: Выравнивание столбцов (если это MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        # Переименовываем 'Close_TICKER' в 'Close'
        close_col_name = f'Close_{ticker.upper()}'
        if close_col_name in data.columns:
            data.rename(columns={close_col_name: 'Close'}, inplace=True)
    
    # Проверяем, есть ли 'Adj Close', как в train.py
    if 'Adj Close' not in data.columns:
        # yfinance > 0.2.0 убрал 'Adj Close' по умолчанию. 
        # Если в train.py он был, нам нужно его добавить (или переобучить)
        # Для простоты, предположим, что train.py *тоже* его не имел.
        # НО! scaler был обучен на 11 фичах. Давайте посмотрим, каких.
        # scaler.feature_names_in_ покажет нам
        pass # Мы предполагаем, что scaler был обучен на 11 фичах

    # Шаг 2b: Расчет фичей
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data = pd.concat([data, macd], axis=1)
    data['SMA_7'] = ta.sma(data['Close'], length=7)
    data['SMA_30'] = ta.sma(data['Close'], length=30)
    
    # Шаг 2c: Удаляем NaN *после* всех расчетов
    data = data.dropna()

    if data.empty:
        raise ValueError("Data became empty after feature engineering and dropna. Not enough data downloaded.")

    # 3. Берем ПОСЛЕДНИЕ 60 дней
    # Нам нужны *только* 60 дней. data.tail() - это то, что нужно.
    last_60_days_raw = data.tail(LOOKBACK_WINDOW)

    # Проверка, что у нас достаточно данных
    if len(last_60_days_raw) < LOOKBACK_WINDOW:
        raise ValueError(f"Not enough data to form a {LOOKBACK_WINDOW}-day lookback window. Only got {len(last_60_days_raw)} rows.")

    # 4. Нормализуем их с помощью ЗАГРУЖЕННОГО scaler'а
    # Убедимся, что столбцы в том же порядке, что и при обучении
    try:
        last_60_days_raw_ordered = last_60_days_raw[scaler.feature_names_in_]
    except KeyError as e:
        raise ValueError(f"Feature mismatch. Model was trained on features: {scaler.feature_names_in_}, but service has: {data.columns.tolist()}. Error: {e}")
        
    last_60_days_scaled = scaler.transform(last_60_days_raw_ordered)

    # 5. Конвертируем в тензор (добавляем batch_size=1)
    input_tensor = torch.tensor(last_60_days_scaled, dtype=torch.float32).unsqueeze(0)

    # 6. Делаем прогноз
    with torch.no_grad():
        prediction_scaled = model(input_tensor).numpy().flatten() # (7,)

    # 7. Инвертируем прогноз обратно к оригинальному масштабу
    # (Мы используем "трюк" для инвертирования только 1 столбца)
    
    # Создаем "пустышку" (dummy array)
    prediction_unscaled_helper = np.zeros((FORECAST_HORIZON, INPUT_SIZE))
    
    # Кладем наши 7-дневные прогнозы в нужную колонку ('Close')
    prediction_unscaled_helper[:, target_column_index] = prediction_scaled

    # Инвертируем *только* этот helper
    final_forecast_full = scaler.inverse_transform(prediction_unscaled_helper)
    
    # Извлекаем столбец 'Close', который теперь в долларах
    final_forecast = final_forecast_full[:, target_column_index]
        
    return final_forecast.tolist()
