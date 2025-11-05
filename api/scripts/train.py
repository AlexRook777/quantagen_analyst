import sys
import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Add the project root to the Python path
from api.app.models.forecaster import ForecastModel

def train_forecasting_model():
    # Загружаем x лет данных для Apple
    data = yf.download("AAPL", start="2019-01-01", end=None)
    
    if data is None or data.empty:
        print("Error: No data downloaded or data is empty. Exiting.")
        return

    # Flatten the MultiIndex columns
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Rename the 'Close_AAPL' column to 'Close' for pandas-ta
    data.rename(columns={'Close_AAPL': 'Close'}, inplace=True)

    # --- Feature Engineering (Создание признаков) ---
    # Pandas-ta сделает за вас всю сложную математику

    # 1. Рассчитываем RSI (Relative Strength Index)
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # 2. Рассчитываем MACD (Moving Average Convergence Divergence)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    # Concatenate the MACD DataFrame, ensuring columns are handled correctly
    data = pd.concat([data, macd], axis=1) # This should now work correctly with single-level index


    # 3. Рассчитываем 7-дневное и 30-дневное скользящие средние (SMA)
    data['SMA_7'] = ta.sma(data['Close'], length=7)
    data['SMA_30'] = ta.sma(data['Close'], length=30)


    # 5. Удаляем строки с NaN (они появляются в начале из-за вычислений)
    data = data.dropna()

    # --- 1. Выбираем, что будем прогнозировать ---
    # Мы будем прогнозировать *только* цену 'Close'
    target_column = 'Close'

    # --- 2. Нормализуем ВСЕ данные ---
    # Нейросеть будет смотреть на 'Close', 'RSI', 'MACD', чтобы предсказать 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # --- 3. Сохраняем "индекс" колонки с нашей целью ('Close') ---
    # Это нужно, чтобы "раз-нормализовать" прогноз в конце
    target_column_index = data.columns.get_loc(target_column)


    LOOKBACK_WINDOW = 60 # Дней "назад"
    FORECAST_HORIZON = 7  # Дней "вперед"

    X = [] # Список для "входных" окон (X)
    y = [] # Список для "целевых" прогнозов (y)

    for i in range(len(data_scaled) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1):
        # X = окно из 60 дней со ВСЕМИ фичами (Close, RSI, MACD...)
        window = data_scaled[i : i + LOOKBACK_WINDOW]
        X.append(window)

        # y = окно из 7 дней ТОЛЬКО с 'Close'
        target = data_scaled[i + LOOKBACK_WINDOW : i + LOOKBACK_WINDOW + FORECAST_HORIZON, target_column_index]
        y.append(target)

    X = np.array(X)
    y = np.array(y)


    split_ratio = 0.9
    split_index = int(len(X) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Конвертируем в Тензоры
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Создаем DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Определяем параметры нашей модели
    # input_size = кол-во фичей
    model = ForecastModel(input_size=X_train.shape[2], hidden_size=64, num_layers=2, output_size=FORECAST_HORIZON)

    criterion = nn.MSELoss() # Mean Squared Error (т.к. это регрессия)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 20
    for epoch in range(num_epochs):
        model.train() # Режим обучения
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Валидация в конце эпохи
        model.eval() # Режим оценки
        with torch.no_grad():
            test_loss = 0
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss / len(test_loader):.4f}")


    # Делаем прогноз на всем тестовом наборе
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    # Создаем "пустышку" той же формы, что и ОРИГИНАЛЬНЫЕ данные
    predictions_unscaled = np.zeros((len(predictions), data_scaled.shape[1]))
    # Кладем наши 7-дневные прогнозы в нужную колонку ('Close')
    predictions_unscaled[:, target_column_index:target_column_index+FORECAST_HORIZON] = predictions 
    # Инвертируем
    predictions_unscaled = scaler.inverse_transform(predictions_unscaled)[:, target_column_index:target_column_index+FORECAST_HORIZON]

    # То же самое для y_test
    y_test_unscaled = np.zeros((len(y_test), data_scaled.shape[1]))
    y_test_unscaled[:, target_column_index:target_column_index+FORECAST_HORIZON] = y_test
    y_test_unscaled = scaler.inverse_transform(y_test_unscaled)[:, target_column_index:target_column_index+FORECAST_HORIZON]


    # 1. Сохраняем модель 
    torch.save(model.state_dict(), "api/saved_models/quant_model.pth")
    # 2. Сохраняем scaler
    joblib.dump(scaler, "api/saved_models/scaler.joblib")
    print("Model and scaler saved successfully.")


    # Рисуем график (например, только первый день прогноза)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_unscaled[:, 0], label="Actual Price (Day +1)")
    plt.plot(predictions_unscaled[:, 0], label="Predicted Price (Day +1)", linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_forecasting_model()