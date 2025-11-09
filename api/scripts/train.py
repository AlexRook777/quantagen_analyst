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
    # Load x years of data for Apple
    data = yf.download("AAPL", start="2019-01-01", end=None)
    
    if data is None or data.empty:
        print("Error: No data downloaded or data is empty. Exiting.")
        return

    # Flatten the MultiIndex columns
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Rename the 'Close_AAPL' column to 'Close' for pandas-ta
    data.rename(columns={'Close_AAPL': 'Close'}, inplace=True)

    # ========================================
    # 1. FEATURE ENGINEERING SECTION
    # ========================================
    # Pandas-ta will do all the complex mathematics for you

    # 1.1. Calculate RSI (Relative Strength Index)
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # 1.2. Calculate MACD (Moving Average Convergence Divergence)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    # Concatenate the MACD DataFrame, ensuring columns are handled correctly
    data = pd.concat([data, macd], axis=1) # This should now work correctly with single-level index

    # 1.3. Calculate 7-day and 30-day simple moving averages (SMA)
    data['SMA_7'] = ta.sma(data['Close'], length=7)
    data['SMA_30'] = ta.sma(data['Close'], length=30)

    # 1.4. Remove rows with NaN (they appear at the beginning due to calculations)
    data = data.dropna()

    # ========================================
    # 2. DATA PREPARATION SECTION
    # ========================================
    # 2.1. Choose what we will predict
    # We will predict *only* the 'Close' price
    target_column = 'Close'

    # 2.2. Normalize ALL data
    # The neural network will look at 'Close', 'RSI', 'MACD' to predict 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # 2.3. Save the "index" of the column with our target ('Close')
    # This is needed to "de-normalize" the prediction at the end
    target_column_index = data.columns.get_loc(target_column)

    # ========================================
    # 3. SEQUENCE CREATION SECTION
    # ========================================
    LOOKBACK_WINDOW = 60 # Days "backward"
    FORECAST_HORIZON = 7  # Days "forward"

    X = [] # List for "input" windows (X)
    y = [] # List for "target" predictions (y)

    for i in range(len(data_scaled) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1):
        # X = window of 60 days with ALL features (Close, RSI, MACD...)
        window = data_scaled[i : i + LOOKBACK_WINDOW]
        X.append(window)

        # y = window of 7 days ONLY with 'Close'
        target = data_scaled[i + LOOKBACK_WINDOW : i + LOOKBACK_WINDOW + FORECAST_HORIZON, target_column_index]
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # ========================================
    # 4. TRAIN/TEST SPLIT SECTION
    # ========================================
    split_ratio = 0.9
    split_index = int(len(X) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # ========================================
    # 5. TENSOR CONVERSION SECTION
    # ========================================
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ========================================
    # 6. MODEL DEFINITION SECTION
    # ========================================
    # Define parameters of our model
    # input_size = number of features
    model = ForecastModel(input_size=X_train.shape[2], hidden_size=64, num_layers=2, output_size=FORECAST_HORIZON)

    criterion = nn.MSELoss() # Mean Squared Error (since this is regression)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ========================================
    # 7. TRAINING SECTION
    # ========================================
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train() # Training mode
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation at the end of epoch
        model.eval() # Evaluation mode
        with torch.no_grad():
            test_loss = 0
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss / len(test_loader):.4f}")

    # ========================================
    # 8. PREDICTION SECTION
    # ========================================
    # Make prediction on entire test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    # ========================================
    # 9. DENORMALIZATION SECTION
    # ========================================
    # Create a "dummy" of the same shape as the ORIGINAL data
    predictions_unscaled = np.zeros((len(predictions), data_scaled.shape[1]))
    # Place our 7-day predictions in the correct column ('Close')
    predictions_unscaled[:, target_column_index:target_column_index+FORECAST_HORIZON] = predictions 
    # Inverse transform
    predictions_unscaled = scaler.inverse_transform(predictions_unscaled)[:, target_column_index:target_column_index+FORECAST_HORIZON]

    # Same thing for y_test
    y_test_unscaled = np.zeros((len(y_test), data_scaled.shape[1]))
    y_test_unscaled[:, target_column_index:target_column_index+FORECAST_HORIZON] = y_test
    y_test_unscaled = scaler.inverse_transform(y_test_unscaled)[:, target_column_index:target_column_index+FORECAST_HORIZON]

    # ========================================
    # 10. MODEL PERSISTENCE SECTION
    # ========================================
    # 10.1. Save the model 
    torch.save(model.state_dict(), "api/saved_models/quant_model.pth")
    # 10.2. Save the scaler
    joblib.dump(scaler, "api/saved_models/scaler.joblib")
    print("Model and scaler saved successfully.")

    # ========================================
    # 11. VISUALIZATION SECTION
    # ========================================
    # Draw a graph (for example, only the first day of prediction)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_unscaled[:, 0], label="Actual Price (Day +1)")
    plt.plot(predictions_unscaled[:, 0], label="Predicted Price (Day +1)", linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_forecasting_model()
