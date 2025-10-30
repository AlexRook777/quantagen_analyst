# Script to train PyTorch model for forecasting
import yfinance as yf
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from api.app.models.forecaster import ForecastModel

TICKER = "AAPL"
SEQ_LEN = 10
FEATURES = ["Close", "Volume"]

# Download data
data = yf.download(TICKER, period="2y")
# Feature engineering (add more indicators as needed)
data["SMA_7"] = data["Close"].rolling(window=7).mean()
data = data.dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[FEATURES + ["SMA_7"]])
joblib.dump(scaler, "../saved_models/scaler.joblib")

# Prepare sequences
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 7):
        x = data[i:i+seq_len]
        y = data[i+seq_len:i+seq_len+7, 0]  # predict Close for next 7 days
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

X, y = create_sequences(scaled, SEQ_LEN)

model = ForecastModel(input_size=X.shape[2], output_size=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

# Save model
torch.save(model.state_dict(), "../saved_models/quant_model.pth")
