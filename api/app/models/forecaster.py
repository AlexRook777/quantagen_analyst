import torch
import torch.nn as nn

class ForecastModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=7, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

# If you want to mock model for tests, add a dummy class
class DummyForecastModel:
    def __call__(self, x):
        return torch.ones((x.shape[0], 7))
