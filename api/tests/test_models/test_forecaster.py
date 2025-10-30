import torch
import pytest

# Dummy ForecastModel for test scaffold
class ForecastModel(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=8, output_size=7):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

def test_model_creation():
    model = ForecastModel()
    assert isinstance(model, ForecastModel)

def test_model_forward_pass_shape():
    model = ForecastModel()
    dummy = torch.randn(1, 10, 4)
    out = model(dummy)
    assert out.shape == (1, 7)
