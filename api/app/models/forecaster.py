import torch
import torch.nn as nn

class ForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM возвращает (output, (h_n, c_n))
        # Нам нужен только output последнего временного шага
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]

        # Передаем его в Линейный слой, чтобы получить 7 прогнозов
        predictions = self.linear(last_hidden_state)
        return predictions


