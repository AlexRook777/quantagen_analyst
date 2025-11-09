import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NEW MODULE: ATTENTION ---
class Attention(nn.Module):
    """
    Simple 'Bahdanau' (additive) attention mechanism.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # "W_a" is a trainable weight matrix in nn.Linear
        self.Wa = nn.Linear(hidden_size, hidden_size)
        # "v_a" is a trainable vector
        self.va = nn.Parameter(torch.rand(hidden_size))
        self.va.data.normal_(mean=0.0, std=1.0 / (hidden_size**0.5))

    def forward(self, all_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hidden_states (torch.Tensor): LSTM output (batch_size, seq_len, hidden_size)
        
        Returns:
            torch.Tensor: Context vector (batch_size, hidden_size)
        """
        # 1. Calculate "energies" (e)
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        energies = torch.tanh(self.Wa(all_hidden_states))
        
        # 2. Calculate "scores" (a)
        # (batch_size, seq_len, hidden_size) * (hidden_size) -> (batch_size, seq_len)
        scores = energies.matmul(self.va)
        
        # 3. Normalize scores to "attention weights" (alpha)
        # (batch_size, seq_len)
        attn_weights = F.softmax(scores, dim=1)
        
        # 4. Calculate "context vector" (c)
        # (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_size) -> (batch_size, 1, hidden_size)
        # .unsqueeze(1) adds an "empty" dimension for matrix multiplication
        context_vector = torch.bmm(attn_weights.unsqueeze(1), all_hidden_states)
        
        # (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
        return context_vector.squeeze(1)

# --- UPDATED MODEL ---
class ForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.1):
        """
        Initialization V2:
        - Added Dropout to LSTM
        - Added Attention mechanism
        - Added advanced "head" (Head)
        """
        super(ForecastModel, self).__init__()
        
        # --- 1. Improved LSTM with Dropout ---
        # Dropout is added between LSTM layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0 # Dropout doesn't work for 1-layer LSTM
        )
        
        # --- 2. Attention Mechanism ---
        self.attention = Attention(hidden_size)
        
        # --- 3. Improved "Head" ---
        # Instead of a single Linear, we build a "feed-forward" block
        self.fc_head = nn.Sequential(
            nn.LayerNorm(hidden_size), # Stabilizes LSTM/Attention output
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, output_size) # Final output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass V2:
        - Uses ALL 60 hidden states, not just the last one
        - Applies Attention to "weight" them
        - Passes the weighted "context vector" to the "head"
        """
        # x shape: (batch_size, seq_len=60, input_size=11)
        
        # 1. Run through LSTM
        # lstm_out shape: (batch_size, seq_len=60, hidden_size)
        lstm_out, _ = self.lstm(x) 
        
        # 2. Apply Attention to ALL 60 LSTM outputs
        # context_vector shape: (batch_size, hidden_size)
        context_vector = self.attention(lstm_out)
        
        # 3. Pass the "concentrated" vector to the "head" for prediction
        # predictions shape: (batch_size, output_size=7)
        predictions = self.fc_head(context_vector)
        
        return predictions
