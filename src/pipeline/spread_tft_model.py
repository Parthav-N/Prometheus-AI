"""
spread_tft_model.py
====================
Contains only the SpreadTFT model class and loss functions.
Import from here to avoid triggering training when loading the model.

Usage:
    from spread_tft_model import SpreadTFT
"""

import torch
import torch.nn as nn


class SpreadTFT(nn.Module):
    """
    Simplified TFT for fire spread prediction.
    Input: flat feature vector (weather + fire state + terrain)
    Output: P10/P50/P90 growth rate at 3 horizons = 9 values total
    """

    def __init__(self, n_features, d_model=64, n_heads=4,
                 n_lstm_layers=2, dropout=0.1,
                 n_horizons=3, n_quantiles=3):
        super().__init__()
        self.n_horizons  = n_horizons
        self.n_quantiles = n_quantiles

        # Variable Selection Network
        self.vsn = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ELU(),
        )
        self.vsn_gate = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.Sigmoid(),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0
        )

        # Multi-head self-attention
        self.attn      = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # Separate output head per horizon
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_quantiles)
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x):
        vsn_out     = self.vsn(x) * self.vsn_gate(x)
        lstm_in     = vsn_out.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)
        h           = attn_out.squeeze(1)
        outputs     = [head(h) for head in self.output_heads]
        stacked     = torch.stack(outputs, dim=1)  # (B, n_horizons, n_quantiles)

        # Enforce P10 <= P50 <= P90 — prevents quantile crossing
        q0     = stacked[:, :, 0:1]
        deltas = torch.relu(stacked[:, :, 1:] - stacked[:, :, :-1])
        return torch.cat([q0, q0 + torch.cumsum(deltas, dim=2)], dim=2)


def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.max(q * err, (q - 1) * err))
    return torch.stack(losses, dim=1).mean()


def combined_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    total = 0.0
    for h in range(pred.shape[1]):
        total += quantile_loss(pred[:, h, :], target[:, h], quantiles)
    return total / pred.shape[1]