from __future__ import annotations

import torch
from torch import nn


class ChannelWiseLSTM(nn.Module):
    """Channel-wise LSTM without deep supervision from Harutyunyan et al."""

    def __init__(
        self,
        channel_indices: dict[str, list[int]],
        dim: int = 8,
        size_coef: float = 4.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even because channel LSTMs are bidirectional")
        self.channel_names = list(channel_indices)
        self.channel_indices = [list(channel_indices[name]) for name in self.channel_names]
        self.input_dropout = nn.Dropout(dropout)
        self.channel_lstms = nn.ModuleList(
            nn.LSTM(
                input_size=len(indices),
                hidden_size=dim // 2,
                batch_first=True,
                bidirectional=True,
            )
            for indices in self.channel_indices
        )
        main_dim = int(size_coef * dim)
        self.main_lstm = nn.LSTM(
            input_size=dim * len(self.channel_indices),
            hidden_size=main_dim,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(main_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed = []
        for indices, lstm in zip(self.channel_indices, self.channel_lstms):
            channel_x = self.input_dropout(x[:, :, indices])
            channel_output, _ = lstm(channel_x)
            processed.append(channel_output)
        merged = self.input_dropout(torch.cat(processed, dim=-1))
        output, _ = self.main_lstm(merged)
        last = self.output_dropout(output[:, -1, :])
        return self.classifier(last).squeeze(-1)

