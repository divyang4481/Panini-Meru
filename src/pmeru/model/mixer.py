import torch
import torch.nn as nn


class GateMixer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Learnable gate parameter.
        # Can be scalar or per-channel vector. Let's do scalar first for v0.1.
        # Initialized to restricted value to bias towards Real Stream (transformer) initially.
        self.gate_logit = nn.Parameter(
            torch.tensor([2.0])
        )  # sigmoid(2.0) ~= 0.88 (88% Real, 12% Prime)

        # LayerNorm for stability after mixing
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, h, m):
        """
        h: [B, T, H] - Real Stream (Transformer hidden states)
        m: [B, T, H] - Prime Stream (Memory features)
        """
        g = torch.sigmoid(self.gate_logit)

        # Mix
        # h' = g * h + (1-g) * m
        # or h' = h + g * m (residual) -- but let's follow the architecture doc
        # which says: LN( sigmoid(g) * h + (1 - sigmoid(g)) * M )

        mixed = g * h + (1.0 - g) * m

        return self.ln(mixed)
