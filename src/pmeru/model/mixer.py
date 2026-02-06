import torch
import torch.nn as nn


class GateMixer(nn.Module):
    def __init__(self, hidden_size, per_channel=False):
        """
        Args:
            hidden_size (int): Dimension of hidden states
            per_channel (bool): If True, learns a separate gate for each feature dimension.
                                If False, learns a single scalar gate for the whole vector.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.per_channel = per_channel

        # Learnable gate parameter.
        # Initialized to restricted value to bias towards Real Stream (transformer) initially.
        # sigmoid(2.0) ~= 0.88 (88% Real, 12% Prime)
        if per_channel:
            # One gate logit per feature dimension [H]
            self.gate_logit = nn.Parameter(torch.full((hidden_size,), 2.0))
        else:
            # Single scalar [1]
            self.gate_logit = nn.Parameter(torch.tensor([2.0]))

        # LayerNorm for stability after mixing
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, h, m):
        """
        Args:
            h (Tensor): [B, T, H] - Real Stream (Transformer hidden states)
            m (Tensor): [B, T, H] - Prime Stream (Memory features)

        Returns:
            Tensor: [B, T, H] - Mixed output
        """
        # Compute Gate
        # If scalar: g is [1] -> broadcasts to [B, T, H]
        # If vector: g is [H] -> broadcasts to [B, T, H]
        g = torch.sigmoid(self.gate_logit)

        # Proper broadcasting is handled automatically by PyTorch:
        # If g is [H], logic applies [..., H] correctly against [B, T, H]
        if self.per_channel:
            # Ensure shape align just in case (though [H] usually works against [B,T,H])
            g = g.view(1, 1, -1)

        # Mix: h' = g * h + (1-g) * m
        # High g -> Trust Transformer (Real)
        # Low g -> Trust Memory (Prime)
        mixed = g * h + (1.0 - g) * m

        return self.ln(mixed)
