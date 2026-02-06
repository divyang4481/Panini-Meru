import torch
import torch.nn as nn


class PrimeMemorySeq(nn.Module):
    """
    Prime Stream Memory Module (v0.1) - The "Bureaucrat"

    This module implements the Recurrent Structural Memory that runs parallel to the Transformer (Real Stream).
    It is responsible for maintaining infinite-context state about structure, scope, and validity tags.

    Architecture:
    - Input: [Hidden States from Transformer] concatenated with [Structural Tag Embeddings]
    - Core: Standard GRU (Gated Recurrent Unit)
    - Output: Memory features projected back to hidden dimensions for mixing

    Why GRU?
    - O(N) linear complexity for sequence length.
    - Captures long-term dependencies (e.g., indentation depth, open brackets) better than
      limited-window attention for "state tracking" tasks.

    Why Float32?
    - RNNs are notoriously unstable in float16 (exploding gradients). We force internal float32 calc.
    """

    def __init__(
        self,
        hidden_size: int,
        prime_dim: int,
        num_struct_tags: int = 32,
        embedding_dim: int = 32,
    ):
        """
        Args:
            hidden_size (int): Dimension of the base Transformer's hidden states (e.g. 1536 for Qwen 1.5B).
            prime_dim (int): Dimension of the GRU hidden state (The "Bureaucrat's" brain capacity).
            num_struct_tags (int): Vocabulary size for structural tags (indent_levels + bracket_types).
            embedding_dim (int): Dimension of tag embeddings.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.prime_dim = prime_dim

        # Structure Tag Embedding
        # Maps integer tags (0=Depth0, 1=Depth1, ...) to vector space
        self.tag_embedding = nn.Embedding(num_struct_tags, embedding_dim)

        # Projection from (Transformer Hidden + Tag Emb) -> GRU Input
        # We compress the massive transformer state down to what the GRU needs
        self.input_proj = nn.Linear(hidden_size + embedding_dim, prime_dim)

        # The Core Recurrent Engine
        self.gru = nn.GRU(input_size=prime_dim, hidden_size=prime_dim, batch_first=True)

        # Projection back to Transformer space for the Mixer
        # Output: [B, T, hidden_size]
        self.output_proj = nn.Linear(prime_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        struct_tags: torch.Tensor,
        state: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (Tensor): [Batch, SeqLen, HiddenSize] - From Real Stream (Transformer)
            struct_tags (Tensor):   [Batch, SeqLen] - Integer IDs of structural info
            state (Tensor, optional): [1, Batch, PrimeDim] - Previous GRU state (for chunked inference)

        Returns:
            memory_out (Tensor): [Batch, SeqLen, HiddenSize] - Features to be mixed back into Transformer
            final_state (Tensor): [1, Batch, PrimeDim] - State to carry over to next chunk
        """
        # === CAST TO FLOAT32 FOR RNN STABILITY ===
        # RNNs in fp16 often produce NaNs. We compute the recurrent step in fp32.
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        if state is not None:
            state = state.float()

        tags_emb = self.tag_embedding(struct_tags).float()

        # 1. Project Input
        # Concatenate Transformer context with explicit Structure Tags
        combined_input = torch.cat([hidden_states, tags_emb], dim=-1)
        gru_input = self.input_proj(combined_input)  # [B, T, PrimeDim]

        # 2. Run Recurrence
        # state is h_{t-1}, gru_out contains h_t for all t
        gru_out, final_state = self.gru(gru_input, state)

        # 3. Project Output
        # Map back to the dimension compatible with the Transformer for mixing
        memory_out = self.output_proj(gru_out)

        # === CAST BACK TO ORIGINAL DTYPE ===
        # Return to fp16/bf16 to match the rest of the model
        memory_out = memory_out.to(dtype=orig_dtype)
        final_state = final_state.to(dtype=orig_dtype)

        return memory_out, final_state
