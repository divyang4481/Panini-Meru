import torch
import torch.nn as nn


class PrimeMemoryGRU(nn.Module):
    def __init__(self, hidden_size, prime_mem_dim, num_struct_tags=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.prime_mem_dim = prime_mem_dim

        # Structure tag embedding (optional, small dim)
        self.tag_dim = 16
        self.tag_embedding = nn.Embedding(num_struct_tags, self.tag_dim)

        # Input projection: projects Transformer hidden state + tag embedding -> GRU input size
        # We'll make the GRU input size equal to prime_mem_dim for simplicity
        self.input_proj = nn.Linear(hidden_size + self.tag_dim, prime_mem_dim)

        # The GRU cell
        # input_size=prime_mem_dim, hidden_size=prime_mem_dim
        self.gru = nn.GRUCell(prime_mem_dim, prime_mem_dim)

        # Output projection: projects GRU state -> Transformer hidden size
        self.output_proj = nn.Linear(prime_mem_dim, hidden_size)

    def forward(self, hidden_states, struct_tags, state=None):
        """
        hidden_states: [Batch, SeqLen, HiddenSize]
        struct_tags: [Batch, SeqLen] (integers)
        state: [Batch, PrimeMemDim] (previous state), or None

        Returns:
            memory_features: [Batch, SeqLen, HiddenSize]
            new_state: [Batch, PrimeMemDim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if state is None:
            state = torch.zeros(
                batch_size,
                self.prime_mem_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Embed tags: [B, T, TagDim]
        tags_emb = self.tag_embedding(struct_tags)

        # Concatenate: [B, T, Hidden + TagDim]
        combined_input = torch.cat([hidden_states, tags_emb], dim=-1)

        # Project inputs: [B, T, PrimeMemDim]
        gru_inputs = self.input_proj(combined_input)

        # Run GRU over sequence
        # We need to loop because standard GRU layer returns all steps, but we need to verify if we want step-by-step for custom logic
        # Ideally, we use nn.GRU if we can, but since we are doing simple integration, let's just use nn.GRU for the whole sequence at once.
        # But wait, nn.GRU(batch_first=True) takes (B, T, Input) and h_0 (1, B, Hidden)

        # Reshape state for GRU layer: [1, Batch, PrimeMemDim]
        # But verify dtype - GRU output might be different if we use mixed precision

        # Let's use loop for clarity in v0.1 or nn.GRU for speed?
        # nn.GRU is much faster (cuDNN). Let's use a temporary nn.GRU module instead of GRUCell loop if possible.
        # However, the class definition has `self.gru = nn.GRUCell`.
        # I will change it to nn.GRU for efficient processing over the sequence.

        pass

    # Re-defining __init__ to use nn.GRU for sequence processing efficiency


class PrimeMemorySeq(nn.Module):
    def __init__(self, hidden_size, prime_mem_dim, num_struct_tags=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.prime_mem_dim = prime_mem_dim

        self.tag_dim = 16
        self.tag_embedding = nn.Embedding(num_struct_tags, self.tag_dim)

        self.input_proj = nn.Linear(hidden_size + self.tag_dim, prime_mem_dim)

        self.gru = nn.GRU(
            input_size=prime_mem_dim, hidden_size=prime_mem_dim, batch_first=True
        )

        self.output_proj = nn.Linear(prime_mem_dim, hidden_size)

    def forward(self, hidden_states, struct_tags, state=None):
        """
        Args:
            hidden_states: [B, T, H]
            struct_tags: [B, T]
            state: [1, B, PrimeDim] or None. Note: GRU expects [NumLayers, B, Hidden]

        Returns:
            memory_out: [B, T, H]
            final_state: [1, B, PrimeDim]
        """
        if state is None:
            # Let GRU initialize it to 0
            pass

        # === CAST TO FLOAT32 FOR RNN STABILITY ===
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if state is not None:
            state = state.float()

        tags_emb = self.tag_embedding(struct_tags).float()

        # [B, T, H+Tag]
        combined = torch.cat([hidden_states, tags_emb], dim=-1)

        # [B, T, PrimeDim]
        # We project down before the GRU to keep GRU light
        gru_input = self.input_proj(combined)

        # Run GRU
        # gru_out: [B, T, PrimeDim]
        # final_state: [1, B, PrimeDim]
        gru_out, final_state = self.gru(gru_input, state)

        # Project back up
        memory_out = self.output_proj(gru_out)

        # === CAST BACK TO ORIGINAL DTYPE ===
        memory_out = memory_out.to(dtype=orig_dtype)

        # We generally want to keep state in float32 for next step precision, but caller might expect orig_dtype
        # Let's return it in orig_dtype to match signature, user must cast if carrying over manually
        final_state = final_state.to(dtype=orig_dtype)

        return memory_out, final_state
