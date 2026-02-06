import torch
import torch.nn as nn
from transformers import PretrainedConfig


class PMeruConfig(PretrainedConfig):
    """
    Configuration for Panini-Meru (PMeru) Architectures.

    Args:
        base_model_name (str): HF ID of the base transformer (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
        hidden_size (int): Hidden dimension size of the base model.
        prime_mem_dim (int): Hidden dimension size of the Prime Memory (GRU).
        num_struct_tags (int): Vocabulary size for structural tags.
    """

    def __init__(
        self,
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_size=1536,  # Qwen 1.5B default
        prime_mem_dim=128,
        num_struct_tags=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.prime_mem_dim = prime_mem_dim
        self.num_struct_tags = num_struct_tags


class PMeruModel(nn.Module):
    """
    PMeru Wrapper - Fuses the Real Stream (Transformer) with Prime Stream (Memory).

    Architecture Flow:
    1. Base Model Forward -> Hidden States (Pre-Norm)
    2. Prime Memory Forward -> Memory Features + New State
    3. Gated Mixing -> Mixed Hidden States
    4. Base Model Final Norm (Cached) -> Normed States
    5. Base Model LM Head (Cached) -> Logits
    """

    def __init__(self, config: PMeruConfig, base_model, prime_memory, mixer):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # External components
        self.prime_memory = prime_memory
        self.mixer = mixer

        # --- CACHE LAYERS ---
        # Instead of searching every forward pass, identify them once.
        # 1. Output Head
        self.lm_head = self.base_model.get_output_embeddings()

        # 2. Final Norm
        # Different architectures name it differently.
        self.final_norm = None

        # Common patterns:
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "norm"):
            # Llama / Qwen pattern
            self.final_norm = self.base_model.model.norm
        elif hasattr(self.base_model, "norm"):
            # Some architectures
            self.final_norm = self.base_model.norm
        elif hasattr(self.base_model, "transformer") and hasattr(
            self.base_model.transformer, "ln_f"
        ):
            # GPT-2 / older style
            self.final_norm = self.base_model.transformer.ln_f

        if self.final_norm is None:
            print(
                "WARNING: Could not identify final normalization layer. Mixing might occur post-norm or require manual handling."
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        struct_tags=None,
        prime_state=None,
        labels=None,
        **kwargs
    ):
        """
        Args:
            input_ids: [B, T] - Token IDs
            struct_tags: [B, T] - Structural Tags (for Prime Stream)
            prime_state: [1, B, PrimeDim] - Initial state for the Memory (optional)
            labels: [B, T] - Next token labels for training

        Returns dict:
            loss (optional): CrossEntropy loss if labels provided
            logits: [B, T, VocabSize]
            prime_state: [1, B, PrimeDim] - Final state for next chunk
        """

        # 1. Run Base Model (Real Stream)
        # We assume output_hidden_states=True gets us the stack
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Get the output state of the last block (usually pre-final-norm)
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        # 2. Run Prime Stream (Memory)
        if struct_tags is None:
            # Fallback for inference if tags skipped
            struct_tags = torch.zeros_like(input_ids)

        mem_features, new_prime_state = self.prime_memory(
            hidden_states=last_hidden, struct_tags=struct_tags, state=prime_state
        )

        # 3. Mix (Fusion)
        mixed_hidden = self.mixer(h=last_hidden, m=mem_features)

        # 4. Final Norm & Head
        # Apply strict type matching because Mixer might be float32
        target_dtype = self.lm_head.weight.dtype

        if self.final_norm:
            mixed_normed = self.final_norm(mixed_hidden)
        else:
            mixed_normed = mixed_hidden

        # Ensure we are ready for the linear head
        mixed_normed = mixed_normed.to(target_dtype)

        logits = self.lm_head(mixed_normed)

        loss = None
        if labels is not None:
            # Shift for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits, "prime_state": new_prime_state}
