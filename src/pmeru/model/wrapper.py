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

        # 3. Aux Structure Head (v1.1)
        # Persistent head for auxiliary loss and structure prediction.
        self.struct_head = nn.Linear(config.prime_mem_dim, config.num_struct_tags)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        struct_tags=None,
        prime_state=None,
        labels=None,
        use_cache=None,
        past_key_values=None,
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
            mem_features: [B, T, PrimeDim] - Raw memory features (for Aux Loss)
        """

        # 1. Run Base Model (Real Stream)
        # We assume output_hidden_states=True gets us the stack
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # DEEP FUSION: Extract Middle Layer
        # Qwen-1.5B has 24 layers. Layer 12 is ~50% depth.
        # hidden_states[0] is embeddings. hidden_states[1] is layer 1 ...
        # So hidden_states[12] is output of 12th block.
        num_layers = len(outputs.hidden_states)
        injection_idx = num_layers // 2  # Adaptive middle

        mid_hidden = outputs.hidden_states[injection_idx]  # [B, T, H]
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        # 2. Run Prime Stream (Memory)
        # Feed MID_HIDDEN to the GRU so it learns from early semantic features.
        if struct_tags is None:
            # Fallback for inference if tags skipped
            struct_tags = torch.zeros_like(input_ids)

        mem_features, new_prime_state, raw_mem_features = self.prime_memory(
            hidden_states=mid_hidden, struct_tags=struct_tags, state=prime_state
        )

        # 3. Mix (Fusion)
        # We mix memory into the LAST hidden state to influence the head
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

        return {
            "loss": loss,
            "logits": logits,
            "prime_state": new_prime_state,
            "mem_features": raw_mem_features,  # Return RAW GRU features (128 dim) for Aux Loss
            "past_key_values": outputs.get("past_key_values", None),
        }

    @torch.no_grad()
    def generate_with_state(
        self,
        input_ids,
        struct_tags=None,
        prime_state=None,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
    ):
        """
        Custom generation loop that maintains Prime Memory state.

        Args:
            input_ids: [B, T] Initial prompt
            struct_tags: [B, T] Initial structure tags
            prime_state: [1, B, PrimeDim] Initial memory state

        Returns:
            full_ids: [B, T + max_new_tokens]
        """
        self.eval()
        curr_ids = input_ids
        curr_state = prime_state
        past_key_values = None

        # If no struct tags provided, use zeros
        if struct_tags is None:
            struct_tags = torch.zeros_like(input_ids)

        curr_tags = struct_tags

        for _ in range(max_new_tokens):
            # Optimization: If we have past_key_values, only process the LAST token
            if past_key_values is not None:
                model_input_ids = curr_ids[:, -1:]
                model_struct_tags = curr_tags[:, -1:]
            else:
                model_input_ids = curr_ids
                model_struct_tags = curr_tags

            # Forward pass
            outputs = self.forward(
                input_ids=model_input_ids,
                struct_tags=model_struct_tags,
                prime_state=curr_state,
                use_cache=True,
                past_key_values=past_key_values,
            )

            next_token_logits = outputs["logits"][:, -1, :]
            curr_state = outputs["prime_state"]  # Update persistent state
            past_key_values = outputs["past_key_values"]  # Update KV Cache

            # Decode
            if do_sample:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Extend tags (naive: repeat 0 or predicted tag? For now 0)
            # Ideally use struct_head to predict next tag!
            # Let's try to use struct_head if available
            next_tag_val = 0
            if hasattr(self, "struct_head"):
                # Predict next tag from memory
                # mem_features is [B, T, PrimeDim]
                mem_feat = outputs["mem_features"][:, -1:, :]  # Last step
                tag_logits = self.struct_head(mem_feat)
                next_tag_val = torch.argmax(tag_logits, dim=-1)
                next_tag = next_tag_val  # [B, 1]
            else:
                next_tag = torch.zeros(
                    (curr_ids.shape[0], 1), dtype=torch.long, device=curr_ids.device
                )

            # Note: For next iteration (if using cache), we append this new tag to curr_tags
            # But the next forward pass will only use this *new* tag slice.
            curr_tags = torch.cat([curr_tags, next_tag], dim=1)

        return curr_ids
