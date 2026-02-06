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
    def __init__(self, config: PMeruConfig, base_model, prime_memory, mixer):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.prime_memory = prime_memory
        self.mixer = mixer

        # v1.1 NEW: Structure Head for Aux Loss
        self.struct_head = nn.Linear(config.prime_mem_dim, config.num_struct_tags)

        # Cache Layers
        self.lm_head = self.base_model.get_output_embeddings()
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
        else:
            self.final_norm = getattr(self.base_model, "norm", None)

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
        # 1. Run Base Model with Hidden States (Deep Fusion)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

        # v1.1 NEW: Deep Fusion (Extract Middle Layer)
        # We feed the raw, early semantics to the memory
        mid_idx = len(outputs.hidden_states) // 2
        mid_hidden = outputs.hidden_states[mid_idx]  # [B, T, H]
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        # 2. Run Prime Stream (Memory) using MID_HIDDEN
        if struct_tags is None:
            struct_tags = torch.zeros_like(input_ids)

        mem_features, new_prime_state, raw_mem_features = self.prime_memory(
            hidden_states=mid_hidden,  # <--- Changed from last_hidden
            struct_tags=struct_tags,
            state=prime_state,
        )

        # 3. Mix into Final Layer
        mixed_hidden = self.mixer(h=last_hidden, m=mem_features)

        # 4. Norm & Head
        target_dtype = self.lm_head.weight.dtype
        if self.final_norm:
            mixed_normed = self.final_norm(mixed_hidden).to(target_dtype)
        else:
            mixed_normed = mixed_hidden.to(target_dtype)

        logits = self.lm_head(mixed_normed)

        # 5. Calculate Losses
        loss = None
        lm_loss = None
        struct_loss = None

        if labels is not None:
            # LM Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # v1.1 NEW: Aux Structure Loss
            # Force memory to predict NEXT tag
            shift_mem = raw_mem_features[..., :-1, :].contiguous()
            shift_tags = struct_tags[..., 1:].contiguous()

            # Predict
            tag_logits = self.struct_head(shift_mem)
            struct_loss = loss_fct(
                tag_logits.view(-1, self.config.num_struct_tags), shift_tags.view(-1)
            )

            # Combined Loss (0.5 weight for structure)
            loss = lm_loss + 0.5 * struct_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "struct_loss": struct_loss,
            "logits": logits,
            "prime_state": new_prime_state,
            "mem_features": raw_mem_features,
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

            curr_tags = torch.cat([curr_tags, next_tag], dim=1)

        return curr_ids
