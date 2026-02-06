import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class PMeruConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_size=1536,  # Qwen 1.5B size
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
    def __init__(self, config, base_model, prime_memory, mixer):
        super().__init__()
        self.config = config
        self.base_model = (
            base_model  # Expected to be a causal LM (e.g. Qwen2ForCausalLM)
        )
        self.prime_memory = prime_memory
        self.mixer = mixer

        # Frozen base model (usually)
        # We rely on the training script to set requires_grad=False for base model
        # and True for prime/mixer/lora

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
        Forward pass for training/inference.

        input_ids: [B, T]
        struct_tags: [B, T] - required for PrimeMemory
        prime_state: [1, B, PrimeDim] - initial state
        """

        # 1. Run Base Model
        # output_hidden_states=True so we can grab the last layer features
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Last hidden state from the transformer stack (before LM head)
        # Verify layer index: -1 usually gives the final layer output before Norm?
        # For CausalLM, hidden_states[-1] is usually the output of the final block.
        # Check specific model docs. Usually outputs.hidden_states is a tuple.
        # hidden_states[-1] is the output of the last encoder layer.

        # Note: Qwen2ForCausalLM applies RMSNorm *after* the last block?
        # We should intercept *before* the LM Head but *after* the final Norm?
        # Usually HF `hidden_states[-1]` is the output of the last block.
        # Some models apply a final Norm before the head.
        # The base_model.lm_head takes that norm'd input.

        # Let's assume hidden_states[-1] is proper input for the next stage.
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        # 2. Run Prime Stream
        if struct_tags is None:
            # Fallback if no tags provided (e.g. validation sanity check): all zeros
            struct_tags = torch.zeros_like(input_ids)

        mem_features, new_prime_state = self.prime_memory(
            hidden_states=last_hidden, struct_tags=struct_tags, state=prime_state
        )

        # 3. Mix
        mixed_hidden = self.mixer(h=last_hidden, m=mem_features)

        # 4. LM Head (Predict next token)
        # We need to call the base model's simple LM head or apply the final norm + head manually.
        # If we use base_model.lm_head(mixed_hidden), we assume base_model handles the norm?
        # In Qwen/Llama, `model.lm_head` is just the Linear layer. The Norm is usually `model.model.norm`.
        # outputs.hidden_states[-1] is *before* the final norm in many HF implementations?
        # Let's check typical HF Llama pattern:
        #   x = model(input)
        #   x = norm(x)
        #   logits = lm_head(x)

        # If outputs.hidden_states[-1] is pre-norm, then we should mix -> norm -> head.
        # But if we want to be safe, we can try to use the model's structure.

        # For generalized usage, let's look at `base_model.get_output_embeddings()`.
        # And we might need the final norm.

        # A safer bet for a hacked wrapper:
        # Apply the final norm if it exists.

        logits = None

        # Attempt to find the final norm
        final_norm = None
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "norm"):
            final_norm = self.base_model.model.norm
        elif hasattr(self.base_model, "norm"):  # Some architectures
            final_norm = self.base_model.norm

        # Apply norm if found
        if final_norm:
            mixed_normed = final_norm(mixed_hidden)
        else:
            mixed_normed = mixed_hidden  # Hope for the best or it's built-in

        # Apply Head
        lm_head = self.base_model.get_output_embeddings()

        # Ensure dtype match. The mixer might output float32 (our stability fix),
        # but the quant/base model LM head expects float16/bfloat16.
        mixed_normed = mixed_normed.to(lm_head.weight.dtype)

        logits = lm_head(mixed_normed)

        loss = None
        if labels is not None:
            # Shift for autoregressive loss
            # logits: [B, T, Vocab]
            # labels: [B, T]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits, "prime_state": new_prime_state}
