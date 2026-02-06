import torch
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer
from transformers import AutoModelForCausalLM, AutoConfig


def verify_structure():
    print("Initializing dummy model...")
    # minimal config
    config = PMeruConfig(
        base_model_name="gpt2",  # smaller for test
        hidden_size=768,
        prime_mem_dim=128,
        num_struct_tags=40,
    )

    # Fake base model
    base_config = AutoConfig.from_pretrained("gpt2")
    base_model = AutoModelForCausalLM.from_config(base_config)

    prime = PrimeMemorySeq(768, 128, 40)
    mixer = GateMixer(768)

    model = PMeruModel(config, base_model, prime, mixer)

    print("Checking for struct_head...")
    if hasattr(model, "struct_head"):
        print("PASS: struct_head exists.")
        print(f"struct_head: {model.struct_head}")
    else:
        print("FAIL: struct_head MISSING from model instance.")
        exit(1)

    print("Checking Forward Pass signature...")
    import inspect

    sig = inspect.signature(model.forward)
    print(f"Forward Params: {sig.parameters.keys()}")


if __name__ == "__main__":
    verify_structure()
