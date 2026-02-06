import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer
from src.pmeru.data.struct_tags import StructTagger, align_tags_to_tokens


def run_comparison():
    # Load Models
    print("Loading Models...")

    # 1. Base Model (The "Before")
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    raw_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2. PMeru Model (The "After")
    # Load Weights
    run_dir = "./output/production_v1_light/final_components"
    prime_dim = 128
    hidden_size = raw_model.config.hidden_size

    prime_memory = PrimeMemorySeq(hidden_size, prime_dim).to("cuda").float()

    # NOTE: Mixer output might be float32, so we ensure it's compatible
    mixer = GateMixer(hidden_size).to("cuda").float()

    try:
        # Check if files exist
        if not os.path.exists(f"{run_dir}/prime_memory.pt"):
            raise FileNotFoundError("Training artifacts not found.")

        prime_memory.load_state_dict(torch.load(f"{run_dir}/prime_memory.pt"))
        mixer.load_state_dict(torch.load(f"{run_dir}/mixer.pt"))
        # Load LoRA adapter
        pmeru_base = PeftModel.from_pretrained(raw_model, f"{run_dir}/lora_adapter")
    except Exception as e:
        print(f"Could not load trained weights: {e}")
        print("Please run training first using: python src/pmeru/train/train_text.py")
        return

    config = PMeruConfig(
        base_model_name=base_model_name,
        hidden_size=hidden_size,
        prime_mem_dim=prime_dim,
    )
    pmeru_model = PMeruModel(config, pmeru_base, prime_memory, mixer)
    pmeru_model.eval()

    # Define Test Cases
    # Fallback content in case file is missing
    fallback_long_range = """
user_request_id = "req_12345"
def process():
    # ... many lines ...
    return user_request_id
"""

    tests = [
        {
            "name": "Long Range Variable",
            "file": "tests/test_data/long_range_scope.py",
            "fallback": fallback_long_range,
            "expected_token": "user_request_id",
        }
    ]

    tagger = StructTagger()

    print("\n" + "=" * 50)
    print("RUNNING COMPARATIVE BENCHMARK")
    print("=" * 50)

    for test in tests:
        print(f"\nTest: {test['name']}")

        full_text = ""
        if os.path.exists(test["file"]):
            with open(test["file"], "r") as f:
                full_text = f.read()
        else:
            print(f"Warning: Test file {test['file']} not found. Using fallback.")
            full_text = test["fallback"].strip()

        # Split prompt vs structure
        lines = full_text.splitlines()
        prompt = "\n".join(lines[:-1])  # Leave out the last line (the answer)
        target = lines[-1].strip()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Tags for PMeru - Real Calculation
        char_tags = tagger.normalize_tags(prompt)

        # Align
        try:
            token_tags_list = align_tags_to_tokens(tokenizer, prompt, char_tags)
            struct_tags = torch.tensor([token_tags_list], device="cuda")
        except Exception as e:
            print(f"Tag alignment failed: {e}. using zeros.")
            struct_tags = torch.zeros_like(inputs["input_ids"])

        with torch.no_grad():
            # Run PMeru
            out = pmeru_model(
                input_ids=inputs["input_ids"], struct_tags=struct_tags, prime_state=None
            )
            logits = out["logits"]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            pred_pmeru = tokenizer.decode(next_token_id)

            # Run Base (LoRA only, no Memory)
            # We call the wrapped base model directly
            base_out = pmeru_base(input_ids=inputs["input_ids"])
            base_token_id = torch.argmax(base_out.logits[:, -1, :], dim=-1)
            pred_base = tokenizer.decode(base_token_id)

        print(f"Target logic: {target}")
        print(f"Base Model Prediction:  ['{pred_base}'] (ID: {base_token_id.item()})")
        print(f"PMeru Model Prediction: ['{pred_pmeru}'] (ID: {next_token_id.item()})")

        # Simple check
        if test.get("expected_token") and test.get("expected_token") in pred_pmeru:
            print("✅ PMeru Correct!")
        else:
            print("❓ PMeru Output differs.")


if __name__ == "__main__":
    run_comparison()
