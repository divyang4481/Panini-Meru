import torch
import torch.nn.functional as F
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
    # We reconstruct the wrapper using the same loaded weights (sharing the base model memory)
    # Note: We need to load adapters for PMeru, but raw_model should be clean.
    # To save VRAM, we will just use the PMeru model and toggle the "gate" to 1.0 (pure transformer) vs Learned.

    # Load Weights
    run_dir = "./output/production_v1_light/final_components"
    prime_dim = 128
    hidden_size = raw_model.config.hidden_size

    prime_memory = PrimeMemorySeq(hidden_size, prime_dim).to("cuda").float()
    mixer = GateMixer(hidden_size).to("cuda").float()

    try:
        prime_memory.load_state_dict(torch.load(f"{run_dir}/prime_memory.pt"))
        mixer.load_state_dict(torch.load(f"{run_dir}/mixer.pt"))
        # Load LoRA adapter
        pmeru_base = PeftModel.from_pretrained(raw_model, f"{run_dir}/lora_adapter")
    except:
        print("Could not load trained weights. Run training first.")
        return

    config = PMeruConfig(
        base_model_name=base_model_name,
        hidden_size=hidden_size,
        prime_mem_dim=prime_dim,
    )
    pmeru_model = PMeruModel(config, pmeru_base, prime_memory, mixer)
    pmeru_model.eval()

    # Define Test Cases
    tests = [
        {
            "name": "Long Range Variable",
            "file": "tests/test_data/long_range_scope.py",
            "prompt_end_line": -1,  # Feed whole file except last line
            "expected_token": "user_request_id",
        },
        {
            "name": "Deep Indentation",
            "file": "tests/test_data/deep_indent.py",
            "prompt_end_line": -1,
            "check_indent": True,
        },
    ]

    tagger = StructTagger()

    print("\n" + "=" * 50)
    print("RUNNING COMPARATIVE BENCHMARK")
    print("=" * 50)

    for test in tests:
        print(f"\nTest: {test['name']}")
        with open(test["file"], "r") as f:
            full_text = f.read()

        # Split prompt vs structure
        # Basic split: feed everything until the final 'print(' or return
        # For this script we'll just slice the text
        lines = full_text.splitlines()
        prompt = "\n".join(lines[:-1])  # Leave out the last line (the answer)
        target = lines[-1].strip()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Tags for PMeru
        char_tags = tagger.normalize_tags(prompt)
        # Simplified token align
        # ... (In full implementation we align properly)
        # Fallback: zeros for now if alignment complexity is high for this script
        struct_tags = torch.zeros_like(inputs["input_ids"])

        with torch.no_grad():
            # Run PMeru
            out = pmeru_model(
                input_ids=inputs["input_ids"], struct_tags=struct_tags, prime_state=None
            )
            logits = out["logits"]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            pred_pmeru = tokenizer.decode(next_token_id)

            # Run Base (Disable memory influence by forcing gate if possible, or just look at raw base outputs?
            # PMeru wrapper calls base model. We can just call pmeru_base.base_model directly)
            # But pmeru_base has LoRA. We want to compare against Trained Code vs Untrained?
            # Or Adelic vs Vanilla?
            # Let's compare Adelic (Memory ON) vs Adelic (Memory OFF / Zero State)

            # Memory OFF run
            # We pass zero state and maybe mask the mixer?
            # Hard to disable mixer without code change.
            # We will use the 'pmeru_base' directly (Pure LoRA Transformer) as the baseline.
            base_out = pmeru_base(input_ids=inputs["input_ids"])
            base_token_id = torch.argmax(base_out.logits[:, -1, :], dim=-1)
            pred_base = tokenizer.decode(base_token_id)

        print(f"Target logic: {target}")
        print(f"Base Model Prediction:  ['{pred_base}'] (ID: {base_token_id.item()})")
        print(f"PMeru Model Prediction: ['{pred_pmeru}'] (ID: {next_token_id.item()})")

        if (
            pred_pmeru.strip() == test.get("expected_token")
            or pred_pmeru.strip() in target
        ):
            print("✅ PMeru Correct!")
        else:
            print("❌ PMeru Incorrect")


if __name__ == "__main__":
    run_comparison()
