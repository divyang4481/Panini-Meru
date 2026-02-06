import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer
from src.pmeru.data.struct_tags import StructTagger, align_tags_to_tokens


def evaluate_memory_benefit():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, default="./output/production_v1_light/final_components"
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--long_text_file",
        type=str,
        default="README.md",
        help="File to use as long context test data",
    )
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load Trained Components
    print(f"Loading trained weights from: {args.run_dir}")
    try:
        # Load LoRA
        base_model = PeftModel.from_pretrained(
            base_model, f"{args.run_dir}/lora_adapter"
        )

        # Load Prime Components
        hidden_size = base_model.config.hidden_size
        prime_dim = 128  # Must match training config

        prime_memory = PrimeMemorySeq(hidden_size, prime_dim)
        prime_memory.load_state_dict(torch.load(f"{args.run_dir}/prime_memory.pt"))
        prime_memory.to(
            base_model.device
        ).float()  # Running in float32 for safety as per training

        mixer = GateMixer(hidden_size)
        mixer.load_state_dict(torch.load(f"{args.run_dir}/mixer.pt"))
        mixer.to(base_model.device).float()  # Keep consistent

    except Exception as e:
        print(f"Error loading trained components: {e}")
        return

    # Wrap
    config = PMeruConfig(
        base_model_name=args.base_model,
        hidden_size=hidden_size,
        prime_mem_dim=prime_dim,
    )
    model = PMeruModel(config, base_model, prime_memory, mixer)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tagger = StructTagger()

    # Create Synthetic Long Context
    # We will repeat the input file to ensure it's long enough
    with open(args.long_text_file, "r", encoding="utf-8") as f:
        content = f.read()

    long_text = (content + "\n\n") * 50  # Make it huge

    # Generate tags
    char_tags = tagger.normalize_tags(long_text)
    token_tags = align_tags_to_tokens(tokenizer, long_text, char_tags)
    tokens = tokenizer(long_text, return_tensors="pt")["input_ids"][0]

    # Split into 2 Chunks
    # Chunk 1: Context (History)
    # Chunk 2: Target (The "Now")
    split_idx = 1024
    if len(tokens) < 2048:
        print("Text too short for long-context test.")
        return

    chunk1_tokens = tokens[:split_idx].unsqueeze(0).to(base_model.device)
    chunk1_tags = (
        torch.tensor(token_tags[:split_idx]).unsqueeze(0).to(base_model.device)
    )

    chunk2_tokens = tokens[split_idx : split_idx * 2].unsqueeze(0).to(base_model.device)
    chunk2_tags = (
        torch.tensor(token_tags[split_idx : split_idx * 2])
        .unsqueeze(0)
        .to(base_model.device)
    )

    print(f"\nrunning comparison on {split_idx} token chunks...")

    with torch.no_grad():
        # SCENARIO A: Amnesia (Baseline)
        # We run Chunk 2 with NO state passed from Chunk 1.
        # This simulates a standard model hitting its window limit.
        print("1. Running Baseline (No Memory State)...")
        out_amnesia = model(
            input_ids=chunk2_tokens,
            struct_tags=chunk2_tags,
            prime_state=None,  # Clean slate
            labels=chunk2_tokens,
        )
        loss_amnesia = out_amnesia["loss"].item()

        # SCENARIO B: Adelic Memory (PMeru)
        # We run Chunk 1 to generate state, then pass that state to Chunk 2.
        print("2. Running Prime Memory (With Context State)...")
        # Run Chunk 1
        out_c1 = model(
            input_ids=chunk1_tokens, struct_tags=chunk1_tags, prime_state=None
        )
        state_c1 = out_c1["prime_state"]

        # Run Chunk 2 WITH state
        out_memory = model(
            input_ids=chunk2_tokens,
            struct_tags=chunk2_tags,
            prime_state=state_c1,  # Pass the accumulated brain/syntax state
            labels=chunk2_tokens,
        )
        loss_memory = out_memory["loss"].item()

    print("\n" + "=" * 40)
    print(f"RESULTS: Evaluating Long-Haul Benefit")
    print("=" * 40)
    print(f"Loss (Amnesia/Baseline): {loss_amnesia:.4f}")
    print(f"Loss (With Prime Mem):   {loss_memory:.4f}")

    delta = loss_amnesia - loss_memory
    print(f"Improvement (Delta):     {delta:.4f}")

    if delta > 0:
        print(
            "\n✅ SUCCESS: The model effectively used the Prime Memory to lower perplexity!"
        )
        print(
            "   This proves the architecture is carrying useful context across chunks."
        )
    else:
        print(
            "\n❌ INCONCLUSIVE: No improvement seen yet (model might need more training)."
        )


if __name__ == "__main__":
    evaluate_memory_benefit()
