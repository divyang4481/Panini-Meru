import torch
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer
from src.pmeru.data.struct_tags import StructTagger, align_tags_to_tokens

"""
eval_longhaul.py
Validates the 'Adelic' architecture's ability to maintain state over long contexts.

Scenario:
1.  Initialize a variable `TARGET_CODE = 999` at the very start.
2.  Generate a massive amount of "noise" code (comments, unrelated functions) exceeding standard context length (e.g., 4000+ tokens).
3.  Throughout the noise, MAINTAIN specific structural depth/state.
4.  At the end, ask the model to `print(TARGET_CODE)`.
5.  Check if it remembers '999' despite it being flushed from the Transformer's attention window.

This works because the Prime Stream's recurrent state should encode "Global Scope contains TARGET_CODE" logic if trained well,
OR (more simply for v1) it tests if the model maintains valid indentation/syntax despite the long haul.
"""


def generate_noise(num_lines=200):
    noise = []
    for i in range(num_lines):
        noise.append(
            f"# Noise line {i}: This is just filler content to push the context window limit."
        )
        noise.append(f"def junk_func_{i}():")
        noise.append(f"    pass  # filler")
        noise.append("")
    return "\n".join(noise)


def encode_chunked(tokenizer, text, chunk_size=1024):
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]


def run_longhaul():
    print("Initializing Long-Haul Evaluation...")

    # 1. Setup Model (Same as benchmark)
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    raw_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load PMeru Artifacts
    run_dir = "./output/smoke_test_v1.1/final_components"
    hidden_size = raw_model.config.hidden_size
    prime_dim = 128  # Match config

    prime_mem = PrimeMemorySeq(hidden_size, prime_dim).cuda().float()
    mixer = GateMixer(hidden_size).cuda().float()

    try:
        if not os.path.exists(f"{run_dir}/prime_memory.pt"):
            print("Skipping LongHaul: No trained model found.")
            print(f"Looked in: {run_dir}")
            return

        prime_mem.load_state_dict(torch.load(f"{run_dir}/prime_memory.pt"))
        mixer.load_state_dict(torch.load(f"{run_dir}/mixer.pt"))
        base_adapter = PeftModel.from_pretrained(raw_model, f"{run_dir}/lora_adapter")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    config = PMeruConfig(base_model_name, hidden_size, prime_dim)
    model = PMeruModel(config, base_adapter, prime_mem, mixer)
    model.eval()

    # 2. Construct The Long Context Probe
    target_value = random.randint(1000, 9999)
    variable_name = "SECRET_KEY_longhaul"

    setup_code = f"{variable_name} = {target_value}\n\n"
    noise_code = generate_noise(
        num_lines=300
    )  # Enough to push > 2k-3k tokens depending on tokenizer
    query_code = f"\n\ndef retrieve_secret():\n    return {variable_name}\n\n# Expected output is the number\nresult = retrieve_secret()\nprint(result)\n# Output: "

    full_text = setup_code + noise_code + query_code

    # Calculate tags for the whole text first (easier than chunking tags)
    tagger = StructTagger()
    char_tags_full = tagger.normalize_tags(full_text)
    token_tags_full = align_tags_to_tokens(tokenizer, full_text, char_tags_full)
    all_input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # 3. Process in Chunks with State Passing
    chunk_size = 1024
    prime_state = None

    print(f"Processing {len(all_input_ids)} tokens in chunks of {chunk_size}...")

    # We iterate until the very last bit, which we want to generate
    # We feed Everything except the very last prediction trigger?
    # Actually, we feed chunks to build up state, then generate continuation.

    # Let's feed everything up to the "Output: " prompt.

    prompt_ids = all_input_ids
    prompt_tags = token_tags_full

    num_chunks = (len(prompt_ids) + chunk_size - 1) // chunk_size

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(prompt_ids))

            chunk_ids = torch.tensor([prompt_ids[start:end]]).cuda()
            chunk_tags = torch.tensor([prompt_tags[start:end]]).cuda()

            # If this is the last chunk, we might want to generate?
            # Impl: Feed chunk, get state.

            out = model(
                input_ids=chunk_ids, struct_tags=chunk_tags, prime_state=prime_state
            )
            prime_state = out["prime_state"]

            print(
                f"  Chunk {i+1}/{num_chunks} processed. State norm: {torch.norm(prime_state).item():.2f}"
            )

        # 4. Generate Answer
        # Now we ask the model to predict next tokens using the Accumulated State
        generated = []
        curr_ids = chunk_ids  # Start from last inputs? No, we need fresh generation inputs (empty? or last token?)
        # Standard generation requires feeding the last token and the state.

        # We'll use the model's base generation loop but we need to inject state manually?
        # HuggingFace `generate` doesn't support our custom `prime_state` easily without wrapper hacking.
        # So we do a manual simple greedy decoding loop here for a few tokens.

        print("Generating response...")
        # Start with the very last token of the prompt to kick off
        last_token = chunk_ids[:, -1:]

        for _ in range(10):  # Generate 10 tokens (the number)
            # We assume tag is 0 for new generated tokens (or we predict tag? Adelic v2...)
            # For now, feed tag=0
            curr_tags = torch.zeros_like(last_token)

            out = model(
                input_ids=last_token, struct_tags=curr_tags, prime_state=prime_state
            )
            logits = out["logits"]
            prime_state = out["prime_state"]

            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated.append(next_token.item())
            last_token = next_token

            if next_token.item() == tokenizer.eos_token_id:
                break

        decoded = tokenizer.decode(generated)
        print(f"\nTarget: {target_value}")
        print(f"Generated: {decoded.strip()}")

        if str(target_value) in decoded:
            print("✅ SUCCESS: Long-Haul Memory Retrieval worked!")
        else:
            print("❌ FAILURE: Retrieval failed.")


if __name__ == "__main__":
    run_longhaul()
