import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint folder (step_X)",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--prompt",
        type=str,
        default="# Write a function to calculate fibonacci\ndef fib(n):",
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

    # Load LoRA
    # Check if checkpoint has 'lora_adapter' submodule or is flat
    # Our script saves full state via accelerator.save_state which saves:
    # - pytorch_model.bin (optimizer etc)
    # - custom_checkpoint/ (if configured)
    # Actually accelerator.save_state saves model.safetensors if unwrapped?
    # No, we implemented custom saving in train_text.py:
    #   if global_step % args.checkpoint_every == 0: accelerator.save_state(save_path)
    # This saves the whole accelerator state. Loading it without accelerator is tricky.

    # BUT we also implemented:
    #   Training finished. Saving final model to {run_dir}/final
    #   AND final_components/

    # For intermediate checkpoints, we need to load via accelerator or manual state dict.
    # To keep it simple, this script assumes loading from the 'final_components' style structure
    # OR the user runs training to completion.

    # Let's try to load from 'final_components' structure primarily.

    if "final_components" in args.checkpoint:
        adapter_path = f"{args.checkpoint}/lora_adapter"
        print(f"Loading LoRA from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(
            "Warning: Intermediate checkpoints from Accelerator are complex to load individually in this simple script."
        )
        print(
            "Please point to 'final_components' folder, or use this script only after training finishes."
        )
        # Try to load as PEFT usually does if folder contains adapter_config.json
        try:
            model = PeftModel.from_pretrained(base_model, args.checkpoint)
        except:
            print(
                "Could not standard load PEFT. Continuing with base model only for test."
            )
            model = base_model

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\n=== OUTPUT ===\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n==============")


if __name__ == "__main__":
    generate()
