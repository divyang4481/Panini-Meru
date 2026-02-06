import argparse
import os
import json
import time
import shutil
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.data.text_stream import TextStreamDataset
from src.pmeru.utils.seed import set_seed
from src.pmeru.utils.logging import get_logger

logger = get_logger(__name__)


def save_metrics(metrics, path):
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--prime_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint folder to resume from",
    )
    args = parser.parse_args()

    set_seed(42)

    # 0. Setup Directories
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # IMPORTANT: Force fp16 mixed precision for 6GB VRAM safety
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    logger.info(f"Starting Run: {args.run_name}")
    logger.info(f"Loading base model: {args.model_name}")

    # Clear cache before start
    torch.cuda.empty_cache()

    # 1. Load Base Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": accelerator.process_index},
        trust_remote_code=True,
    )

    # Enable gradient checkpointing and prepare for kbit training (handles input requires_grad)
    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    base_model = get_peft_model(base_model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Init Prime Stream Components
    hidden_size = base_model.config.hidden_size
    prime_memory = PrimeMemorySeq(hidden_size, args.prime_dim)
    mixer = GateMixer(hidden_size)

    # NOTE: Keeping Prime/Mixer in float32 for stability, wrapper will handle casting if needed.
    # Do NOT cast to float16 here unless we are sure stability is fine.

    # 3. Wrap
    config = PMeruConfig(
        base_model_name=args.model_name,
        hidden_size=hidden_size,
        prime_mem_dim=args.prime_dim,
    )
    model = PMeruModel(config, base_model, prime_memory, mixer)

    # 4. Data
    # Calculate skip_batches if resuming
    skip_batches = 0
    start_step = 0

    # 5. Optimizer
    trainable_params = [
        *filter(lambda p: p.requires_grad, model.base_model.parameters()),
        *model.prime_memory.parameters(),
        *model.mixer.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    # Check resumption BEFORE dataset creation if possible, but we need variables loaded
    # Accelerator load_state loads optimizer and RNG states

    # Prepare first
    # Note: text_stream dataset is an iterable, logic for skipping must be handled carefully.

    model, optimizer = accelerator.prepare(model, optimizer)

    # Determine start step from checkpoint or arg
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)

        # We need to know which step we were at.
        # Usually accelerator saves a 'custom_checkpoint' or we infer from dir name?
        # Standard: user provides "step_X" folder.
        try:
            # Assumes folder name ends with step number or we track it in a meta file
            # Let's check for a training_state.json if we created one, or parse folder name
            dirname = os.path.basename(args.resume_from_checkpoint.rstrip("/\\"))
            if "step_" in dirname:
                step_str = dirname.split("_")[-1]
                start_step = int(step_str)
                logger.info(f"Resuming from step {start_step}")
            else:
                logger.warning(
                    "Could not infer step from checkpoint name. Starting step count at 0 usually wrong for scheduling."
                )
        except:
            pass

        # Update skip_batches
        # skip_batches = start_step * gradient_accumulation_steps * batch_size ??
        # The dataset is iterated by DataLoader. DataLoader yields batches.
        # One valid step = grad_accum * batch iterations.
        # Total batches consumed = start_step * args.gradient_accumulation_steps
        skip_batches = start_step * args.gradient_accumulation_steps

    # Create dataset with skipping
    dataset = TextStreamDataset(
        tokenizer, seq_len=args.seq_len, skip_batches=skip_batches
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader = accelerator.prepare(dataloader)

    logger.info("Starting training...")
    model.train()

    running_loss = 0.0
    global_step = start_step

    update_progress_bar = None
    if accelerator.is_local_main_process:
        progress_bar = tqdm(
            total=args.steps, initial=global_step, desc="Training", unit="step"
        )

    for batch_idx, batch in enumerate(dataloader):
        model.train()
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs["loss"]

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        if accelerator.sync_gradients:
            global_step += 1
            if accelerator.is_local_main_process:
                progress_bar.update(1)

            # Log
            if global_step % 10 == 0:  # Log every 10 steps
                avg_loss = (
                    running_loss / (10 * args.gradient_accumulation_steps)
                    if global_step > start_step
                    else running_loss
                )
                # running_loss accumulates loss.item() (mean of batch).
                # Logic: running_loss += loss.item() (happens every microbatch)
                # In 10 Global Steps, we run 10 * 16 = 160 microbatches.
                # So we should divide by 160.
                avg_loss = running_loss / (10 * args.gradient_accumulation_steps)

                if accelerator.is_local_main_process:
                    progress_bar.set_postfix(loss=avg_loss)

                    save_metrics(
                        {
                            "step": global_step,
                            "loss": avg_loss,
                            "timestamp": time.time(),
                        },
                        metrics_path,
                    )

                running_loss = 0.0

            # Generate Sample (Qualitative Validation)
            if global_step % 50 == 0 and accelerator.is_local_main_process:
                model.eval()
                try:
                    # Simple generation test
                    prompt = "# Function to calculate prime numbers\ndef"
                    inputs = tokenizer(prompt, return_tensors="pt").to(
                        accelerator.device
                    )

                    with torch.no_grad():
                        gen_out = model.base_model.generate(
                            **inputs,
                            max_new_tokens=50,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,
                            temperature=0.7,
                        )
                    decoded = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                    tqdm.write(
                        f"\n=== SAMPLE GENERATION (Step {global_step}) ===\n{decoded}\n==========================================\n"
                    )
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                model.train()

            # Checkpoint
            if global_step % args.checkpoint_every == 0:
                tqdm.write(f"Saving checkpoint at step {global_step}")
                save_path = os.path.join(run_dir, f"step_{global_step}")
                accelerator.save_state(save_path)

            if global_step >= args.steps:
                break

    if accelerator.is_local_main_process:
        progress_bar.close()

    logger.info(f"Training finished. Saving final model to {run_dir}/final")
    accelerator.save_state(os.path.join(run_dir, "final"))

    # Save standalone components for easy inference without accelerator
    if accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_weights_dir = os.path.join(run_dir, "final_components")
        os.makedirs(final_weights_dir, exist_ok=True)

        torch.save(
            unwrapped.prime_memory.state_dict(), f"{final_weights_dir}/prime_memory.pt"
        )
        torch.save(unwrapped.mixer.state_dict(), f"{final_weights_dir}/mixer.pt")
        unwrapped.base_model.save_pretrained(f"{final_weights_dir}/lora_adapter")


if __name__ == "__main__":
    train()
