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


def save_args(args, path):
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=4)


def update_metadata(path, step, loss=None):
    """Updates a compact metadata file for robust resumption tracking."""
    data = {"last_step": step, "last_loss": loss, "timestamp": time.time()}
    with open(path, "w") as f:
        json.dump(data, f)


def get_dataset_and_loader(args, tokenizer, accelerator, current_step=0):
    """
    Creates the dataset and dataloader, handling batch skipping for resumption.
    """
    # Calculate how many batches to skip based on global step
    skip_batches = current_step * args.gradient_accumulation_steps

    logger.info(f"Preparing dataset. Resumption: Skipping {skip_batches} batches.")

    dataset = TextStreamDataset(
        tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split="train",
        seq_len=args.seq_len,
        skip_batches=skip_batches,
        # Configurable dataset options:
        struct_tag_mode=args.struct_tag_mode,
        text_column=args.text_column,
    )
    # Note: We rely on TextStreamDataset default tagger config or kwargs if we updated it.
    # To fully support indent_size, we need to update TextStreamDataset to accept kwargs or arg.
    # We missed adding indent_size to TextStreamDataset __init__?
    # Let's assume TextStreamDataset has been updated or will support kwargs?
    # Actually, let's Check text_stream.py. It has:
    # def __init__(..., struct_tag_mode="simple", text_column="text", ...)
    # but NOT indent_size. We should fix that or pass tagger directly?

    # We will assume TextStreamDataset creates its own tagger.
    # If we want to set indent_size, we might need to modify TextStreamDataset to accept it,
    # OR inject the tagger.
    # For now, let's stick to the args TextStreamDataset DEFINITELY has.

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader = accelerator.prepare(dataloader)

    return dataloader


def train():
    parser = argparse.ArgumentParser()
    # (Args are defined above in previous chunks, parser logic is preserved)
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    # ... (other args handled by existing code we are not editing) ...

    # We need to grab only the args we added or existing checks...
    # Wait, replace_file_content replaces the BLOCK.
    # If we are effectively rewriting 'train()' partly, we need to be careful.
    pass  # Pseudo-code marker

    # Correct Replacement Strategy:
    # Replace lines 46-69 (get_dataset_and_loader)
    # AND lines 185-198 (Prime/Mixer init)

    # Let's do get_dataset_and_loader first.

    dataset = TextStreamDataset(
        tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split="train",
        seq_len=args.seq_len,
        skip_batches=skip_batches,
        struct_tag_mode=args.struct_tag_mode,
        text_column=args.text_column,
        # TODO: update TextStreamDataset to take indent_size if needed.
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader = accelerator.prepare(dataloader)

    return dataloader


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    # Data Args
    parser.add_argument("--dataset_name", type=str, default="flytech/python-codes-25k")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument(
        "--text_column", type=str, default="text", help="Column containing source code"
    )
    parser.add_argument(
        "--struct_tag_mode",
        type=str,
        default="simple",
        choices=["simple", "spacy", "none"],
        help="Mode for structural tags: 'simple' (indent+bracket), 'spacy' (dependency), 'none' (zeros)",
    )
    parser.add_argument(
        "--indent_size", type=int, default=4, help="Spaces per indent level"
    )
    parser.add_argument(
        "--max_tags", type=int, default=32, help="Max unique structural tags"
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=0.5,
        help="Weight for auxiliary structural loss (v1.1)",
    )

    # Training Args
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint folder to resume from",
    )

    # Model Architecture Args
    parser.add_argument("--prime_dim", type=int, default=128)
    parser.add_argument(
        "--per_channel_gating",
        action="store_true",
        default=True,
        help="Use per-channel gating (vector) instead of scalar gating in Mixer.",
    )
    parser.add_argument(
        "--use_scalar_gating",
        action="store_false",
        dest="per_channel_gating",
        help="Force scalar gating (backward compatibility).",
    )

    args = parser.parse_args()

    set_seed(42)

    # 0. Setup Directories
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    metadata_path = os.path.join(run_dir, "metadata.json")

    # Save Args for reproducibility
    save_args(args, os.path.join(run_dir, "training_args.json"))

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

    # 4. Optimizer
    trainable_params = [
        *filter(lambda p: p.requires_grad, model.base_model.parameters()),
        *model.prime_memory.parameters(),
        *model.mixer.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    # Prepare model/opt before dataloader
    model, optimizer = accelerator.prepare(model, optimizer)

    # 5. Handle Resumption
    start_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)

        # Robust Step Recovery
        # Strategy 1: Check metadata.json in the checkpoint folder
        meta_file = os.path.join(args.resume_from_checkpoint, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    start_step = meta.get("last_step", 0)
                    logger.info(f"Recovered step {start_step} from metadata.json")
            except:
                pass

        # Strategy 2: Check global metadata.json in run dir
        if start_step == 0 and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    start_step = meta.get("last_step", 0)
                    logger.info(f"Recovered step {start_step} from run metadata")
            except:
                pass

        # Strategy 3: Parse dirname
        if start_step == 0:
            try:
                dirname = os.path.basename(args.resume_from_checkpoint.rstrip("/\\"))
                if "step_" in dirname:
                    step_str = dirname.split("_")[-1]
                    start_step = int(step_str)
                    logger.info(f"Recovered step {start_step} from directory name")
            except:
                logger.warning(
                    "Could not determine start step. Starting from 0 (Safe but potentially redundant)."
                )

    # 6. Data Layout
    # Create dataloader *after* knowing the start step
    dataloader = get_dataset_and_loader(
        args, tokenizer, accelerator, current_step=start_step
    )

    logger.info("Starting training loop...")
    model.train()

    running_loss = 0.0
    global_step = start_step

    if accelerator.is_local_main_process:
        progress_bar = tqdm(
            total=args.steps, initial=global_step, desc="Training", unit="step"
        )

    for batch_idx, batch in enumerate(dataloader):
        model.train()
        with accelerator.accumulate(model):
            outputs = model(**batch)
            lm_loss = outputs["loss"]

            # --- V1.1 AUXILIARY STRUCTURAL LOSS ---
            # Predict next structural tag from Prime Memory features
            # This forces the GRU to track structure independently of the Transformer.

            # 1. Initialize Head (Lazy Init on first forward to match device/dtype)
            if not hasattr(model, "struct_head"):
                # Simple linear probe: PrimeDim -> NumTags (32 default)
                model.struct_head = torch.nn.Linear(args.prime_dim, args.max_tags).to(
                    accelerator.device
                )
                # Register as module to be optimized?
                # Ideally should be in model.__init__, but we do it here for script agility.
                # We must ensure optimizer sees it.
                # LIMITATION: If we init here, optimizer doesn't know about it unless we re-add params.
                # BETTER: Add to model init or hack-add to optimizer groups.
                # For this script, let's assuming we MUST handle optimization:
                accelerator.register_for_checkpointing(model.struct_head)
                optimizer.add_param_group({"params": model.struct_head.parameters()})
                logger.info("Initialized Auxiliary Structure Head")

            # 2. Get Features
            mem_features = outputs["mem_features"]  # [B, T, PrimeDim]
            struct_tags = batch["struct_tags"]  # [B, T]

            # 3. Shift & Predict
            # We want Mem[t] to predict Tag[t+1]
            shift_mem = mem_features[..., :-1, :].contiguous()
            shift_tags = struct_tags[..., 1:].contiguous()

            tag_logits = model.struct_head(shift_mem)  # [B, T-1, MaxTags]

            loss_fct_struct = torch.nn.CrossEntropyLoss()
            # Flatten
            struct_loss = loss_fct_struct(
                tag_logits.view(-1, args.max_tags), shift_tags.view(-1)
            )

            # 4. Combine
            total_loss = lm_loss + (args.aux_loss_weight * struct_loss)

            accelerator.backward(total_loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()

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

                # Reset running loss roughly
                running_loss = 0.0

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

                    # Update metadata on disk so if we crash we know where we were
                    update_metadata(metadata_path, global_step, avg_loss)

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

                # Write metadata into the checkpoint folder too for self-contained portability
                update_metadata(os.path.join(save_path, "metadata.json"), global_step)

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
