import argparse
import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

from src.pmeru.model.prime_memory import PrimeMemorySeq
from src.pmeru.model.mixer import GateMixer
from src.pmeru.model.wrapper import PMeruModel, PMeruConfig
from src.pmeru.data.event_stream import EventStreamDataset, SyntheticEventConfig
from src.pmeru.utils.seed import set_seed
from src.pmeru.utils.logging import get_logger

logger = get_logger(__name__)


def train_events():
    parser = argparse.ArgumentParser(
        description="Train Adelic Model on Enterprise Event Logs"
    )
    parser.add_argument("--run_name", type=str, default="event_run_v1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    # Event Config
    parser.add_argument(
        "--num_events",
        type=int,
        default=50000,
        help="Number of synthetic events to stream",
    )

    # Model Args
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prime_dim", type=int, default=128)
    parser.add_argument("--per_channel_gating", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./output_events")

    args = parser.parse_args()
    set_seed(42)

    # Setup
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=16)
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Base Model & Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": accelerator.process_index},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # LoRA
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    )
    model_base = get_peft_model(model_base, peft_config)

    # 2. Adelic Components
    hidden_size = model_base.config.hidden_size
    prime_mem = PrimeMemorySeq(hidden_size, args.prime_dim).cuda()
    mixer = GateMixer(hidden_size, per_channel=args.per_channel_gating).cuda()

    config = PMeruConfig(args.model_name, hidden_size, args.prime_dim)
    model = PMeruModel(config, model_base, prime_mem, mixer)

    # 3. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    # 4. Data
    event_config = SyntheticEventConfig(num_events=args.num_events)
    dataset = EventStreamDataset(tokenizer, event_config, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 5. Loop
    model.train()
    global_step = 0
    progress = tqdm(total=args.steps, desc="Event Training")

    for batch in dataloader:
        with accelerator.accumulate(model):
            out = model(**batch)
            lm_loss = out["loss"]

            # --- V1.1 AUX Loss ---
            # 1. Aux Head initialized in model.__init__

            mem_features = out["mem_features"]
            struct_tags = batch["struct_tags"]

            shift_mem = mem_features[..., :-1, :].contiguous()
            shift_tags = struct_tags[..., 1:].contiguous()

            tag_logits = model.struct_head(shift_mem)
            loss_fct_struct = torch.nn.CrossEntropyLoss()

            struct_loss = loss_fct_struct(tag_logits.view(-1, 32), shift_tags.view(-1))

            # Loss Weight 0.5 hardcoded for events for now
            total_loss = lm_loss + (0.5 * struct_loss)

            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=total_loss.item())

            if global_step >= args.steps:
                break

    # Save
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.prime_memory.state_dict(), f"{run_dir}/prime_memory.pt")
        print(f"Saved event model to {run_dir}")


if __name__ == "__main__":
    train_events()
