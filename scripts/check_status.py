import torch
from safetensors.torch import load_file
import os
import glob
import sys
import argparse


def check_checkpoint(run_dir):
    # Find latest checkpoint
    steps = glob.glob(os.path.join(run_dir, "step_*"))
    if not steps:
        print("No checkpoints found yet.")
        return

    latest_step = max(steps, key=os.path.getmtime)
    print(f"Inspecting checkpoint: {latest_step}")

    safetensors_path = os.path.join(latest_step, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print("model.safetensors not found in checkpoint.")
        return

    state_dict = load_file(safetensors_path)

    # Check for Mixer Gate
    # Key usually: "mixer.gate_logit" or "module.mixer.gate_logit" depending on wrapping
    gate_key = next((k for k in state_dict.keys() if "gate_logit" in k), None)

    if gate_key:
        logit = state_dict[gate_key].item()
        sigmoid_val = torch.sigmoid(torch.tensor(logit)).item()
        print(f"\n[Mixer Gate Analysis]")
        print(f"Logit: {logit:.4f}")
        print(f"Weight (0=Memory, 1=Transformer): {sigmoid_val:.4f}")

        prime_influence = (1.0 - sigmoid_val) * 100
        print(f"✅ Prime Memory Influence: {prime_influence:.2f}%")

        if prime_influence < 1.0:
            print("  -> Model is currently ignoring the memory (training early stage).")
        elif prime_influence > 10.0:
            print("  -> Model is HEAVILY relying on Prime memory! Design is working.")
        else:
            print("  -> Model is starting to use memory.")
    else:
        print("Could not find mixer gate in state dict.")

    # Check Prime Memory Weights presence
    prime_keys = [k for k in state_dict.keys() if "prime_memory" in k]
    print(f"\n[Prime Memory]")
    print(f"Found {len(prime_keys)} trainable parameters for Prime Stream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./output/production_v1_light")
    args = parser.parse_args()
    check_checkpoint(args.run_dir)
