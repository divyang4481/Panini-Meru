import json
import matplotlib.pyplot as plt
import os
import argparse


def plot_metrics(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return

    steps = []
    losses = []
    lm_losses = []
    struct_losses = []

    with open(metrics_path, "r") as f:
        for line in f:
            data = json.loads(line)
            steps.append(data.get("step"))
            losses.append(data.get("loss"))
            # Depending on logging format, we might have these or need to parse differently
            # The current train_text.py logs only "loss" to metrics.jsonl,
            # BUT the progress bar showed "lm" and "auc".
            # Let's check if we logged them.
            # Looking at train_text.py:
            # save_metrics({"step": global_step, "loss": avg_loss, ...})
            # It seems we ONLY saved "loss" to the jsonl file in the current script version!
            # We can only plot total loss.

    if not steps:
        print("Empty metrics file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Total Loss", linewidth=2)
    plt.title(f"Training Progress: {os.path.basename(run_dir)}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    output_img = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(output_img)
    print(f"Analysis saved to {output_img}")
    print(f"Final Loss: {losses[-1]:.4f} at Step {steps[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./output/v1.1_final")
    args = parser.parse_args()

    plot_metrics(args.run_dir)
