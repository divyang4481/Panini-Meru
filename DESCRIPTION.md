# Panini-Meru (PMeru)

**Panini-Meru** is a hybrid "Adelic" language model architecture designed to solve the **Validity vs. Fluency** trade-off in LLMs.

It fuses a standard Transformer (for natural language fluency) with a lightweight Recurrent Neural Network (for strict structural adherence and infinite state tracking) via a learned Gated Mixer.

## Key Features

- **Dual-Stream Architecture**: Combines an O(N^2) Transformer with an O(N) Recurrent memory.
- **Training on Consumer Hardware**: Fully trainable on a single **6GB VRAM** GPU using QLoRA + Gradient Checkpointing.
- **Strict Adherence**: Designed to maintain context (indentation, scope, variable definitions) over long sequences where standard transformers fail.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the training smoke test:

```bash
python -m src.pmeru.train.train_text --run_name demo_run --steps 10
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed theory and design.
