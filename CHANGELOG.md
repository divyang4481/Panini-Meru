# Changelog

All notable changes to the **Panini-Meru** project will be documented in this file.

## [v1.1.0] - 2026-02-06

### 🚀 Major Features

- **Deep Fusion Architecture (`wrapper.py`)**:
  - Integrated Prime Memory injection at middle layers (Layer 12 for Qwen-1.5B) instead of just the input.
  - Implemented `GateMixer` with per-channel gating for bidirectional context flow.
- **Auxiliary Structural Loss (`train_text.py`)**:
  - Added a persistent `struct_head` to `PMeruModel` that predicts structural tags (indentation/risk) from the Prime Stream.
  - This forces the Prime Stream to remain active and theoretically grounded, preventing the "Lazy Bureaucrat" issue.
- **Event Compliance Pipeline (`train_events.py`)**:
  - New training script for structured enterprise logs.
  - Supports synthetic event generation and risk-based tagging.
- **Composite Event Tags**:
  - `EventTokenizer` encodes (Risk \* 10 + ActionHash), enabling the Prime Stream to learn complex workflow sequences.
- **Long-Haul Evaluation**:
  - Added `eval_longhaul.py` to test context retention beyond the transformer's sliding window.
  - Added `model.generate_with_state()` API to support custom generation loops with persistent memory.

### ⚡ Improvements

- **Configurability**:
  - `TextStreamDataset` now accepts `--indent_size` from CLI (default: 4).
  - Added `struct_tag_mode` options ("simple", "spacy", "none").
- **Persistence**:
  - Moved `struct_head` initialization to `PMeruModel.__init__` to ensure it is always saved in checkpoints and properly optimized.
- **Optimization**:
  - Removed fragile "lazy initialization" logic from training loops.
  - Added `per_channel_gating` support for finer-grained control over memory mixing.

### 🐛 Bug Fixes

- Fixed `NameError` in `train_events.py` logging loop.
- Removed redundant/dead code blocks in `train_text.py`.
- Fixed indentation handling in `StructTagger` for various coding styles.

### 🔮 Future Work (v1.2)

- **KV-Cache Optimization**: Update `generate_with_state` to use the Key-Value cache for faster inference.
- **Advanced Gating**: Explore dynamic layer selection for injection.
