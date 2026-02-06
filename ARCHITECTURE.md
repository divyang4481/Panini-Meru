# ARCHITECTURE.md — Project Panini-Meru (PMeru)

This document describes the **production-buildable architecture** for Panini-Meru: a **hybrid Adelic model** that combines a short-window Transformer (“Real Stream”) with a recurrent structural engine (“Prime Stream”) to achieve **long-horizon coherence + rule adherence** on **~6GB VRAM** hardware.

---

## 1) Design goals

### Primary goals

- **Laptop-first training:** Works on **6GB VRAM** via **QLoRA + checkpointing**.
- **Long-horizon coherence without long attention:** Avoid O(N²) attention matrices for structural memory.
- **Syntax + workflow correctness:** Prime stream enforces “legal moves” (valid code-like / business-like sequences).
- **Composable:** Prime stream can be swapped (GRU → S4/Mamba later) without changing the base LLM.

### Non-goals (v0.1)

- Full pretraining of a large Transformer from scratch.
- Custom CUDA kernels (later).
- “Perfect” linguistic parsing; structure tags are _signals_, not ground truth.

---

## 2) Core concept: Adelic dual-stream model

Panini-Meru implements a pragmatic version of an “Adelic” split:

- **Real Stream (R):** continuous semantics, local fluency  
  → implemented as a quantized Transformer base (HF CausalLM)

- **Prime Stream (P):** discrete-ish structure + legality, long-horizon state  
  → implemented as a recurrent state machine (GRU in v0.1)

- **Bridge / Mixer (A):** learns how much structure should influence generation  
  → implemented as a gated residual + layernorm mixer

**Key point:** Prime Stream must be **stateful** (O(1) memory per step / per token chunk).  
It must **not** build a full NxN attention matrix.

---

## 3) High-level architecture

### 3.1 The "Why": Theoretical Benefit of Adelic (O(N) State)

In a pure Transformer, "State" (e.g., _I am currently inside the `generate_user` function_) must be re-derived at every token by attending back to the function definition line.

In Adelic/PMeru:

1.  **Prime Stream** maintains a compact hidden state vector $h_t$ that physically encodes "I am in a function".
2.  At every step, this state is injected into the Transformer via a **Learned Gate**.
3.  **Result**: The Transformer doesn't need to attend back 5,000 tokens to know where it is. It effectively gets "Infinite Short-Term Memory" regarding structure.

| Component     | Stream Name  | Architecture                | Responsibility                                           | Memory Cost          |
| :------------ | :----------- | :-------------------------- | :------------------------------------------------------- | :------------------- |
| **Fluency**   | Real Stream  | **Transformer** (Attention) | Choosing the right _word_. Local grammar. Creativity.    | $O(N^2)$ (Quadratic) |
| **Structure** | Prime Stream | **Recurrent** (GRU/LSTM)    | Choosing the right _type_. Indentation. Scope. Validity. | $O(N)$ (Linear)      |

### 3.2 Component diagram (conceptual)

Tokens ──► Tokenizer ──► Input IDs ──► Real Stream (Transformer, 4-bit)
│
├─► Hidden States h_t (last layer)
│
Structure Tags ──────────────────────┘
│
▼
Prime Stream (Recurrent Memory)
state: S_t (persistent)
output: M_t (memory features)
│
▼
Mixer / Gate
h'\_t = Mix(h_t, M_t)
│
▼
LM Head / Logits
│
▼
Next token

### 3.2 What “long context” means here

- Real Stream sees a **short window** (e.g., 1024–4096 tokens).
- Prime Stream carries “what matters” across chunks as **persistent state** `S_t`.
- Long-horizon generation is done by **chunked decoding** while keeping `S_t`.

---

## 4) Model variants

### 4.1 PMeru-AR (Adelic Renderer) — text model

Purpose: natural language generation with stronger structure adherence.

**Base (Real Stream):**

- Default: `Qwen/Qwen2.5-1.5B` (4-bit NF4)
- Optional: `Llama-3.x-3B` (tight on 6GB; slower, smaller seq)

**Trainable parts:**

- LoRA adapters on Transformer
- PrimeMemory module
- Mixer/Gate module

### 4.2 PMeru-AC (Adelic Compliance) — event model

Purpose: enterprise workflow compliance, “cannot skip steps”.

- Can be a much smaller model (10M–50M params).
- Optimized around event tuples and constraint scoring.
- Often does not need a 1.5B base LLM.

---

## 5) Data interfaces

### 5.1 Inputs for PMeru-AR (Text)

- `input_ids`: `[B, T]`
- `attention_mask`: `[B, T]`
- `struct_tags`: `[B, T]` integer tags aligned to tokens

**struct_tags meaning (v0.1):**
A robust, cheap signal of “where in the structure we are”.
Examples:

- markdown heading depth (# / ## / ###)
- bracket depth in JSON/code
- paragraph/sentence bucket index
- (optional) dependency depth from spaCy (slow)

### 5.2 Inputs for PMeru-AC (Events)

Represent each step as a tuple token:

`event_t = (action_id, state_id, risk_id, actor_id?, channel_id?, region_id?, amount_bucket?)`

Fed as parallel streams or concatenated embedding.

---

## 6) Prime Stream design (v0.1)

### 6.1 PrimeMemory (GRU-based)

PrimeMemory maintains a hidden state `S_t` and produces memory features `M_t`.

- Input: Transformer hidden state `h_t` + normalized `struct_tag_t`
- Update: `S_{t+1} = GRU(Proj([h_t, s_t]), S_t)`
- Output: `M_t = OutProj(S_t)` (same dimensionality as `h_t`)

**Why GRU?**

- Stable, fast, low VRAM.
- Works well as a “rolling editor” / “syntax engine”.

**Future replacements (same interface):**

- S4 / DSS / Mamba-style state space models (better long-range state)
- Slot-based memory (K learned slots updated recurrently)

### 6.2 State persistence

Prime state should persist across:

- chunks in a long sequence (training)
- chunked decoding in evaluation/generation

This is the “infinite context” mechanism: persistent state, not giant attention.

---

## 7) Mixer / Gate

### 7.1 Mixer function

Combine Real hidden states `h` with memory features `M`.

One safe default:

- `h' = LN( sigmoid(g) * h + (1 - sigmoid(g)) * M )`

Where:

- `g` is a learnable scalar or per-channel gate parameter
- LN stabilizes training and prevents memory dominance blowups

### 7.2 Why mixing at hidden states (not token level)

- Mixing at hidden states makes memory influence “semantic planning” earlier.
- Avoids brittle post-hoc constraints that can cause degenerative decoding.

---

## 8) Training architecture (6GB-safe)

### 8.1 Base model loading

- Load Transformer in **4-bit** (NF4).
- Train only LoRA + PrimeMemory + Mixer.

### 8.2 Required memory savers

- `gradient_checkpointing = true`
- micro-batch = 1
- gradient accumulation = 16–64
- fp16 autocast (bf16 if supported and stable)

### 8.3 Sequence lengths (curriculum)

Recommended:

- Stage 1: `T=1024` (stability)
- Stage 2: `T=2048` (after loss stabilizes)
- Stage 3: `T=4096` (only if stable on 6GB)

Long-horizon behavior is measured by _chunking + prime state persistence_.

### 8.4 Losses (v0.1)

**Text LM loss:**

- next-token cross entropy on logits.

**Optional auxiliary losses (v0.2+):**

- structure consistency loss (predict struct*tag*{t+1})
- constraint loss (penalize illegal transitions for structured datasets)
- memory utilization regularizer (prevent memory collapse)

---

## 9) Inference / Generation

### 9.1 Chunked decoding protocol

For long generation beyond base context:

1. Generate `T_chunk` tokens using Real Stream window.
2. Update Prime state `S_t` across those tokens.
3. Slide window (or re-prompt with summary) while keeping `S_t`.
4. Repeat.

The Prime state becomes the long-horizon “thread of truth”.

### 9.2 Modes

- **Creative Mode:** higher reliance on Real Stream (gate biased toward Real).
- **Strict Mode:** higher reliance on Prime stream (gate biased toward Prime).
- **Compliance Mode (Events):** Prime/constraint dominates; Real stream optional.

---

## 10) Evaluation suite (must-have)

### 10.1 Long-haul text tests

Run chunked generation with persistent Prime state; measure:

- **Entity consistency:** names/roles stable
- **Outline adherence:** returns to headings / planned structure
- **Contradiction rate:** fewer self-conflicts after long context
- **Topic return rate:** recovers main topic after deep sub-clauses

### 10.2 Compliance tests (Events)

- **Next-action accuracy**
- **Constraint violation rate** (top metric)
- **Adversarial robustness** (“skip step”, “urgent override”, “bribe”, etc.)

---

## 11) Implementation modules (repo mapping)

### 11.1 Core model code

- `src/pmeru/model/prime_memory.py`
  - PrimeMemoryGRU: forward(h, struct_tags, state) -> (mem, new_state)

- `src/pmeru/model/mixer.py`
  - GateMixer: forward(h, mem) -> h_mix

- `src/pmeru/model/wrapper.py`
  - HF wrapper:
    1. run base transformer with `output_hidden_states=True`
    2. get last hidden states `h`
    3. run PrimeMemory to get `mem` and `state`
    4. mix -> `h_mix`
    5. apply lm_head -> logits
    6. compute loss (train) / sampling (inference)

### 11.2 Data pipeline

- `src/pmeru/data/struct_tags.py`
  - lightweight taggers:
    - markdown depth
    - bracket depth
    - paragraph index
  - optional spaCy tagger behind a flag

- `src/pmeru/data/text_stream.py`
  - streaming datasets
  - tokenization + struct alignment

- `src/pmeru/data/event_tokenizer.py`
  - tuple tokenization for event logs
  - mapping vocab creation and serialization

### 11.3 Training / eval scripts

- `src/pmeru/train/train_text.py`
- `src/pmeru/train/train_events.py`
- `src/pmeru/train/eval_longhaul.py`

---

## 12) Configuration knobs (what matters)

### 12.1 PMeru-AR key knobs

- base_model: `Qwen/Qwen2.5-1.5B`
- load_in_4bit: true
- lora_r: 16 (start)
- lora_alpha: 32
- seq_len: 1024 → 2048
- grad_accum: 32
- lr_lora: 2e-4
- lr_prime: 1e-3 (often okay)
- prime_mem_dim: 128
- struct_tag_source: `lightweight` | `spacy`

### 12.2 PMeru-AC key knobs

- tuple vocab sizes (action/state/risk/etc.)
- constraint rules configuration
- loss weights for constraint penalty

---

## 13) Roadmap upgrades (v0.2+)

### 13.1 Replace GRU with state-space memory

- Benefit: stronger long-range state retention.
- Keep the same interface: `(h, tags, state) -> (mem, new_state)`.

### 13.2 Add memory slots (K)

Instead of per-token memory output:

- maintain K slot vectors updated recurrently
- transformer attends to K slots (small K) — still cheap

### 13.3 Prime streams as multiple “channels”

- p=2 stream: syntax legality / binary rules
- p=3 stream: state progression / phases
- p=5 stream: risk magnitude / anomaly sensitivity
  (Implementation: separate recurrent cells or separated embeddings inside one cell)

---

## 14) Summary

Panini-Meru’s architecture is intentionally simple where it must be:

- **Transformer does language**
- **Recurrent PrimeMemory does long-horizon structure + legality**
- **Mixer decides who wins**
- **Training fits on 6GB via QLoRA + checkpointing**
- **Evaluation focuses on long-horizon coherence + constraint violations**
