# BDH Document-Level NLI Pipeline

A complete training and inference pipeline for document-level Natural Language Inference using the Baby Dragon Hatchling (BDH) architecture.

## Task

Given a novel (100k+ words) and a hypothetical character backstory, predict whether the backstory is **consistent** or **contradictory** with the novel.

## Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BDH Document-Level NLI Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Novel   │───▶│ Chunking │───▶│   BDH    │───▶│ Sparse   │───▶│  Cross   │  │
│  │  Text    │    │ (1024ch) │    │ Encoder  │    │ Retrieval│    │ Encoder  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │              │                │                │               │        │
│       │         1092-3860         Embeddings       Top-K=20      Chunk Scores   │
│       │          chunks          + Neurons         Chunks                       │
│       │                                                                │        │
│  ┌──────────┐                                                   ┌──────────┐   │
│  │Backstory │──────────────────────────────────────────────────▶│   MIL    │   │
│  │  Input   │                                                   │Aggregator│   │
│  └──────────┘                                                   └──────────┘   │
│                                                                       │         │
│                                                                       ▼         │
│                                                              ┌──────────────┐   │
│                                                              │  Prediction  │   │
│                                                              │ + Evidence   │   │
│                                                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BDH Encoder (Adapted)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Input: Token IDs (B, T)                                                       │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────┐                                                             │
│   │   Embedding   │ → (B, 1, T, D=256)                                          │
│   └───────────────┘                                                             │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────────────────────────────────────────┐                         │
│   │              BDH Layer (×6)                        │                         │
│   │  ┌─────────────────────────────────────────────┐  │                         │
│   │  │  x_latent = x @ Encoder                     │  │                         │
│   │  │  x_sparse = ReLU(x_latent) ← SPARSE NEURONS │  │                         │
│   │  │  attn_out = LinearAttention(Q=K=x_sparse, V=x)│ │                         │
│   │  │  y_sparse = ReLU(attn_out @ Encoder_V)      │  │                         │
│   │  │  xy_sparse = x_sparse * y_sparse ← GATING   │  │                         │
│   │  │  output = xy_sparse @ Decoder + residual    │  │                         │
│   │  └─────────────────────────────────────────────┘  │                         │
│   └───────────────────────────────────────────────────┘                         │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────┐   ┌───────────────┐                                         │
│   │   Pooling     │   │ Neuron Trace  │                                         │
│   │   (B, D)      │   │ (B, H×N)      │                                         │
│   └───────────────┘   └───────────────┘                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Why BDH? Exploiting Unique Properties

| BDH Property | How We Exploit It |
|--------------|-------------------|
| **Sparse Positive Activations** | ReLU outputs create sparse neuron traces. We use these as semantic fingerprints for retrieval. |
| **Monosemantic Neurons** | Individual neurons activate for specific concepts. Neuron overlap = semantic similarity. |
| **Linear Attention** | Enables stateful chunk processing without memory explosion. State carries across 1000+ chunks. |
| **Hebbian-like Learning** | Co-activation patterns naturally learn consistency/contradiction relationships. |

### Tensor Shapes

```
┌────────────────────────────────────────────────────────┐
│                    Tensor Dimensions                    │
├────────────────────┬───────────────────────────────────┤
│ Component          │ Shape                             │
├────────────────────┼───────────────────────────────────┤
│ Input tokens       │ (B, T=512)                        │
│ Embedded           │ (B, 1, T, D=256)                  │
│ Sparse activations │ (B, H=4, T, N=8192)               │
│ Pooled embedding   │ (B, D=256)                        │
│ Neuron trace       │ (B, H×N=32768)                    │
│ Chunk embeddings   │ (num_chunks, D=256)               │
│ Retrieved indices  │ (top_k=20,)                       │
│ Chunk scores       │ (top_k=20,)                       │
│ Final prediction   │ (1,) ∈ [0, 1]                     │
└────────────────────┴───────────────────────────────────┘
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PREPROCESS                                                   │
│     Novel → Sentence Chunking → ~1000-4000 chunks                │
│     Backstory → Structured Format → Tokenization                 │
│                                                                  │
│  2. ENCODE                                                       │
│     Each chunk → BDH Encoder → (embedding, neuron_trace)        │
│     Backstory  → BDH Encoder → (embedding, neuron_trace)        │
│                                                                  │
│  3. RETRIEVE                                                     │
│     Cosine(backstory_trace, chunk_traces) → Top-20 chunks       │
│                                                                  │
│  4. VERIFY                                                       │
│     For each retrieved chunk:                                    │
│       Score = Verifier(chunk_emb, backstory_emb,                │
│                        chunk_neurons, backstory_neurons)         │
│                                                                  │
│  5. AGGREGATE (MIL)                                              │
│     final_score = max(chunk_scores)  # or attention/noisy-or    │
│     best_chunk = argmax(chunk_scores)                           │
│                                                                  │
│  6. LOSS                                                         │
│     BCE(final_score, label) + λ × MarginLoss(chunk_scores)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Function

$$\mathcal{L} = \mathcal{L}_{BCE} + \lambda_{MIL} \cdot \mathcal{L}_{margin}$$

Where:
- $\mathcal{L}_{BCE} = -[y \log(\hat{p}) + (1-y) \log(1-\hat{p})]$ with class weighting
- $\mathcal{L}_{margin}$: Encourages best chunk to score > 0.5 + margin (consistent) or < 0.5 - margin (contradict)


## Project Structure

```
bdh_nli_pipeline/
├── config/
│   └── default.yaml          # Default configuration
├── data/                     # Data files (CSVs + novels)
│   ├── train - train.csv
│   ├── test - test.csv
│   ├── In search of the castaways.txt
│   └── The Count of Monte Cristo.txt
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Text normalization
│   │   ├── chunking.py       # Novel chunking
│   │   ├── tokenizer.py      # Byte-level tokenizer
│   │   └── dataset.py        # PyTorch datasets
│   ├── models/
│   │   ├── bdh_encoder.py    # Adapted BDH encoder
│   │   ├── retriever.py      # Sparse neuron retriever
│   │   ├── cross_encoder.py  # Verification cross-encoder
│   │   ├── aggregator.py     # MIL aggregator
│   │   └── pipeline.py       # Full pipeline
│   ├── training/
│   │   ├── loss.py           # Loss functions
│   │   └── trainer.py        # Training loop
│   └── inference/
│       ├── predictor.py      # Inference pipeline
│       └── explainability.py # Explanation extraction
├── scripts/
│   ├── train.py              # Training CLI
│   ├── predict.py            # Inference CLI
│   └── test_preprocessing.py # Data preprocessing tests
├── checkpoints/              # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Directory Structure

**All data files are in the `data/` folder inside the project:**

```
bdh_nli_pipeline/
├── data/                                  <-- Default data directory
│   ├── train - train.csv                  <-- Training data (80 rows)
│   ├── test - test.csv                    <-- Test data (60 rows)
│   ├── In search of the castaways.txt     <-- Novel 1 (~826KB)
│   └── The Count of Monte Cristo.txt      <-- Novel 2 (~2.6MB)
├── src/
├── scripts/
├── config/
└── README.md
```

**Expected file names (case-sensitive):**

| File | Expected Name |
|------|---------------|
| Training CSV | `train - train.csv` |
| Test CSV | `test - test.csv` |
| Novel 1 | `In search of the castaways.txt` |
| Novel 2 | `The Count of Monte Cristo.txt` |

**Default:** All paths are configured in `config/default.yaml`.


## Usage

### Configuration

All settings are in `config/default.yaml`:

```yaml
# Key settings in config/default.yaml
paths:
  data_dir: "./data"
  train_csv: "train - train.csv"
  test_csv: "test - test.csv"
  novel_files:
    "In Search of the Castaways": "In search of the castaways.txt"
    "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
  checkpoint_dir: "./checkpoints"

model:
  n_layer: 6
  n_embd: 256
  n_head: 4

training:
  epochs: 20
  learning_rate: 0.001  # Official BDH default
  weight_decay: 0.1
```

### Training

```bash
# Use default config
python scripts/train.py

# Override specific settings
python scripts/train.py --epochs 30 --lr 0.0001

# Use custom config
python scripts/train.py --config config/custom.yaml

# With torch.compile (faster on GPU)
python scripts/train.py --compile --device cuda
```

### Multi-GPU Training (2x H100)

```bash
# Train on 2 GPUs using DDP
torchrun --nproc_per_node=2 scripts/train_multi_gpu.py --config config/h100.yaml --compile

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_multi_gpu.py

# Override settings
torchrun --nproc_per_node=2 scripts/train_multi_gpu.py --epochs 100 --compile
```

**H100 Config (`config/h100.yaml`):**
- `dtype: bfloat16` - Native H100 precision
- `compile: true` - torch.compile for 2-3x speedup
- `top_k: 50` - More chunks (fits in 81GB)
- `epochs: 100` - More training

### Inference

```bash
# Generate predictions
python scripts/predict.py --checkpoint ./checkpoints/best_model.pt

# With explanations
python scripts/predict.py \
    --checkpoint ./checkpoints/best_model.pt \
    --output results.csv \
    --include_explanation
```

## Data Format

### Training Data (train.csv)

| Column | Description |
|--------|-------------|
| id | Unique sample ID |
| book_name | Novel name |
| char | Character name |
| caption | Optional section title |
| content | Backstory text |
| label | consistent / contradict |

### Test Data (test.csv)

Same as training but without `label` column.

### Novels

- `In search of the castaways.txt`
- `The Count of Monte Cristo.txt`

## Output Format (results.csv)

| Column | Description |
|--------|-------------|
| id | Sample ID |
| prediction | consistent / contradict |
| confidence | Probability score |
| evidence_chunk | Index of evidence chunk |
| reasoning | Evidence text (if --include_explanation) |

## Key Components

### BDH Encoder (Official Pathway Implementation)

Based on the official [Pathway BDH repository](https://github.com/pathwaycom/bdh), with adaptations for NLI:

```python
# Core BDH operations (matching official)
x_latent = x @ self.encoder           # Project to neuron space
x_sparse = F.relu(x_latent)           # Sparse positive activations
yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)  # K is Q (official requirement)
y_sparse = F.relu(yKV @ self.encoder_v)
xy_sparse = x_sparse * y_sparse       # Multiplicative gating
output = xy_sparse @ self.decoder     # Decode back
```

**Key features matching official BDH:**
- RoPE (Rotary Position Embedding) with `theta=2^16`
- Causal attention mask using `.tril(diagonal=-1)` for LM
- `K == Q` assertion (BDH self-attention requirement)
- LayerNorm without affine parameters
- `lm_head` for language modeling
- `generate()` method for text generation

**NLI adaptations:**
- Optional bidirectional attention (`use_causal=False`) for encoding
- Neuron trace extraction for retrieval
- Mean pooling for sequence embeddings

### Sparse Neuron Retriever
Uses cosine similarity between neuron activation patterns to find relevant chunks. Exploits BDH's monosemantic property.

### MIL Aggregator
Aggregates chunk-level scores using:
- **max**: Take best chunk (assumes one is sufficient)
- **noisy_or**: Probabilistic "at least one" aggregation
- **attention**: Learned attention-weighted average

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_layer | 6 | BDH layers |
| n_embd | 256 | Embedding dimension |
| n_head | 4 | Attention heads |
| top_k | 20 | Retrieved chunks |
| lr | 1e-4 | Learning rate |
| epochs | 20 | Training epochs |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Pandas
- tqdm
- PyYAML


