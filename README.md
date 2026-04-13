# IMDb Transformers: BERT vs GPT Comparison

**A beginner-friendly project to understand encoder vs decoder transformers using the same dataset.**

This project fine-tunes two pretrained transformer families on IMDb movie reviews to show their fundamental differences:

| Model Family | Architecture | Task | Output |
|--------------|--------------|------|--------|
| **BERT (Encoder)** | `distilbert-base-uncased` | Sentiment Classification | positive / negative |
| **GPT (Decoder)** | `distilgpt2` | Causal Language Modeling | "this movie was..." → continuation |

**Same data → different model families → different objectives → different outputs.**

## 🎯 Why This Project?

**The goal is simple: observe how encoder and decoder transformers behave differently on the same domain.**

- **BERT excels at representation learning** → perfect for classification tasks
- **GPT excels at next-token prediction** → perfect for text generation
- **Same IMDb dataset** isolates the architecture difference
- **Pretrained + fine-tuned** = realistic model behavior, not toy implementations

**What you'll learn:**
Encoder (BERT): "Is this review positive or negative?" → Single label output
Decoder (GPT): "Continue this review..." → Text generation


## Project Structure
## Project Structure

| Path | Purpose |
|------|---------|
| `src/data/imdb_loader.py` | Loads the IMDb dataset, creates smaller splits for experiments, and prepares text-only data for GPT training. |
| `src/training/train_bert.py` | Fine-tunes `distilbert-base-uncased` for sentiment classification on IMDb reviews. |
| `src/training/train_gpt.py` | Fine-tunes `distilgpt2` for causal language modeling on IMDb review text. |
| `src/demo.py` | Loads the saved models and runs inference demos for both sentiment classification and text generation. |

## Quick Start

**Missing files** (`bert_model.py`, `gpt_model.py`, `utils/`) are **intentionally simple wrappers** around pretrained models.

## 🚀 Quick Start

```bash
# Install dependencies
pip install transformers datasets evaluate accelerate torch

# Train BERT (sentiment classification)
python src/training/train_bert.py

# Train GPT (text generation)
python src/training/train_gpt.py

# Run demos
python src/demo.py
```

## Exact Setup

### BERT Side (Encoder)
Model: distilbert-base-uncased
Task: Sentiment classification
Dataset: IMDb (4k train, 1k test)
Output: {positive, negative} labels

### GPT Side (Decoder)
Model: distilgpt2
Task: Causal language modeling
Dataset: IMDb reviews (5k text samples)
Block size: 128 tokens
Prompt: "This movie was" -> generates continuation

## Model Saving
Models save to `outputs/` as complete Hugging Face bundles:
outputs/distilbert-imdb/ → model.safetensors + tokenizer + config
outputs/distilgpt2-imdb/ → model.safetensors + tokenizer + config

## 🔍 Why DistilBERT + DistilGPT2?
**Perfect for learning** - realistic pretrained behavior without waiting hours to train.

## 🎓 Key Insights From This Project

1. **Encoder vs Decoder Architecture**
BERT: Bidirectional context → "What's the sentiment of this review?"
GPT: Unidirectional (left-to-right) -> "What's the next word?"

2. **Same Data, Different Processing**
BERT keeps labels, uses full bidirectional context
GPT drops labels, trains on raw text sequences only

## 🔄 Reusable Template
This project is not limited to DistilBERT and DistilGPT2.  
The same structure can be reused to fine-tune and compare many other Hugging Face models because the code follows the standard `AutoTokenizer`, `AutoModelForSequenceClassification`, `AutoModelForCausalLM`, `Trainer`, and pipeline-based inference workflow
For example:
- `AutoModelForSequenceClassification` is used for encoder-style classification tasks.
- `AutoModelForCausalLM` is used for decoder-style next-token generation tasks.
