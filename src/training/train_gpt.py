"""
Training script for fine-tuning DistilGPT2 on IMDb for causal language modeling.

This script performs the full GPT-side training pipeline for the project:
1. Load IMDb dataset and prepare text-only training data.
2. Tokenize raw review text into token IDs.
3. Group tokens into fixed-length blocks for causal language modeling.
4. Load pretrained DistilGPT2 causal language model.
5. Fine-tune on IMDb movie reviews for next-token prediction.
6. Save the trained model and tokenizer to disk.

Unlike BERT classification, this trains GPT to predict the next word in movie
reviews, learning the style and patterns of IMDb text generation.
"""
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.imdb_loader import load_imdb_dataset, prepare_gpt_text_dataset

# Base pretrained model used for causal language modeling and text generation.
# DistilGPT2 is a smaller, faster version of GPT2 that's suitable for fine-tuning
# on consumer hardware like Colab GPUs.
MODEL_NAME = "distilgpt2"

# Directory where the final trained model, tokenizer, and checkpoints will be saved.
OUTPUT_DIR = "outputs/distilgpt2-imdb"

# Fixed sequence length for training blocks. All input sequences are chunked
# into 128-token blocks. This must be consistent across the entire pipeline.
BLOCK_SIZE = 128


def tokenize_function(examples, tokenizer):
    """
    Tokenize raw IMDb review text into token IDs.

    Args:
        examples: A batch from the dataset containing a "text" field.
        tokenizer: The Hugging Face tokenizer instance.

    Returns:
        A dictionary containing tokenized model inputs:
        - input_ids: sequence of token IDs

    What this function does:
    - Converts raw movie review text into numerical token IDs.

    """
    return tokenizer(examples["text"])

def group_texts(examples):
    """
    Concatenate texts and split into fixed-length chunks for causal LM training.
    Standard preprocessing pattern for GPT-style causal language modeling.

    Args:
        examples: Tokenized dataset batch containing input_ids.

    Returns:
        Dictionary with fixed-length token blocks ready for training:
        - input_ids: chunks of BLOCK_SIZE tokens
        - labels: copy of input_ids (self-supervised next-token prediction)

    What this function does:
    1. Concatenates all token sequences from the batch into one long sequence.
    2. Trims to exact multiple of BLOCK_SIZE.
    3. Splits into fixed 128-token chunks.
    4. Copies input_ids to labels for self-supervised training.

    Variables used:
    - concatenated: merged token sequences from the entire batch.
    - total_length: length of concatenated sequence (trimmed to BLOCK_SIZE multiple).
    - result: final chunked sequences with input_ids and labels.
    """
    # Concatenate all texts in the batch into single long sequences
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    # Trim to exact multiple of BLOCK_SIZE to avoid partial chunks
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    # Split into fixed-length chunks and create labels (self-supervised)
    result = {
        k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    """
    Run the full DistilGPT2 causal language modeling pipeline.

    Step-by-step flow:
    1. Create output directory.
    2. Load IMDb dataset and prepare text-only training data.
    3. Load DistilGPT2 tokenizer and set pad_token.
    4. Tokenize raw text into token IDs.
    5. Group tokens into fixed BLOCK_SIZE chunks.
    6. Load pretrained DistilGPT2 model.
    7. Configure data collator for LM.
    8. Define training arguments.
    9. Create Trainer and train.
    10. Save model and tokenizer.

    Variables used:
    - dataset: full IMDb DatasetDict.
    - train_text_ds: text-only training data from imdb_loader.
    - tokenizer: DistilGPT2 tokenizer with pad_token configured.
    - tokenized_dataset: tokenized but unchunked sequences.
    - lm_dataset: final chunked dataset ready for LM training.
    - model: pretrained DistilGPT2 language model.
    - data_collator: handles padding and label masking for LM.
    - training_args: GPT fine-tuning hyperparameters.
    - trainer: Hugging Face Trainer for the training loop.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading IMDb dataset...")
    dataset = load_imdb_dataset()
    # Prepare text-only dataset for causal language modeling (no labels needed).
    # GPT learns to predict next token in a self-supervised manner.
    train_text_ds = prepare_gpt_text_dataset(
        dataset,
        split="train",
        sample_size=5000,
        seed=42,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token by default, so align it to eos token.
    # This ensures proper padding during batching.
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing raw text...")
    # Tokenize raw text and remove the text column to save memory.
    tokenized_dataset = train_text_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    print("Grouping tokens into blocks...")
    # Concatenate and chunk into fixed BLOCK_SIZE sequences for causal LM.
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )

    print("Loading GPT model...")
    # Load pretrained causal language model for next-token prediction.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Sync model config with tokenizer pad token.
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data collator specifically for causal language modeling.
    # Handles dynamic padding and creates labels with -100 for non-prediction tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM (BERT-style)
    )

    # Training configuration optimized for GPT causal LM fine-tuning.
    # Key differences from BERT training:
    # - No evaluation (eval_strategy="no") since it's generative
    # - Higher learning rate (5e-5) for generative models
    # - Smaller batch size (8) due to sequence length and memory
    # - Single epoch sufficient for style adaptation
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        report_to="none",
    )

    # Trainer configured for causal language modeling.
    # data_collator handles the special LM padding/label logic.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting GPT training...")
    # Run the causal language modeling training loop.
    trainer.train()

    print("Saving model...")
    # Save trained model weights, config, and tokenizer for later inference.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
