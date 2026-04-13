import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data.imdb_loader import load_imdb_dataset, prepare_gpt_text_dataset


MODEL_NAME = "distilgpt2"   # lighter than gpt2 for Colab
OUTPUT_DIR = "outputs/distilgpt2-imdb"
BLOCK_SIZE = 128


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples):
    """
    Concatenate texts and split into fixed-length chunks.
    Standard CLM preprocessing pattern.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    result = {
        k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading IMDb dataset...")
    dataset = load_imdb_dataset()
    train_text_ds = prepare_gpt_text_dataset(
        dataset,
        split="train",
        sample_size=5000,
        seed=42,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token by default, so align it to eos token
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing raw text...")
    tokenized_dataset = train_text_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    print("Grouping tokens into blocks...")
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )

    print("Loading GPT model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting GPT training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
