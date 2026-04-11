import os
import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.data.imdb_loader import load_imdb_dataset, get_small_splits


MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "outputs/distilbert-imdb"


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading IMDb dataset...")
    dataset = load_imdb_dataset()

    # smaller subset first for fast testing
    train_dataset, test_dataset = get_small_splits(
        dataset,
        train_size=4000,
        test_size=1000,
        seed=42,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    tokenized_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    tokenized_test.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    print("Loading DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT training...")
    trainer.train()

    print("Evaluating DistilBERT...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
