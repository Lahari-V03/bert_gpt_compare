"""
Training script for fine-tuning DistilBERT on IMDb sentiment classification.

This script performs the full BERT training pipeline for the project:
1. Load the IMDb dataset.
2. Create smaller train/test splits for faster experimentation.
3. Tokenize the text reviews.
4. Load a pretrained DistilBERT model for sequence classification.
5. Fine tune the model on the IMDb labels.
6. Evaluate the model.
7. Save the trained model and tokenizer to disk.

The goal of this file is to provide a reproducible training pipeline that can
be run from start to finish without modifying the core logic.
"""

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


# Base pretrained model used for sentiment classification.
# DistilBERT is a smaller, faster version of BERT that works well for
# text classification tasks such as sentiment analysis.
MODEL_NAME = "distilbert-base-uncased"

# Directory where the final trained model, tokenizer, and checkpoints will be saved.
OUTPUT_DIR = "outputs/distilbert-imdb"


def tokenize_function(examples, tokenizer):
    """
    Tokenize a batch of IMDb review examples.

    Args:
        examples: A batch from the dataset containing a "text" field.
        tokenizer: The Hugging Face tokenizer instance.

    Returns:
        A dictionary containing tokenized model inputs such as:
        - input_ids
        - attention_mask

    What this function does:
    - Takes raw review text from the dataset.
    - Converts it into token IDs compatible with DistilBERT.
    - Applies truncation so long reviews do not exceed the maximum sequence length.
    - Pads sequences to a fixed length so batches are uniform.

    Variables used:
    - examples: the batched dataset examples passed into the function.
    - tokenizer: the tokenizer used to convert raw text into tokens.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for classification predictions.

    Args:
        eval_pred: A tuple returned by the Trainer during evaluation.
            - predictions: raw model logits
            - labels: ground-truth sentiment labels

    Returns:
        A dictionary with the computed accuracy score.

    Variables used:
    - accuracy: metric loader for accuracy computation.
    - predictions: raw model outputs before argmax.
    - labels: true labels from the dataset.
    - preds: predicted class indices after applying argmax.
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)


def main():
    """
    Run the full DistilBERT fine-tuning pipeline.

    Step-by-step flow:
    1. Create the output directory.
    2. Load the IMDb dataset.
    3. Select a smaller subset for faster training and testing.
    4. Load the tokenizer.
    5. Tokenize the train and test splits.
    6. Rename the label column to labels for PyTorch/Trainer compatibility.
    7. Convert datasets to torch tensor format.
    8. Load pretrained DistilBERT for sequence classification.
    9. Define training arguments.
    10. Create a Trainer instance.
    11. Train the model.
    12. Evaluate the model.
    13. Save the model and tokenizer.

    Variables used:
    - dataset: full IMDb DatasetDict.
    - train_dataset: smaller training subset.
    - test_dataset: smaller evaluation subset.
    - tokenizer: DistilBERT tokenizer.
    - tokenized_train: tokenized training dataset.
    - tokenized_test: tokenized evaluation dataset.
    - model: pretrained DistilBERT classification model.
    - training_args: configuration for training and evaluation.
    - trainer: Hugging Face Trainer object.
    - metrics: evaluation results after training.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading IMDb dataset...")
    dataset = load_imdb_dataset()

    # Smaller subset first for fast testing and shorter training runs.
    # This helps during development before scaling up.
    train_dataset, test_dataset = get_small_splits(
        dataset,
        train_size=4000,
        test_size=1000,
        seed=42,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    # Map the tokenizer across the dataset in batches for efficiency.
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    # Trainer expects the label column to be named "labels".
    # The IMDb dataset uses "label", so we rename it here.
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Convert the dataset into PyTorch tensor format.
    # Only the model input fields and labels are kept in the memory.
     tokenized_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    tokenized_test.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    print("Loading DistilBERT model...")
    # Load a pretrained DistilBERT model with a classification head.
    # num_labels=2 because IMDb sentiment is a binary classification task
    # (positive=1, negative=0).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    
    # Training configuration for the Hugging Face Trainer.
    # Key settings explained:
    # - eval_strategy="epoch": evaluate after every epoch
    # - save_strategy="epoch": save checkpoint after every epoch
    # - logging_steps=50: log training progress every 50 steps
    # - learning_rate=2e-5: standard fine-tuning learning rate
    # - batch_size=16: reasonable size for most GPUs/CPUs
    # - num_train_epochs=2: enough for convergence on this task
    # - load_best_model_at_end=True: keep the best performing model
    # - metric_for_best_model="accuracy": optimize for accuracy
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

    # Trainer handles the full training loop, evaluation, logging, and saving.
    # processing_class=tokenizer ensures inputs are tokenized during training.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT training...")
     # This runs the full training loop according to training_args.
    trainer.train()

    print("Evaluating DistilBERT...")
    # Run final evaluation on the test set and print results.
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    print("Saving model...")
    # Save the trained model weights, config, and tokenizer.
    # trainer.save_model() saves everything needed for inference.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
    
