"""
This file is used to verify the final behavior of the BERT and GPT fine tuned models:
1. A BERT sentiment classifier.
2. A GPT text generator.

The purpose of this script is not training. Instead, it loads the saved models and runs sample inference so you can confirm that everything works
end-to-end after training and saving.
"""

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Path to the fine-tuned BERT/DistilBERT sentiment classification model.
# This directory contains the saved model weights, tokenizer files,
# and configuration files needed for inference.
BERT_MODEL_PATH = "outputs/distilbert-imdb"

# Path to the fine-tuned GPT/DistilGPT2 causal language model.
# This directory contains the saved model weights, tokenizer files,
# and configuration files needed for generation.
GPT_MODEL_PATH = "outputs/distilgpt2-imdb"


def bert_demo():
    """
    To run a small sentiment classification demo using the saved BERT-side model.

    This function:
    - Loads a Hugging Face text classification pipeline.
    - Sends a few example IMDb style reviews to the model.
    - Prints the predicted sentiment label and confidence score.

    Variables used:
    - sentiment_pipe: the inference pipeline for text classification.
    - samples: list of example sentences used for testing.
    - text: the current review being passed through the model.
    - pred: the model's prediction output for one review.
    """
    print("\nBERT SENTIMENT DEMO")

    # Create a Hugging Face pipeline for text classification.
    # The model and tokenizer are loaded from the same saved directory.
    sentiment_pipe = pipeline(
        "text-classification",
        model=BERT_MODEL_PATH,
        tokenizer=BERT_MODEL_PATH,
    )

    # Example inputs used to test the classifier.
    # These are chosen to represent clearly positive, clearly negative,
    # and mixed sentiment so you can observe model behavior.
    samples = [
        "This movie was very wonderful, emotional, and well acted.",
        "This film was boring and slow",
        "The cinematography was nice but the story felt weak.",
    ]

    # Loop through each sample review and print the prediction.
    for text in samples:
        pred = sentiment_pipe(text)[0]
        print(f"\nReview: {text}")
        print(f"Prediction: {pred}")


def gpt_demo():
    """
    To run a text generation demo using the saved GPT-side model.

    This function:
    - Loads the tokenizer and language model.
    - Tokenizes a short prompt.
    - Generates two possible continuations from the prompt.
    - Decodes and prints the generated text.

    Variables used:
    - tokenizer: converts text to token IDs and token IDs back to text.
    - model: the fine-tuned causal language model.
    - prompt: the starting text used to guide generation.
    - inputs: tokenized version of the prompt returned as PyTorch tensors.
    - outputs: generated token sequences returned by the model.
    - i: sequence counter for printed generations.
    - output: one generated token sequence from the model.
    - generated_text: decoded text string for display.
    """
    print("\nGPT TEXT GENERATION DEMO")

    # Load the tokenizer associated with the saved GPT model.
    # This is required to convert raw text into token IDs.
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_PATH)

    # Load the fine-tuned GPT-style causal language model.
    # This model predicts the next token in a sequence.
    model = AutoModelForCausalLM.from_pretrained(GPT_MODEL_PATH)

    # Short seed text that the model will continue from.
    prompt = "This movie was"

    # Convert the prompt into PyTorch tensors so it can be passed to the model.
    inputs = tokenizer(prompt, return_tensors="pt")

    # Disable gradient tracking because this is inference, not training.
    # This reduces memory use and makes generation more efficient.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each generated token sequence back into readable text.
    for i, output in enumerate(outputs, start=1):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\nGeneration {i}: {generated_text}")


if __name__ == "__main__":
    # Run the sentiment classification demo
    bert_demo()

    # Run the GPT generation demo
    gpt_demo()
