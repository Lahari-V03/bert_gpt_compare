
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)

BERT_MODEL_PATH = "outputs/distilbert-imdb"
GPT_MODEL_PATH = "outputs/distilgpt2-imdb"


def bert_demo():
    print("\n===== BERT SENTIMENT DEMO =====")
    sentiment_pipe = pipeline(
        "text-classification",
        model=BERT_MODEL_PATH,
        tokenizer=BERT_MODEL_PATH,
    )

    samples = [
        "This movie was absolutely wonderful, emotional, and beautifully acted.",
        "This film was boring, slow, and a complete waste of time.",
        "The cinematography was nice but the story felt weak.",
    ]

    for text in samples:
        pred = sentiment_pipe(text)[0]
        print(f"\nReview: {text}")
        print(f"Prediction: {pred}")


def gpt_demo():
    print("\n===== GPT TEXT GENERATION DEMO =====")
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(GPT_MODEL_PATH)

    prompt = "This movie was"
    inputs = tokenizer(prompt, return_tensors="pt")

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

    for i, output in enumerate(outputs, start=1):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\nGeneration {i}: {generated_text}")


if __name__ == "__main__":
    bert_demo()
    gpt_demo()
