# src/data/imdb_loader.py

from datasets import load_dataset


def load_imdb_dataset():
    """
    Load the IMDb dataset from Hugging Face.

    Returns:
        DatasetDict with train and test splits.
    """
    return load_dataset("imdb")

def get_small_splits(dataset, train_size=2000, test_size=1000, seed=42):
    """
    
    Args:
        dataset: Hugging Face DatasetDict
        train_size: number of training examples
        test_size: number of test examples
        seed: random seed

    Returns:
        train_dataset, test_dataset
    """
    train_dataset = dataset["train"].shuffle(seed=seed).select(range(train_size))
    test_dataset = dataset["test"].shuffle(seed=seed).select(range(test_size))
    return train_dataset, test_dataset


def prepare_gpt_text_dataset(dataset, split="train", sample_size=3000, seed=42):
    """
    Prepare raw text-only dataset for GPT training.

    Args:
        dataset: Hugging Face DatasetDict
        split: "train" or "test"
        sample_size: number of examples to keep
        seed: random seed

    Returns:
        Hugging Face Dataset with only 'text'
    """
    ds = dataset[split].shuffle(seed=seed).select(range(sample_size))
    return ds.remove_columns([col for col in ds.column_names if col != "text"])
