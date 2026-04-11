import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"


def basic_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"<br\\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\\s']", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text.split()


class Vocab:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.itos = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]
        self.stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2}

    def build(self, texts: List[str]) -> None:
        counter = Counter()

        for text in texts:
            tokens = basic_tokenize(text)
            counter.update(tokens)

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, tokens: List[str]) -> List[int]:
        unk_idx = self.stoi[UNK_TOKEN]
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[idx] for idx in ids]

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def cls_idx(self) -> int:
        return self.stoi[CLS_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)


class IMDBBertDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocab,
        max_len: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = basic_tokenize(text)

        # reserve 1 spot for <CLS>
        tokens = tokens[: self.max_len - 1]

        token_ids = [self.vocab.cls_idx] + self.vocab.encode(tokens)
        attention_mask = [1] * len(token_ids)

        pad_length = self.max_len - len(token_ids)

        if pad_length > 0:
            token_ids += [self.vocab.pad_idx] * pad_length
            attention_mask += [0] * pad_length

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_imdb_data(train_size: int = 5000, test_size: int = 1000) -> Tuple[List[str], List[int], List[str], List[int]]:
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"][:train_size]
    train_labels = dataset["train"]["label"][:train_size]

    test_texts = dataset["test"]["text"][:test_size]
    test_labels = dataset["test"]["label"][:test_size]

    return train_texts, train_labels, test_texts, test_labels


def build_vocab_from_train(train_texts: List[str], min_freq: int = 2) -> Vocab:
    vocab = Vocab(min_freq=min_freq)
    vocab.build(train_texts)
    return vocab


def create_bert_dataloaders(
    train_size: int = 5000,
    test_size: int = 1000,
    max_len: int = 128,
    batch_size: int = 32,
    min_freq: int = 2,
):
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(
        train_size=train_size,
        test_size=test_size
    )

    vocab = build_vocab_from_train(train_texts, min_freq=min_freq)

    train_dataset = IMDBBertDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        max_len=max_len,
    )

    test_dataset = IMDBBertDataset(
        texts=test_texts,
        labels=test_labels,
        vocab=vocab,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, vocab
