
import torch
import torch.nn as nn


class MiniBERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        num_classes=2,
        max_len=512
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.encoder(x)

        cls_representation = x[:, 0, :]
        logits = self.classifier(cls_representation)

        return logits
