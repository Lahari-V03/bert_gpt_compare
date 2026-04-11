import torch
import torch.nn as nn


class MiniBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_classes: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = x.new_zeros((batch_size, seq_len), dtype=torch.bool)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls_representation = x[:, 0, :]
        logits = self.classifier(self.dropout(cls_representation))

        return logits
