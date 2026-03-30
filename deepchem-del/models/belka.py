import math
from typing import Optional

import torch
import torch.nn as nn
from deepchem.models import TorchModel
import torch.nn.functional as F


class Encodings(nn.Module):
    def __init__(self, depth: int, max_length: int):
        super().__init__()
        self.depth = depth
        self.max_length = max_length

        enc = self._pos_encodings(depth, max_length)  # [max_length, depth]
        self.register_buffer("encodings", enc, persistent=False)

    @staticmethod
    def _pos_encodings(depth: int, max_length: int) -> torch.Tensor:
        """
        """
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        idx = torch.arange(depth, dtype=torch.float32).unsqueeze(0)            # [1, D]
        power = 2 * torch.div(idx, 2, rounding_mode='floor')   # [1, D]
        power = power / depth
        angles = 1.0 / (10000.0 ** power)                     # [1, D]
        radians = positions * angles                   # [L, D]

        sin = torch.sin(radians[:, 0::2])
        cos = torch.cos(radians[:, 1::2])
        encodings = torch.cat([sin, cos], dim=-1)
        return encodings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        scale = math.sqrt(self.depth)
        x = x * scale
        seq_len = x.size(1)
        x = x + self.encodings[:seq_len, :].unsqueeze(0)  # [1, L, D]
        return x


class Embeddings(nn.Module):
    def __init__(self, max_length: int, depth: int, input_dim: int):
        super().__init__()
        self.depth = depth
        self.max_length = max_length
        self.input_dim = input_dim

        # Keras: Embedding(mask_zero=True) -> padding_idx=0
        self.embeddings = nn.Embedding(num_embeddings=input_dim,
                                       embedding_dim=depth,
                                       )
        self.encodings = Encodings(depth=depth, max_length=max_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        """
        print("embed inputs", inputs)
        x = self.embeddings(inputs)       # [B, L, D]
        print("embed output", x)
        x = self.encodings(x)             # add positional encodings + scaling
        return x

    def compute_padding_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        """
        return inputs.eq(0)


class FeedForward(nn.Module):
    def __init__(self, activation: str, depth: int, dropout_rate: float, epsilon: float):
        super().__init__()
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

        self.norm = nn.LayerNorm(depth, eps=epsilon)
        self.dense1 = nn.Linear(depth, depth * 2)
        self.dense2 = nn.Linear(depth * 2, depth)
        self.dropout = nn.Dropout(dropout_rate)

        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        """
        x = self.norm(inputs)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = x + inputs
        return x


class SelfAttention(nn.Module):
    """
    """
    def __init__(self, causal: bool, depth: int, dropout_rate: float,
                 epsilon: float, max_length: int, num_heads: int):
        super().__init__()
        self.causal = causal
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.max_length = max_length
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(depth, eps=epsilon)
        # PyTorch MHA expects (L, N, E) by default
        self.mha = nn.MultiheadAttention(embed_dim=depth,
                                         num_heads=num_heads,
                                         dropout=dropout_rate,
                                         batch_first=True)  # so use [B, L, D]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        """

        attn_mask = None
        if self.causal:
            L = inputs.size(1)
            attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=inputs.device), diagonal=1)

        x = self.norm(inputs)
  
        x_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,  # [B, L]
            need_weights=False,
        )
        x_out = self.dropout(x_out)
        x_out = x_out + inputs
        return x_out


# ========= Encoder Layer =========

class EncoderLayer(nn.Module):
    """

    """
    def __init__(self, activation: str, depth: int, dropout_rate: float,
                 epsilon: float, max_length: int, num_heads: int):
        super().__init__()
        self.self_attention = SelfAttention(
            causal=False,
            depth=depth,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            max_length=max_length,
            num_heads=num_heads,
        )
        self.ffn = FeedForward(
            activation=activation,
            depth=depth,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
        )

    def forward(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attention(inputs, padding_mask=padding_mask)
        x = self.ffn(x)
        return x


# ========= Belka Model =========

class Belka(nn.Module):
    """
    Modes:
        - "mlm_pretraining": Masked Language Modeling pretraining. The model learns to predict masked tokens in the input SMILES.
        - "ecfp_pretraining": Pretraining mode to predict ECFP (Extended Connectivity Fingerprint) molecular representations from input SMILES.
        - "classification": Classification mode.
    """
    def __init__(self, dropout_rate: float, mode: str,
                 num_layers: int, vocab_size: int,
                 max_length: int, depth: int, num_heads: int,
                 activation: str = "gelu", epsilon: float = 1e-6):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mode = mode

        self.embeddings = Embeddings(
            input_dim=vocab_size,
            max_length=max_length,
            depth=depth,
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                activation=activation,
                depth=depth,
                dropout_rate=dropout_rate,
                epsilon=epsilon,
                max_length=max_length,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        if mode == "mlm_pretraining":
            self.head = nn.Linear(depth, vocab_size)
        elif mode == "ecfp_pretraining":
            self.dropout = nn.Dropout(dropout_rate)
            self.head = nn.Linear(depth, 2048)
        elif mode == "classification":
            self.dropout = nn.Dropout(dropout_rate)
            self.head = nn.Linear(depth, 3)
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        """
        padding_mask = self.embeddings.compute_padding_mask(inputs)  # [B, L] bool
        x = self.embeddings(inputs)  # [B, L, D]

        for layer in self.encoder_layers:
            x = layer(x, padding_mask=padding_mask)

        if self.mode == "mlm_pretraining":
            # token-level predictions [B, L, vocab]
            logits = self.head(x)
            return F.softmax(logits, dim=-1)
        else:
            # GlobalAvgPool1D
            mask = (~padding_mask).unsqueeze(-1).type_as(x)  # [B, L, 1]
            x_masked = x * mask
            denom = mask.sum(dim=1).clamp(min=1.0)           # [B, 1]
            pooled = x_masked.sum(dim=1) / denom             # [B, D]

            pooled = self.dropout(pooled)
            out = self.head(pooled)  # [B, out_dim]
            out = torch.sigmoid(out)
            return out


class BelkaModel(TorchModel):
    def __init__(self, mode, *args, **kwargs):
        model = Belka(mode, *args, **kwargs)
        if mode == "mlm_pretraining":
            # need to add Category Loss
            pass
        elif mode == "ecfp_pretraining":
            # need to add Binary Loss
            pass
        elif mode == "classification":
            # need to add MultiLabel Loss
            pass
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        super().__init__(model, mode, *args, **kwargs)

    def _prepare_batch(self, batch):
        # handle prepare batch for each mode
        pass
