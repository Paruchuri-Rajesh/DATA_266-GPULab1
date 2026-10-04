"""
Sentiment classifiers. All embeddings are nn.Embedding layers initialised
randomly and learned from scratch.

  bag     — mean-pooled word embeddings + MLP (fastText-style baseline)
  cnn     — multi-width 1-D convolutions + max-over-time pooling (Kim 2014)
  bilstm  — packed BiLSTM with additive attention pooling
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask(x):
    return (x != 0).float()


class BagOfEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        m = _mask(x).unsqueeze(-1)
        pooled = (self.embedding(x) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(pooled)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, dropout, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths=None):
        m = _mask(x).unsqueeze(1)  # (B,1,T)
        emb = self.embedding(x).transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(emb))[:, :, : x.size(1)]
            h = h.masked_fill(m == 0, float("-inf")).max(dim=2).values
            feats.append(torch.nan_to_num(h, neginf=0.0))
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.emb_drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        scores = self.attn(out).squeeze(-1).masked_fill(x == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (out * weights).sum(1)
        return self.fc(self.dropout(pooled))


def build_model(model_cfg: dict, vocab_size: int) -> nn.Module:
    t = model_cfg["type"]
    if t == "bag":
        return BagOfEmbeddings(vocab_size, model_cfg["embed_dim"], model_cfg["hidden_dim"], model_cfg["dropout"])
    if t == "cnn":
        return CNNClassifier(vocab_size, model_cfg["embed_dim"], model_cfg["num_filters"], model_cfg["kernel_sizes"], model_cfg["dropout"])
    if t == "bilstm":
        return BiLSTMAttention(vocab_size, model_cfg["embed_dim"], model_cfg["hidden_dim"], model_cfg["num_layers"], model_cfg["dropout"])
    raise ValueError(f"unknown model type {t}")
