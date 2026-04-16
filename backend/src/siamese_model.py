"""
Siamese Neural Network Models

Implements Siamese architectures with LSTM and GRU encoders for
paraphrase detection and semantic textual similarity. These serve
as baseline models using static (GloVe) embeddings, contrasting
with contextualised transformer-based representations.

Supports both:
  - Classification (paraphrase detection): outputs binary logits
  - Regression (STS): outputs a single similarity score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    """
    Self-attention pooling layer.

    Learns to weight each encoder timestep by importance, producing
    a weighted-average sentence representation instead of relying
    solely on the final hidden state.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, mask=None):
        """
        Args:
            encoder_outputs: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            context: (batch, hidden_dim) — attention-weighted representation
        """
        scores = self.attn(encoder_outputs).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class SiameseLSTM(nn.Module):
    """
    Siamese network with LSTM encoders.

    Each sentence is independently encoded by a shared LSTM. The resulting
    sentence embeddings are compared using cosine similarity and Manhattan
    distance, which are concatenated and passed through a fully connected
    classifier/regressor.
    """

    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=1,
                 dropout=0.3, task="classification"):
        """
        Args:
            embedding_matrix: numpy array (vocab_size, embed_dim) with pre-trained vectors
            hidden_dim: LSTM hidden state dimensionality
            num_layers: number of stacked LSTM layers
            dropout: dropout rate for regularisation
            task: 'classification' (binary) or 'regression' (similarity score)
        """
        super(SiameseLSTM, self).__init__()

        self.task = task
        vocab_size, embed_dim = embedding_matrix.shape

        # Embedding layer (frozen — static embeddings)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            requires_grad=False
        )

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention pooling over all encoder timesteps
        self.attention = Attention(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)

        # FC layers: input = cosine_sim (1) + manhattan_dist (1) + |h1 - h2| (hidden*2) + h1*h2 (hidden*2)
        fc_input_dim = 2 + hidden_dim * 4
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_input_dim),
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
        )

        if task == "classification":
            self.output_layer = nn.Linear(64, 2)  # binary logits
        else:
            self.output_layer = nn.Linear(64, 1)  # regression score

    def encode(self, x):
        """Encode a sentence through embedding + LSTM + attention pooling."""
        # Create mask: 1 for real tokens, 0 for padding (padding_idx=0)
        mask = (x != 0).float()  # (batch, seq_len)

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch, seq_len, hidden_dim*2)

        # Use attention pooling over all timesteps
        attended = self.attention(output, mask)  # (batch, hidden_dim*2)
        attended = self.dropout(attended)
        return attended

    def forward(self, s1_input_ids, s2_input_ids, labels=None):
        """
        Forward pass for the Siamese LSTM.

        Args:
            s1_input_ids: (batch, seq_len) word indices for sentence 1
            s2_input_ids: (batch, seq_len) word indices for sentence 2
            labels: optional labels for loss computation

        Returns:
            dict with 'logits' (or 'scores' for regression) and optionally 'loss'
        """
        h1 = self.encode(s1_input_ids)  # (batch, hidden_dim*2)
        h2 = self.encode(s2_input_ids)  # (batch, hidden_dim*2)

        # Distance features
        cosine_sim = F.cosine_similarity(h1, h2, dim=1).unsqueeze(1)  # (batch, 1)
        manhattan_dist = torch.sum(torch.abs(h1 - h2), dim=1).unsqueeze(1)  # (batch, 1)
        abs_diff = torch.abs(h1 - h2)  # (batch, hidden_dim*2)
        element_product = h1 * h2  # (batch, hidden_dim*2)

        combined = torch.cat([cosine_sim, manhattan_dist, abs_diff, element_product], dim=1)
        features = self.fc(combined)
        output = self.output_layer(features)

        result = {}
        if self.task == "classification":
            result["logits"] = output
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                result["loss"] = loss_fn(output, labels)
        else:
            scores = torch.sigmoid(output.squeeze(1))
            result["scores"] = scores
            if labels is not None:
                loss_fn = nn.SmoothL1Loss()
                result["loss"] = loss_fn(scores, labels.float())

        return result


class SiameseGRU(nn.Module):
    """
    Siamese network with GRU encoders.

    Identical architecture to SiameseLSTM but uses GRU cells instead,
    which have fewer parameters (no separate cell state) and can train
    faster on smaller datasets.
    """

    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=1,
                 dropout=0.3, task="classification"):
        """
        Args:
            embedding_matrix: numpy array (vocab_size, embed_dim) with pre-trained vectors
            hidden_dim: GRU hidden state dimensionality
            num_layers: number of stacked GRU layers
            dropout: dropout rate for regularisation
            task: 'classification' (binary) or 'regression' (similarity score)
        """
        super(SiameseGRU, self).__init__()

        self.task = task
        vocab_size, embed_dim = embedding_matrix.shape

        # Embedding layer (frozen — static embeddings)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            requires_grad=False
        )

        # Shared GRU encoder
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention pooling over all encoder timesteps
        self.attention = Attention(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)

        # FC layers: input = cosine_sim (1) + manhattan_dist (1) + |h1 - h2| (hidden*2) + h1*h2 (hidden*2)
        fc_input_dim = 2 + hidden_dim * 4
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_input_dim),
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
        )

        if task == "classification":
            self.output_layer = nn.Linear(64, 2)  # binary logits
        else:
            self.output_layer = nn.Linear(64, 1)  # regression score

    def encode(self, x):
        """Encode a sentence through embedding + GRU + attention pooling."""
        # Create mask: 1 for real tokens, 0 for padding (padding_idx=0)
        mask = (x != 0).float()  # (batch, seq_len)

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        output, hidden = self.gru(embedded)
        # output: (batch, seq_len, hidden_dim*2)

        # Use attention pooling over all timesteps
        attended = self.attention(output, mask)  # (batch, hidden_dim*2)
        attended = self.dropout(attended)
        return attended

    def forward(self, s1_input_ids, s2_input_ids, labels=None):
        """
        Forward pass for the Siamese GRU.

        Args:
            s1_input_ids: (batch, seq_len) word indices for sentence 1
            s2_input_ids: (batch, seq_len) word indices for sentence 2
            labels: optional labels for loss computation

        Returns:
            dict with 'logits' (or 'scores' for regression) and optionally 'loss'
        """
        h1 = self.encode(s1_input_ids)  # (batch, hidden_dim*2)
        h2 = self.encode(s2_input_ids)  # (batch, hidden_dim*2)

        # Distance features
        cosine_sim = F.cosine_similarity(h1, h2, dim=1).unsqueeze(1)  # (batch, 1)
        manhattan_dist = torch.sum(torch.abs(h1 - h2), dim=1).unsqueeze(1)  # (batch, 1)
        abs_diff = torch.abs(h1 - h2)  # (batch, hidden_dim*2)
        element_product = h1 * h2  # (batch, hidden_dim*2)

        combined = torch.cat([cosine_sim, manhattan_dist, abs_diff, element_product], dim=1)
        features = self.fc(combined)
        output = self.output_layer(features)

        result = {}
        if self.task == "classification":
            result["logits"] = output
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                result["loss"] = loss_fn(output, labels)
        else:
            scores = torch.sigmoid(output.squeeze(1))
            result["scores"] = scores
            if labels is not None:
                loss_fn = nn.SmoothL1Loss()
                result["loss"] = loss_fn(scores, labels.float())

        return result
