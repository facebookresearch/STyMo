# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model for motion style transfer.
"""

import torch.nn as nn
from utils import SinusoidalPositionalEncoding
from transformer import Transformer


class ModelTransformer(nn.Module):
    """
    Transformer-based model for motion style transfer.

    Uses encoder-decoder architecture for learning style patterns.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        window_size,
        emb_dim=64,
        dropout_p=0.2,
        n_layers=2,
        n_heads=4,
        dim_feedforward=512,
    ):
        super(ModelTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Input Features - Kinematics
        self.linear_emb_k = nn.Linear(input_dim, self.emb_dim)

        # Transformer
        self.transformer = Transformer(
            d_model=self.emb_dim,
            nhead=n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout_p,
            activation="relu",
            batch_first=True,
        )

        # Decoder
        self.out = nn.Linear(self.emb_dim, output_dim)
        self.linear_emb_p = nn.Linear(output_dim, self.emb_dim)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            window_size, d_model=self.emb_dim
        )

    def forward(
        self,
        x_k,
        x_p,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        encoder_output_scale=1.0,
    ):
        """
        Forward pass through the transformer model.

        Args:
            x_k: Source kinematics [batch, window_frames, features_in]
            x_p: Previous target kinematics [batch, window_frames, features_out]
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            encoder_output_scale: Scale factor for encoder output

        Returns:
            Predicted target kinematics [batch, window_frames, features_out]
        """
        # Input Features - Kinematics
        x_k = self.linear_emb_k(x_k)
        x_p = self.linear_emb_p(x_p)

        # Transformer
        x_k = x_k + self.pos_encoder.forward_temporality(x_k.shape[1])
        x_p = x_p + self.pos_encoder.forward_temporality(x_p.shape[1])
        y_hat = self.transformer(
            src=x_p,
            tgt=x_k,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            encoder_output_scale=encoder_output_scale,
        )

        # Decoder
        y_hat_res = self.out(y_hat)

        return y_hat_res
