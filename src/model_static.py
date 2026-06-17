# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class AveragePoseModel(nn.Module):
    """
    Chunk classifier model - predicts which chunk an average rotation pattern belongs to.
    """

    def __init__(
        self, input_dim, output_dim, hidden_dim=256, num_layers=2, dropout=0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.mlp = nn.Sequential()
        self.mlp.add_module("linear1", nn.Linear(self.input_dim, self.hidden_dim))
        self.mlp.add_module("relu1", nn.ReLU())
        self.mlp.add_module("dropout1", nn.Dropout(self.dropout))
        for i in range(self.num_layers - 1):
            self.mlp.add_module(
                f"linear{i + 2}", nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.mlp.add_module(f"relu{i + 2}", nn.ReLU())
            self.mlp.add_module(f"dropout{i + 2}", nn.Dropout(self.dropout))
        self.mlp.add_module(
            f"linear{self.num_layers + 1}", nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        """
        x: [batch, num_joints-1, 3, 2] - average rotations in 6D representation (root excluded)
        returns: [batch, num_chunks] - classification logits
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # Flatten to [batch, (num_joints-1) * 6]
        return self.mlp(x_flat)  # [batch, num_chunks]
