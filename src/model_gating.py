# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class GatingModel(nn.Module):
    """
    Stylizability Gate model - predicts whether a pose should be stylized.

    This network learns to output a value in [0, 1] that indicates:
    - 1.0: The pose is within the training distribution and should be fully stylized
    - 0.0: The pose is out-of-distribution (rare/unusual) and should NOT be stylized

    The output is used to interpolate between identity (no stylization) and
    the predicted stylization from the static/dynamic models:
        P_style_pred = P_neutral + s * Delta_R_pred

    where s is the gating score from this model.

    Input: Same as AveragePoseModel - flattened 6D rotations (root excluded)
    Output: Single sigmoid value [0, 1]
    
    Args:
        input_dim: Input dimension (flattened 6D rotations)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout rate
        always_one: If True, forward() always returns 1.0 (full stylization).
                    Use this when no gating training data is available.
    """

    def __init__(
        self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, always_one=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.always_one = always_one

        # Only create the MLP if we're not in always_one mode
        if not always_one:
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
                f"linear{self.num_layers + 1}", nn.Linear(self.hidden_dim, 1)
            )
            self.mlp.add_module("sigmoid", nn.Sigmoid())
        else:
            self.mlp = None

    def forward(self, x):
        """
        x: [batch, num_joints-1, 3, 2] - rotations in 6D representation (root excluded)
           or [batch, (num_joints-1) * 6] - already flattened
        returns: [batch, 1] - stylizability score in [0, 1]
        """
        batch_size = x.shape[0]
        
        # If always_one mode, return ones without any computation
        if self.always_one:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        
        x_flat = x.reshape(batch_size, -1)
        return self.mlp(x_flat)
