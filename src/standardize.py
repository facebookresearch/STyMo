# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


class Standardize:
    def __init__(self):
        self.mean = None
        # self.std = None

    def fit(self, x):
        self.mean = np.mean(x.reshape(-1, x.shape[-1]), axis=0)
        self.std = np.std(x.reshape(-1, x.shape[-1]))
        self.std = np.where(self.std < 1e-9, 1.0, self.std)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_torch = torch.from_numpy(self.mean).float().to(device)
        self.std_torch = torch.from_numpy(self.std).float().to(device)
        return self

    def standardize(self, x):
        return (x - self.mean) / self.std
        # return x - self.mean

    def standardize_torch(self, x):
        return (x - self.mean_torch) / self.std_torch
        # return x - self.mean_torch

    def unstandardize(self, x):
        return x * self.std + self.mean
        # return x + self.mean

    def unstandardize_torch(self, x):
        return x * self.std_torch + self.mean_torch
        # return x + self.mean_torch
