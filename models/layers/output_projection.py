import torch
import os
import json
import torch.nn as nn


class FlattenHead(nn.Module):
    def __init__(self, configs):
        super(FlattenHead, self).__init__()
        self.patch_nums = int((configs["seq_len"] - configs["stride"]) / configs["stride"] + 2)
        self.head_nf = configs["d_ff"] * self.patch_nums

        self.n_vars = configs["enc_in"]
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.head_nf, configs["pred_len"])
        self.dropout = nn.Dropout(configs["dropout"])

    def forward(self, x):
        x = x[:, :, :, -self.patch_nums:]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()
        return x