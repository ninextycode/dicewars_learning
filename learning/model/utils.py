import torch
from torch import nn


class DWNorm(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, x):
        # downscale dice on a node
        return x / 8
    

class DWDeNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # downscale dice on a node
        return x * 8
        

def indexed_softmax(values: torch.Tensor, indices: torch.Tensor, indexed_dim=-1):
    # values - [..., n_items, ...]
    # indices - [n_items]
    n_indices = indices.max() + 1  

    max_per_index_shape = list(values.shape)
    max_per_index_shape[indexed_dim] = n_indices

    # max_per_index - [..., n_indices, ...]
    max_per_index = torch.full(
        max_per_index_shape,
        -float('inf'),
        device=values.device,
        dtype=values.dtype
    )

    max_per_index.index_reduce_(
        dim=indexed_dim,
        index=indices,
        source=values,
        reduce="amax"
    )
    
    index_expand_idx = [slice(None)] * len(max_per_index_shape)
    index_expand_idx[indexed_dim] = indices
    index_expand_idx = tuple(index_expand_idx)
    # values_exp / max_per_index[index_expand_idx] - [..., n_items, ...]
    values_exp = torch.exp(values - max_per_index[index_expand_idx])

    # sum_per_index - [..., n_indices, ...]
    sum_per_index = torch.zeros(
        size=max_per_index.shape,
        dtype=values_exp.dtype,
        device=values_exp.device
    )
    sum_per_index.index_add_(
        dim=indexed_dim,
        index=indices,
        source=values_exp
    )

    # values_softmax / sum_per_index[index_expand_idx] - [..., n_items, ...]
    # groups of items corresponding to indices are transformed with softmax for each group
    values_softmax = values_exp / sum_per_index[index_expand_idx]
    return values_softmax