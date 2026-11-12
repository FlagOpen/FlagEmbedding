from typing import List

import torch


def tensorize_batch(sequences: List[torch.Tensor], padding_value, align_right=False) -> torch.Tensor:
    if len(sequences[0].size()) == 1:
        max_len_1 = max([s.size(0) for s in sequences])
        out_dims = (len(sequences), max_len_1)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            if align_right:
                out_tensor[i, -length_1:] = tensor
            else:
                out_tensor[i, :length_1] = tensor
        return out_tensor
    elif len(sequences[0].size()) == 2:
        max_len_1 = max([s.size(0) for s in sequences])
        max_len_2 = max([s.size(1) for s in sequences])
        out_dims = (len(sequences), max_len_1, max_len_2)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            length_2 = tensor.size(1)
            if align_right:
                out_tensor[i, -length_1:, :length_2] = tensor
            else:
                out_tensor[i, :length_1, :length_2] = tensor
        return out_tensor
    else:
        raise
