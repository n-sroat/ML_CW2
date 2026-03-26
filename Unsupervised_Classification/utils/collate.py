"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
import collections.abc

string_classes = (str, bytes)

def collate_custom(batch):
    if len(batch) == 0:
        return batch  # handle empty batch

    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(elem, np.ndarray):
        return torch.from_numpy(np.stack(batch, 0))
    elif isinstance(elem, (int, np.int64)):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        out = {}
        for key in elem:
            if "idx" in key:
                continue
            # make list of values for this key
            values = [d[key] for d in batch]
            out[key] = collate_custom(values)
        return out
    elif isinstance(elem, collections.abc.Sequence):
        # avoid infinite recursion on strings
        if isinstance(elem, string_classes):
            return batch
        # transpose and collate
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    else:
        raise TypeError(f"Unsupported data type: {type(elem)}")
