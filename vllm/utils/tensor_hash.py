# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from collections.abc import Iterable

import torch


def _flatten(items):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _update_tensor_hash(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous()
    hasher.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))


def tensor_hash(tensor_or_list) -> int:
    """Return a deterministic 64-bit hash for a tensor or nested tensor list."""
    hasher = hashlib.sha256()

    if isinstance(tensor_or_list, torch.Tensor):
        _update_tensor_hash(hasher, tensor_or_list)
    elif isinstance(tensor_or_list, Iterable):
        for item in _flatten(tensor_or_list):
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(item)!r}")
            _update_tensor_hash(hasher, item)
    else:
        raise TypeError(
            f"Expected torch.Tensor or iterable, got {type(tensor_or_list)!r}"
        )

    return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)
