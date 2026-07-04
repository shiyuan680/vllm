# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import NamedTuple

import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4FusedMoE,
    ModelOptNvFp4LinearMethod,
    ModelOptNvFp4W4A16LinearMethod,
)

# Chunk to avoid high peak memory while comparing large models.
CHUNK_NUMEL = 64 * 1024 * 1024


class CompareResult(NamedTuple):
    equal: bool
    max_abs_err: float
    mean_abs_err: float
    num_exceed: int


class ComparableWeight:
    """Base comparable-weight class; one subclass per precision or raw tensor."""

    @staticmethod
    def _quant_ulp(w_q: torch.Tensor) -> torch.Tensor:
        finfo = torch.finfo(w_q.dtype)
        x = w_q.to(torch.float32).abs()
        _, exponent = torch.frexp(x)
        binade = torch.exp2((exponent - 1).to(torch.float32))
        binade = binade.masked_fill(x < finfo.smallest_normal, finfo.smallest_normal)
        return binade * finfo.eps

    def iter_chunks(self) -> Iterable[tuple[torch.Tensor, torch.Tensor | None]]:
        raise NotImplementedError

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        raise NotImplementedError


def _upcast_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.to(torch.float32)
    if scale.dtype == torch.uint8:
        return (scale.to(torch.int32) << 23).view(torch.float32)
    return scale.to(torch.float32)


def _block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    block_n, block_k = block_size
    *_, n, k = x_q_block.shape
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]
    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)


class Fp8BlockComparable(ComparableWeight):
    """Block-wise FP8 comparable used by DeepSeek-style checkpoints."""

    def __init__(self, w_q: torch.Tensor, w_s: torch.Tensor):
        self.w_q = w_q
        self.w_s = w_s

    def __repr__(self) -> str:
        return f"fp8_block(shape={tuple(self.w_q.shape)} dtype={self.w_q.dtype})"

    @staticmethod
    def _normalize_scale(w_s: torch.Tensor) -> torch.Tensor:
        return _upcast_e8m0_to_fp32(w_s)

    @staticmethod
    def _infer_block_size(w_q: torch.Tensor, w_s: torch.Tensor) -> list[int]:
        block_n = -(-w_q.shape[-2] // w_s.shape[-2])
        block_k = -(-w_q.shape[-1] // w_s.shape[-1])
        return [block_n, block_k]

    @staticmethod
    def _iter_quant_chunks(w_q: torch.Tensor, w_s: torch.Tensor, block_n: int):
        q3 = w_q.reshape(-1, *w_q.shape[-2:])
        s3 = w_s.reshape(-1, *w_s.shape[-2:])
        n, k = q3.shape[-2:]
        rows = max(block_n, CHUNK_NUMEL // max(k, 1) // block_n * block_n)
        for b in range(q3.shape[0]):
            for r0 in range(0, n, rows):
                r1 = min(r0 + rows, n)
                yield q3[b, r0:r1], s3[b, r0 // block_n : -(-r1 // block_n)]

    def _scale_and_block_size(self):
        scale = self._normalize_scale(self.w_s)
        return scale, self._infer_block_size(self.w_q, scale)

    def iter_chunks(self):
        scale, block_size = self._scale_and_block_size()
        for q, s_chunk in self._iter_quant_chunks(self.w_q, scale, block_size[0]):
            yield (
                _block_quant_dequant(q, s_chunk, block_size, dtype=torch.bfloat16),
                _block_quant_dequant(
                    self._quant_ulp(q), s_chunk, block_size, dtype=torch.float32
                ),
            )

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        scale, block_size = self._scale_and_block_size()
        return _block_quant_dequant(self.w_q, scale, block_size, dtype=dtype)


class RawComparable(ComparableWeight):
    """Bitwise equality compare on a raw tensor."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"raw(shape={tuple(self.tensor.shape)} dtype={self.tensor.dtype})"

    def iter_chunks(self):
        flat = self.tensor.reshape(-1)
        for start in range(0, flat.numel(), CHUNK_NUMEL):
            yield flat[start : start + CHUNK_NUMEL], None

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return self.tensor


def compare_weights(expect: ComparableWeight, actual: ComparableWeight) -> CompareResult:
    """Chunked element-wise compare in ComparableWeight space."""
    equal = True
    max_abs_err = torch.zeros((), dtype=torch.float32)
    sum_abs_err = 0.0
    num_exceed = 0
    numel = 0
    for (expect_dq, expect_tol), (actual_dq, actual_tol) in zip(
        expect.iter_chunks(), actual.iter_chunks(), strict=True
    ):
        assert expect_dq.shape == actual_dq.shape, (
            f"{expect_dq.shape=} {actual_dq.shape=}"
        )
        numel += expect_dq.numel()
        abs_diff = (actual_dq.float() - expect_dq.float()).abs()
        if torch.all(abs_diff == 0):
            continue
        equal = False
        tol = (
            0.0
            if expect_tol is None or actual_tol is None
            else expect_tol + actual_tol
        )
        max_abs_err = torch.maximum(max_abs_err, abs_diff.max().cpu())
        sum_abs_err += abs_diff.sum().item()
        num_exceed += int((~(abs_diff <= tol)).sum())
    return CompareResult(
        equal,
        max_abs_err.item(),
        sum_abs_err / max(numel, 1),
        num_exceed,
    )


def select_comparable_weight(quant_method) -> type[ComparableWeight] | None:
    """Map a module's quant_method to its ComparableWeight."""
    if (
        isinstance(quant_method, (Fp8LinearMethod, Fp8MoEMethod))
        and quant_method.block_quant
        and not getattr(quant_method, "use_mxfp8", False)
    ):
        return Fp8BlockComparable
    if isinstance(
        quant_method,
        (
            ModelOptNvFp4LinearMethod,
            ModelOptNvFp4W4A16LinearMethod,
            ModelOptNvFp4FusedMoE,
        ),
    ):
        raise NotImplementedError(
            f"weight checker has no ComparableWeight for {type(quant_method).__name__}"
        )
    return None
