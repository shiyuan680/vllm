# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import logging
import time
from collections.abc import Iterable
from typing import NamedTuple

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict

from vllm.utils.tensor_hash import tensor_hash
from vllm.utils.weight_checker_comparator import (
    CHUNK_NUMEL,
    ComparableWeight,
    RawComparable,
    compare_weights,
    select_comparable_weight,
)

logger = logging.getLogger(__name__)


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParallelismInfo(_StrictBaseModel):
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int
    pp_size: int
    rank: int
    size: int


class ChecksumInfo(_StrictBaseModel):
    checksums: dict[str, str]
    per_gpu_checksum: str
    parallelism_info: ParallelismInfo


class CheckEntry(NamedTuple):
    name: str
    should_compare: bool
    comparable: ComparableWeight


class QuantizedWeight(NamedTuple):
    comparable_cls: type[ComparableWeight]
    scale_name: str


_NON_PERSISTENT_BUFFER_PATTERNS = (
    "cos_sin_cache",
    "inv_freq",
    "freqs_cis",
    "_weight_fp32",
)


def _is_non_persistent_buffer_name(name: str) -> bool:
    return any(pat in name for pat in _NON_PERSISTENT_BUFFER_PATTERNS)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._snapshot_tensors: dict[str, torch.Tensor] | None = None

    def handle(self, action: str, allow_quant_error: bool = False) -> dict | None:
        logger.info(
            "[WeightChecker] handle action=%s allow_quant_error=%s",
            action,
            allow_quant_error,
        )
        if action == "snapshot":
            return self._snapshot()
        if action == "reset_tensors":
            return self._reset_tensors()
        if action == "compare":
            return self._compare(allow_quant_error=allow_quant_error)
        if action == "checksum":
            return self._compute_checksum()
        raise ValueError(f"Unsupported action={action!r}")

    def _snapshot(self) -> None:
        named_tensors = [
            (name, param.data.detach().cpu().clone())
            for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(named_tensors), (
            "should not have duplicated tensor name"
        )

    def _reset_tensors(self) -> None:
        with torch.no_grad():
            for name, param in self._model_state():
                if _is_non_persistent_buffer_name(name):
                    continue
                param.copy_(_random_like(param))

    def _compare(self, allow_quant_error: bool = False) -> None:
        assert self._snapshot_tensors is not None

        quantized_set = _build_quantized_set(self._model_runner.model)
        skip_compare_names = {
            name
            for name, param in self._model_state()
            if getattr(param, "_skip_weight_check", False)
        }
        _check_tensors(
            expect_tensors=_build_check_entries(
                self._snapshot_tensors, skip_compare_names, quantized_set
            ),
            actual_tensors=_build_check_entries(
                dict(self._model_state()), skip_compare_names, quantized_set
            ),
            allow_quant_error=allow_quant_error,
        )

    def _compute_checksum(self) -> dict:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        quantized_set = _build_quantized_set(self._model_runner.model)
        skip_compare_names = {
            name
            for name, param in self._model_state()
            if getattr(param, "_skip_weight_check", False)
        }

        checksums = {}
        for name, should_compare, comparable in _build_check_entries(
            dict(self._model_state()), skip_compare_names, quantized_set
        ):
            if should_compare:
                checksums[name] = _hash_tensor(comparable.dequantize().data)

        h = hashlib.sha256()
        for name in sorted(checksums):
            h.update(name.encode())
            h.update(checksums[name].encode())
        overall = h.hexdigest()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logger.info(
            "[WeightChecker] checksum computed for %d tensors in %.3fs",
            len(checksums),
            elapsed,
        )

        info = ChecksumInfo(
            checksums=checksums,
            per_gpu_checksum=overall,
            parallelism_info=self._parallelism_info(),
        )
        return info.model_dump()

    def _parallelism_info(self) -> ParallelismInfo:
        mr = self._model_runner
        return ParallelismInfo(
            tp_rank=getattr(mr, "tp_rank", 0),
            tp_size=getattr(mr, "tp_size", 1),
            dp_rank=getattr(mr, "dp_rank", 0) or 0,
            dp_size=getattr(mr, "dp_size", 1),
            pp_rank=getattr(mr, "pp_rank", 0),
            pp_size=getattr(mr, "pp_size", 1),
            rank=dist.get_rank() if dist.is_initialized() else 0,
            size=dist.get_world_size() if dist.is_initialized() else 1,
        )

    def _model_state(self):
        yield from self._model_runner.model.named_parameters()
        yield from self._model_runner.model.named_buffers()


def _hash_tensor(t: torch.Tensor) -> str:
    return f"{tensor_hash(t):016x}"


def _check_tensors(
    expect_tensors: Iterable[CheckEntry],
    actual_tensors: Iterable[CheckEntry],
    allow_quant_error: bool = False,
) -> None:
    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, should_compare, expect_comparable), (
        actual_name,
        actual_should_compare,
        actual_comparable,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert should_compare == actual_should_compare, (
            f"{should_compare=} {actual_should_compare=}"
        )
        name = expect_name

        try:
            equal, max_abs_err, mean_abs_err, num_exceed = compare_weights(
                expect_comparable, actual_comparable
            )
        except Exception as e:
            e.add_note(
                f"when handling {name=} expect={expect_comparable!r} "
                f"actual={actual_comparable!r}"
            )
            raise
        if equal:
            good_names.append(name)
            continue
        msg = (
            f"name={name} "
            f"max_abs_err={max_abs_err} "
            f"mean_abs_err={mean_abs_err} "
            f"num_exceed={num_exceed} "
            f"expect={expect_comparable!r} actual={actual_comparable!r} "
        )
        if not should_compare:
            info_messages.append(msg)
        elif allow_quant_error and num_exceed == 0:
            info_messages.append(msg + "(within quantization ULP tolerance)")
        else:
            error_messages.append(msg)

    logger.info("[check_tensors] equal tensors: %s", good_names)
    if info_messages:
        logger.info("[check_tensors] info: %s", info_messages)
    if error_messages:
        raise ValueError("check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        out = torch.empty(shape, device=device, dtype=dtype)
        for chunk in out.view(-1).split(CHUNK_NUMEL):
            chunk.copy_(
                torch.rand(chunk.shape, device=device, dtype=torch.float32).to(dtype)
            )
        return out

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _build_quantized_set(model) -> dict[str, QuantizedWeight]:
    quantized_set = {}
    for module_name, module in model.named_modules():
        comparable_cls = select_comparable_weight(getattr(module, "quant_method", None))
        if comparable_cls is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        own = {name for name, _ in module.named_parameters(recurse=False)}
        for name in own:
            scale = name.replace("weight", "weight_scale_inv")
            if name.endswith("weight") and scale in own:
                quantized_set[prefix + name] = QuantizedWeight(
                    comparable_cls, prefix + scale
                )
    return quantized_set


def _build_check_entries(
    raw: dict[str, torch.Tensor],
    skip_compare_names: set[str],
    quantized_set: dict[str, QuantizedWeight] | None = None,
) -> Iterable[CheckEntry]:
    skip_compare_names = set(skip_compare_names)
    quantized_set = quantized_set or {}
    scale_names = {qw.scale_name for qw in quantized_set.values()}

    for name, tensor in raw.items():
        if name in scale_names:
            continue
        if name in quantized_set:
            qw = quantized_set[name]
            yield CheckEntry(name, True, qw.comparable_cls(tensor, raw[qw.scale_name]))
        else:
            should_compare = name not in skip_compare_names and (
                not _is_non_persistent_buffer_name(name)
            )
            yield CheckEntry(name, should_compare, RawComparable(tensor))
