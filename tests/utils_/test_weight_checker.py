# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.utils import WeightChecker
from vllm.utils.tensor_hash import tensor_hash
from vllm.utils.weight_checker_comparator import RawComparable, compare_weights

pytestmark = pytest.mark.cpu_test


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("tracked_buffer", torch.arange(4, dtype=torch.float32))
        self.register_buffer("cos_sin_cache", torch.ones(4))


class FakeModelRunner:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.tp_rank = 0
        self.tp_size = 1
        self.dp_rank = None
        self.dp_size = 1
        self.pp_rank = 0
        self.pp_size = 1


def make_checker() -> tuple[WeightChecker, TinyModel]:
    torch.manual_seed(0)
    model = TinyModel()
    model.linear.bias._skip_weight_check = True
    return WeightChecker(FakeModelRunner(model)), model


def test_weight_checker_snapshot_and_compare_detects_mutation():
    checker, model = make_checker()

    assert checker.handle("snapshot") is None
    assert checker.handle("compare") is None

    with torch.no_grad():
        model.linear.weight.add_(1.0)

    with pytest.raises(ValueError, match="linear.weight"):
        checker.handle("compare")


def test_weight_checker_checksum_skips_marked_and_non_persistent_buffers():
    checker, _ = make_checker()

    checksum = checker.handle("checksum")

    assert checksum["parallelism_info"] == {
        "tp_rank": 0,
        "tp_size": 1,
        "dp_rank": 0,
        "dp_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "rank": 0,
        "size": 1,
    }
    assert "linear.weight" in checksum["checksums"]
    assert "tracked_buffer" in checksum["checksums"]
    assert "linear.bias" not in checksum["checksums"]
    assert "cos_sin_cache" not in checksum["checksums"]
    assert isinstance(checksum["per_gpu_checksum"], str)
    assert len(checksum["per_gpu_checksum"]) == 64


def test_weight_checker_reset_tensors_preserves_non_persistent_buffers():
    checker, model = make_checker()
    old_weight = model.linear.weight.detach().clone()
    old_cache = model.cos_sin_cache.detach().clone()

    checker.handle("reset_tensors")

    assert not torch.equal(model.linear.weight, old_weight)
    assert torch.equal(model.cos_sin_cache, old_cache)


def test_weight_checker_rejects_unknown_action():
    checker, _ = make_checker()

    with pytest.raises(ValueError, match="Unsupported action"):
        checker.handle("unknown")


def test_raw_comparable_compare_weights_reports_equal_and_different_tensors():
    lhs = RawComparable(torch.tensor([1.0, 2.0, 3.0]))
    rhs = RawComparable(torch.tensor([1.0, 2.0, 3.0]))
    result = compare_weights(lhs, rhs)
    assert result.equal
    assert result.max_abs_err == 0.0
    assert result.mean_abs_err == 0.0
    assert result.num_exceed == 0

    different = compare_weights(lhs, RawComparable(torch.tensor([1.0, 2.5, 3.0])))
    assert not different.equal
    assert different.max_abs_err == 0.5
    assert different.num_exceed == 1


def test_tensor_hash_is_stable_for_nested_tensor_lists():
    first = torch.arange(4, dtype=torch.int32)
    second = torch.arange(4, 8, dtype=torch.int32)

    assert tensor_hash([first, [second]]) == tensor_hash(
        [first.clone(), [second.clone()]]
    )
    assert tensor_hash([first, [second]]) != tensor_hash([first + 1, [second]])
