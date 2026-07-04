# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual e2e test for WeightChecker on a real Qwen3 MoE model.

This test intentionally defaults to a large model and is skipped unless
RUN_QWEN3_MOE_WEIGHT_CHECKER=1 is set.

Example:
    $env:RUN_QWEN3_MOE_WEIGHT_CHECKER="1"
    $env:MODEL_NAME="Qwen/Qwen3-30B-A3B"
    $env:TENSOR_PARALLEL_SIZE="4"
    pytest tests/weight_loading/test_weight_checker_qwen3_moe.py -s
"""

import os

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.utils.weight_checker import WeightChecker

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
DISTRIBUTED_EXECUTOR_BACKEND = os.environ.get("DISTRIBUTED_EXECUTOR_BACKEND") or None
DTYPE = os.environ.get("DTYPE", "bfloat16")


def _get_weight_checker(worker) -> WeightChecker:
    checker = getattr(worker, "_weight_checker_e2e", None)
    if checker is None:
        checker = WeightChecker(worker.model_runner)
        worker._weight_checker_e2e = checker
    return checker


def _weight_checker_snapshot(worker):
    return _get_weight_checker(worker).handle("snapshot")


def _weight_checker_compare(worker, allow_quant_error: bool = True):
    return _get_weight_checker(worker).handle(
        "compare", allow_quant_error=allow_quant_error
    )


def _weight_checker_checksum(worker):
    return _get_weight_checker(worker).handle("checksum")


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN3_MOE_WEIGHT_CHECKER") != "1",
    reason="Set RUN_QWEN3_MOE_WEIGHT_CHECKER=1 to run this large-model e2e test.",
)
@pytest.mark.skipif(
    torch.accelerator.device_count() < TENSOR_PARALLEL_SIZE,
    reason="Not enough accelerator devices for requested tensor parallelism.",
)
def test_qwen3_30b_moe_weight_checker_e2e():
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    llm = LLM(
        model=MODEL_NAME,
        dtype=DTYPE,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        distributed_executor_backend=DISTRIBUTED_EXECUTOR_BACKEND,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
    )
    outputs = llm.generate(["Give a one-sentence description of vLLM."], sampling_params)
    assert outputs
    assert outputs[0].outputs[0].text

    checksums_before = llm.collective_rpc(_weight_checker_checksum)
    assert len(checksums_before) == TENSOR_PARALLEL_SIZE
    assert all("per_gpu_checksum" in item for item in checksums_before)
    assert all(item["checksums"] for item in checksums_before)

    assert llm.collective_rpc(_weight_checker_snapshot) == [None] * TENSOR_PARALLEL_SIZE

    outputs = llm.generate(["Name the capital of France."], sampling_params)
    assert outputs
    assert outputs[0].outputs[0].text

    assert llm.collective_rpc(_weight_checker_compare) == [None] * TENSOR_PARALLEL_SIZE

    checksums_after = llm.collective_rpc(_weight_checker_checksum)
    assert [
        item["per_gpu_checksum"] for item in checksums_after
    ] == [
        item["per_gpu_checksum"] for item in checksums_before
    ]
