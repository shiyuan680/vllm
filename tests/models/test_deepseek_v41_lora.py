# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA interface of the DeepSeek-V4.1 entry point.

The registry resolves every DeepSeek-V4.1 checkpoint, text-only included, to
the multimodal wrapper, and ``LoRAConfig`` validates that class. It therefore
has to carry the LoRA interface itself; the inner text backbone implementing
``SupportsLoRA`` is not enough.
"""

from vllm.model_executor.models import supports_lora
from vllm.models.deepseek_v41.nvidia.model import DeepseekV41LLMForCausalLM
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM


def test_entry_point_implements_the_lora_interface():
    assert supports_lora(DeepseekV41ForCausalLM)
    assert supports_lora(DeepseekV41LLMForCausalLM)


def test_entry_point_exposes_the_packed_projections():
    mapping = DeepseekV41ForCausalLM.packed_modules_mapping
    assert mapping["fused_wqa_wkv"] == ["wq_a", "wkv"]
    assert mapping["fused_wkv_wgate"] == ["wkv", "wgate"]
    assert mapping["gate_up_proj"] == ["w1", "w3"]
    # The vision tower's own fused projections, for tower adapters.
    assert mapping["wqkv"] == ["wqkv"]
    assert mapping["w1"] == ["w1"]
    assert DeepseekV41ForCausalLM.lora_skip_prefixes == ["mtp."]


def test_entry_point_maps_the_multimodal_modules():
    # The LoRA manager reads this off the instance before wrapping modules.
    wrapper = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    mm_mapping = wrapper.get_mm_mapping()
    assert mm_mapping.language_model == ["language_model"]
    assert mm_mapping.connector == ["aligner"]
    assert mm_mapping.tower_model == ["vision."]
    # Tower/connector adapters are opt-in but supported.
    assert DeepseekV41ForCausalLM.supports_tower_connector_lora
