import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from ada.config import ADAConfig


@pytest.fixture
def tiny_config() -> ADAConfig:
    """A deliberately tiny config so unit tests run fast on CPU."""
    return ADAConfig(
        name="test-tiny",
        vocab_size=100,
        hidden_dim=32,
        num_layers=2,
        max_seq_len=16,
        ssm_state_dim=8,
        ssm_num_heads=4,
        ssm_head_dim=8,
        ssm_conv_kernel=4,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_mult=2.0,
        num_query_heads=4,
        num_kv_heads=2,
        depth_controller_hidden=8,
        batch_size_tokens=64,
        micro_batch_size=2,
        training_tokens=1000,
    )
