import pytest

from memory_engine.dreaming.config import DreamConfig


def test_dream_config_defaults():
    config = DreamConfig()

    assert config.max_memories_per_job == 100
    assert config.max_relationships_per_job == 50
    assert config.max_duplicate_candidates_per_job == 50
    assert config.max_conflict_candidates_per_job == 50
    assert config.max_attempts == 3


@pytest.mark.parametrize(
    "field",
    [
        "max_memories_per_job",
        "max_relationships_per_job",
        "max_duplicate_candidates_per_job",
        "max_conflict_candidates_per_job",
        "max_attempts",
    ],
)
def test_dream_config_rejects_zero_limits(field):
    with pytest.raises(ValueError, match=field):
        DreamConfig(**{field: 0})


def test_dream_config_rejects_negative_limits():
    with pytest.raises(ValueError, match="max_attempts"):
        DreamConfig(max_attempts=-1)