import pytest

from afterimage.runtime.config import EngineConfig


def test_default_is_lossless():
    """The default must be lossless -- the AirLLM comparison depends on it,
    and a silent lossy default would invalidate every accuracy claim."""
    cfg = EngineConfig()
    assert cfg.quantize is None
    assert cfg.is_lossless
    assert "LOSSLESS" in cfg.describe()


def test_q8_is_opt_in_and_declares_itself_lossy():
    cfg = EngineConfig(quantize="q8")
    assert not cfg.is_lossless
    assert "NOT bit-exact" in cfg.describe()


def test_rejects_unknown_quantize_mode():
    with pytest.raises(ValueError, match="quantize must be"):
        EngineConfig(quantize="q4")


def test_rejects_non_power_of_two_block_chunks():
    with pytest.raises(ValueError, match="power of 2"):
        EngineConfig(block_chunks=37)
    EngineConfig(block_chunks=32)  # valid


def test_rejects_out_of_range_max_bits():
    with pytest.raises(ValueError):
        EngineConfig(max_bits=17)
    with pytest.raises(ValueError):
        EngineConfig(max_bits=0)


def test_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        EngineConfig(chunk_size=0)
