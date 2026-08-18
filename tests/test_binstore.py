import numpy as np
import pytest

from afterimage.runtime.binstore import (
    BinaryWeightReader, BinaryWeightWriter, blobref_to_dict,
)


def test_write_then_read_roundtrip_exact(tmp_path):
    path = tmp_path / "weights.bin"
    arrays = [
        np.random.randint(0, 256, size=1000, dtype=np.uint8),
        np.random.randn(50, 30).astype(np.float32),
        np.array([1, 2, 3], dtype=np.int32),
    ]
    refs = []
    with BinaryWeightWriter(path) as w:
        for a in arrays:
            refs.append(w.write(a))

    with BinaryWeightReader(path) as r:
        for a, ref in zip(arrays, refs):
            got = r.read(blobref_to_dict(ref))
            assert np.array_equal(got, a)
            assert got.dtype == a.dtype
            assert got.shape == a.shape


def test_offsets_are_sequential_and_non_overlapping(tmp_path):
    path = tmp_path / "w.bin"
    with BinaryWeightWriter(path) as w:
        r1 = w.write(np.zeros(100, dtype=np.uint8))
        r2 = w.write(np.zeros(50, dtype=np.uint8))
        r3 = w.write(np.zeros(200, dtype=np.uint8))
    assert r1.offset == 0
    assert r2.offset == 100
    assert r3.offset == 150
    assert path.stat().st_size == 350


def test_random_access_does_not_require_sequential_reads(tmp_path):
    """The whole point: seek+read one blob without touching the others."""
    path = tmp_path / "w.bin"
    arrays = [np.full(1000, i, dtype=np.uint8) for i in range(10)]
    refs = []
    with BinaryWeightWriter(path) as w:
        for a in arrays:
            refs.append(blobref_to_dict(w.write(a)))

    with BinaryWeightReader(path) as r:
        # read in reverse order, and only every other one
        for i in [9, 7, 5, 3, 1]:
            got = r.read(refs[i])
            assert np.all(got == i)


def test_empty_array_roundtrips(tmp_path):
    path = tmp_path / "w.bin"
    with BinaryWeightWriter(path) as w:
        ref = blobref_to_dict(w.write(np.array([], dtype=np.uint8)))
    with BinaryWeightReader(path) as r:
        got = r.read(ref)
        assert got.shape == (0,)


def test_read_row_matches_full_read_bit_exact(tmp_path):
    """The primitive the embedding row-gather lever (streaming_engine.py
    lever 2) depends on for correctness: reading row i via read_row must be
    IDENTICAL to reading the whole table and slicing row i, for every row --
    not just close, bit-for-bit, since these are reinterpreted as bf16 bit
    patterns downstream and any mismatch would silently corrupt embeddings."""
    path = tmp_path / "w.bin"
    vocab, hidden = 37, 24
    table = np.random.randint(-30000, 30000, size=(vocab, hidden), dtype=np.int16)

    with BinaryWeightWriter(path) as w:
        ref = w.write(table)

    with BinaryWeightReader(path) as r:
        full = r.read(blobref_to_dict(ref))
        row_nbytes = hidden * 2  # int16 -> 2 bytes/element
        for i in range(vocab):
            row = r.read_row(ref.offset, i, row_nbytes, "int16")
            assert np.array_equal(row, full[i]), f"row {i} mismatch"


def test_read_row_does_not_touch_neighbouring_rows(tmp_path):
    """Reading row i must be independent of what's stored at other rows --
    guards against an off-by-one in the offset arithmetic that would read
    the wrong row without necessarily crashing."""
    path = tmp_path / "w.bin"
    hidden = 8
    rows = [np.full(hidden, i, dtype=np.int16) for i in range(5)]
    table = np.stack(rows)

    with BinaryWeightWriter(path) as w:
        ref = w.write(table)

    with BinaryWeightReader(path) as r:
        for i in range(5):
            got = r.read_row(ref.offset, i, hidden * 2, "int16")
            assert np.all(got == i), f"row {i} leaked data from a neighbour: {got}"
