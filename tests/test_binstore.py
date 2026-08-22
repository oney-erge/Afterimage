import json

import numpy as np
import pytest

from afterimage.runtime.binstore import (
    BinaryWeightReader, BinaryWeightWriter, blobref_to_dict, verify_store,
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


def test_read_many_coalesces_contiguous_blobs_bit_exact(tmp_path):
    path = tmp_path / "w.bin"
    arrays = [
        np.arange(64, dtype=np.uint8),
        np.arange(20, dtype=np.float32).reshape(4, 5),
        np.arange(11, dtype=np.int32),
    ]
    with BinaryWeightWriter(path) as writer:
        refs = [blobref_to_dict(writer.write(array)) for array in arrays]

    with BinaryWeightReader(path) as reader:
        decoded, calls, nbytes = reader.read_many(refs, max_extent_bytes=4096)

    assert calls == 1
    assert nbytes == path.stat().st_size
    assert all(np.array_equal(left, right)
               for left, right in zip(decoded, arrays))


def test_read_many_respects_extent_bound_and_original_order(tmp_path):
    path = tmp_path / "w.bin"
    arrays = [np.full(32, value, dtype=np.uint8) for value in range(3)]
    with BinaryWeightWriter(path) as writer:
        refs = [blobref_to_dict(writer.write(array)) for array in arrays]

    with BinaryWeightReader(path) as reader:
        decoded, calls, nbytes = reader.read_many(
            [refs[2], refs[0], refs[1]], max_extent_bytes=48)

    assert calls == 3
    assert nbytes == 96
    assert [int(array[0]) for array in decoded] == [2, 0, 1]


def test_read_many_returns_writable_torch_consumable_arrays_even_when_unaligned(tmp_path):
    """read_many's memory contract is deliberately different from read()'s:
    non-owning, and not itemsize-aligned whenever a blob's offset within its
    merged extent isn't a multiple of that dtype's size. Odd-length leading
    blobs force later blobs off their natural alignment, exercising exactly
    that case: values must still be exact and torch must still accept the
    result without raising."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "w.bin"
    arrays = [
        np.arange(7, dtype=np.uint8),                    # odd size -> misaligns what follows
        np.arange(5, dtype=np.uint32),
        np.arange(3, dtype=np.float32),
        np.arange(9, dtype=np.uint16).reshape(3, 3),
    ]
    with BinaryWeightWriter(path) as writer:
        refs = [blobref_to_dict(writer.write(array)) for array in arrays]

    with BinaryWeightReader(path) as reader:
        decoded, calls, nbytes = reader.read_many(
            refs, max_gap_bytes=0, max_extent_bytes=1 << 28, verify=True)

    assert calls == 1
    for original, got in zip(arrays, decoded):
        assert np.array_equal(original, got)
        assert got.flags.writeable
        assert not got.flags.owndata
        torch.from_numpy(got)  # must not raise for any dtype/alignment here


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


def test_verified_read_of_intact_data_succeeds(tmp_path):
    path = tmp_path / "w.bin"
    arr = np.random.randn(200).astype(np.float32)
    with BinaryWeightWriter(path) as w:
        ref = blobref_to_dict(w.write(arr))
    with BinaryWeightReader(path) as r:
        got = r.read(ref, verify=True)
        assert np.array_equal(got, arr)


def test_verified_read_of_corrupted_data_raises(tmp_path):
    """The entire point of P0-2: a corrupted weights.bin must be DETECTED,
    not silently served as if it were the correct weight."""
    path = tmp_path / "w.bin"
    arr = np.random.randint(0, 256, size=500, dtype=np.uint8)
    with BinaryWeightWriter(path) as w:
        ref = blobref_to_dict(w.write(arr))

    # flip a byte in the middle of the blob, simulating disk corruption
    with open(path, "r+b") as f:
        f.seek(ref["offset"] + 250)
        f.write(bytes([arr[250] ^ 0xFF]))

    with BinaryWeightReader(path) as r:
        with pytest.raises(ValueError, match="checksum mismatch"):
            r.read(ref, verify=True)


def test_unverified_read_does_not_raise_on_corrupted_data(tmp_path):
    """verify=False (the default) must stay silent about corruption -- it
    exists specifically so the hot per-token read path never pays CRC32
    cost by default; verify_store() is where that cost belongs."""
    path = tmp_path / "w.bin"
    arr = np.random.randint(0, 256, size=100, dtype=np.uint8)
    with BinaryWeightWriter(path) as w:
        ref = blobref_to_dict(w.write(arr))
    with open(path, "r+b") as f:
        f.seek(ref["offset"])
        f.write(bytes([arr[0] ^ 0xFF]))
    with BinaryWeightReader(path) as r:
        got = r.read(ref)  # no verify -- must not raise
        assert not np.array_equal(got, arr)  # but the corruption is real


def test_verify_store_passes_on_an_intact_store(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with BinaryWeightWriter(store / "weights.bin") as w:
        ref_a = blobref_to_dict(w.write(np.arange(50, dtype=np.int32)))
        ref_b = blobref_to_dict(w.write(np.arange(30, dtype=np.float32)))
    manifest = {"tensors": {
        "a": {"blobs": {"raw": ref_a}},
        "b": {"blobs": {"raw": ref_b}},
    }}
    (store / "manifest.json").write_text(json.dumps(manifest))

    ok, bad = verify_store(store)
    assert ok
    assert bad == []


def test_verify_store_flags_corrupted_tensors_by_key(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with BinaryWeightWriter(store / "weights.bin") as w:
        ref_a = blobref_to_dict(w.write(np.arange(50, dtype=np.int32)))
        ref_b = blobref_to_dict(w.write(np.arange(30, dtype=np.float32)))
    manifest = {"tensors": {
        "a": {"blobs": {"raw": ref_a}},
        "b": {"blobs": {"raw": ref_b}},
    }}
    (store / "manifest.json").write_text(json.dumps(manifest))

    with open(store / "weights.bin", "r+b") as f:
        f.seek(ref_b["offset"])
        f.write(b"\xff" * 4)

    ok, bad = verify_store(store)
    assert not ok
    assert bad == ["b"]


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
