# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The C++ delta checker against the Python check pass."""

import zlib

import numpy as np
import pytest

from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.staging import StagedDelta, join_nonzero, nonzero_bytes, write_staging
from modelexpress_rl.utils import compress_delta

pytestmark = pytest.mark.skipif(
    receiver_module._delta_ext is None, reason="built without the C++ delta checker"
)

META = {"compression_format": "zstd", "checksum_format": "adler32"}
# Around the checker's 1 MiB chunks, and tensors of one byte or none.
SIZES = {
    "empty": 0,
    "byte": 1,
    "small": 3072,
    "short_of_a_chunk": (1 << 20) - 3,
    "chunk": 1 << 20,
    "past_a_chunk": (1 << 20) + 1,
    "chunks": 3 * (1 << 20) + 7,
}


def _checksum(data):
    return f"{zlib.adler32(data):08x}"


def _sparse(rng, size, density):
    return np.where(rng.random(size) < density, rng.integers(1, 256, size), 0).astype(np.uint8)


@pytest.fixture
def scene(tmp_path):
    """Checkpoint files, two deltas staged on them, and the delta under check."""
    rng = np.random.default_rng(9)
    path = tmp_path / "model.bin"
    files = rng.integers(0, 256, sum(SIZES.values()), dtype=np.uint8)
    files.tofile(path)
    locations, offset = {}, 0
    for name, size in SIZES.items():
        locations[name] = (path, offset, size)
        offset += size
    head = {name: files[start : start + size].copy() for name, (_p, start, size) in locations.items()}
    layers = []
    for index in range(2):
        tensors = {}
        for name, size in SIZES.items():
            xor = _sparse(rng, size, 0.01)
            head[name] ^= xor
            tensors[name] = (size, *join_nonzero(size, [nonzero_bytes(xor, 0)]))
        directory = tmp_path / f"layer{index}"
        directory.mkdir()
        write_staging(directory, base_version=f"v{index}", tensors=tensors)
        layers.append(StagedDelta(directory))
    delta = {name: _sparse(rng, size, 0.002) for name, size in SIZES.items()}
    checkpoint = receiver_module._LocalCheckpoint.__new__(receiver_module._LocalCheckpoint)
    checkpoint.locations = locations
    yield checkpoint, head, layers, delta
    for layer in layers:
        layer.close()


def _passes(head, layers, delta, bases, frames=None):
    frames = frames or {name: compress_delta(xor).tobytes() for name, xor in delta.items()}
    return [
        receiver_module._TensorPass(
            name,
            check=memoryview(frames[name]),
            checksum=_checksum(head[name] ^ delta[name]),
            overlay=tuple(layer.tensor(name) for layer in layers),
            base_checksum=bases.get(name),
        )
        for name in delta
    ]


def _run(monkeypatch, checkpoint, passes, native):
    monkeypatch.setenv("MX_REFIT_NATIVE_CHECK", str(native).lower())
    sink, stats = {}, {}
    checkpoint._run_pass(passes, index_metadata=META, sink=sink, stats=stats)
    return sink, stats


@pytest.mark.parametrize("bases", ["known", "unknown", "stale"])
def test_native_checks_match_the_python_pass(monkeypatch, scene, bases):
    checkpoint, head, layers, delta = scene
    known = {
        "known": {name: _checksum(value) for name, value in head.items()},
        "unknown": {},
        "stale": {name: "00000002" for name in head},
    }[bases]
    calls = []
    check = receiver_module._delta_ext.check
    monkeypatch.setattr(
        receiver_module._delta_ext, "check", lambda *args: calls.append(len(args[4])) or check(*args)
    )

    native, native_stats = _run(monkeypatch, checkpoint, _passes(head, layers, delta, known), True)
    python, python_stats = _run(monkeypatch, checkpoint, _passes(head, layers, delta, known), False)

    assert calls == [len(SIZES)]
    assert native_stats == python_stats
    assert native_stats == {
        "known": {"sparse_checks": len(SIZES), "sparse_fallbacks": 0},
        "unknown": {"sparse_checks": 0, "sparse_fallbacks": 0},
        "stale": {"sparse_checks": 0, "sparse_fallbacks": len(SIZES)},
    }[bases]
    for name, xor in delta.items():
        expected = np.flatnonzero(xor)
        for sink in (native, python):
            nbytes, positions, values = sink[name]
            assert nbytes == SIZES[name] and positions.dtype == np.uint32
            assert positions.tolist() == expected.tolist()
            assert values.tolist() == xor[expected].tolist()


@pytest.mark.parametrize("sparse", [True, False])
def test_a_native_check_rejects_a_wrong_target(monkeypatch, scene, sparse):
    checkpoint, head, layers, delta = scene
    bases = {name: _checksum(value) for name, value in head.items()} if sparse else {}
    passes = _passes(head, layers, delta, bases)
    passes[2] = receiver_module._TensorPass(
        passes[2].name,
        check=passes[2].check,
        checksum="00000003",
        overlay=passes[2].overlay,
        base_checksum=passes[2].base_checksum,
    )

    with pytest.raises(ValueError, match="target checksum differs for 'small'"):
        _run(monkeypatch, checkpoint, passes, True)


def test_a_delta_in_several_frames_checks_natively(monkeypatch, scene):
    checkpoint, head, layers, delta = scene
    xor = delta["chunks"]
    frames = {name: compress_delta(value).tobytes() for name, value in delta.items()}
    frames["chunks"] = compress_delta(xor[:100_000]).tobytes() + compress_delta(xor[100_000:]).tobytes()
    bases = {name: _checksum(value) for name, value in head.items()}

    sink, stats = _run(monkeypatch, checkpoint, _passes(head, layers, delta, bases, frames), True)

    assert stats["sparse_checks"] == len(SIZES)
    assert sink["chunks"][1].tolist() == np.flatnonzero(xor).tolist()


@pytest.mark.parametrize("damage", ["long", "short", "trailing", "truncated"])
def test_malformed_deltas_fail_as_in_python(monkeypatch, scene, damage):
    checkpoint, head, layers, delta = scene
    frames = {name: compress_delta(xor).tobytes() for name, xor in delta.items()}
    xor = delta["chunk"]
    frames["chunk"] = {
        "long": compress_delta(np.concatenate([xor, [1]]).astype(np.uint8)).tobytes(),
        "short": compress_delta(xor[:-1]).tobytes(),
        "trailing": frames["chunk"] + b"junk",
        "truncated": frames["chunk"][:-16],
    }[damage]
    bases = {name: _checksum(value) for name, value in head.items()}
    failures = []
    for native in (True, False):
        with pytest.raises(Exception) as caught:
            _run(monkeypatch, checkpoint, _passes(head, layers, delta, bases, frames), native)
        failures.append((type(caught.value), str(caught.value)))

    assert failures[0] == failures[1]
