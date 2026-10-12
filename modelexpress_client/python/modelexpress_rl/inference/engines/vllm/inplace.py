# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a staged XOR delta into a live vLLM model in place.

When a model's post-load processing only rearranges bits, processing the XOR
delta like weights and XORing the result into live storage gives the same bytes
as a reload of the target. Constant parts of processing cancel because zeros are
processed as well. Routed experts take a batched path from the staged bytes;
every other changed tensor goes through the layerwise reload with the final copy
replaced by an XOR.
"""

from __future__ import annotations

import itertools
import logging
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modelexpress_rl.inference.staging import (
    POSITIONS_FILE,
    VALUES_FILE,
    StagedDelta,
    StagedTensor,
)
from modelexpress_rl.utils import index_checkpoint_tensors

logger = logging.getLogger(__name__)

_EXPERT = re.compile(
    r"\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj|w1|w3|w2)\.([A-Za-z_]+)$"
)
_PER_EXPERT = re.compile(r"\.(?:mlp|block_sparse_moe)\.experts\.\d+\.")
# Checkpoint projection names, both the gate/up/down and the w1/w3/w2 conventions.
_PROJECTIONS = {
    "gate_proj": "gate",
    "w1": "gate",
    "up_proj": "up",
    "w3": "up",
    "down_proj": "down",
    "w2": "down",
}
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_ARRANGED_KINDS = ("weight_packed", "weight_scale", "weight")
_TORCH_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class InPlaceUnsupported(RuntimeError):
    """The delta cannot be installed in place; reload the target instead."""


def raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return a byte view that shares storage with ``tensor``."""
    if not tensor.is_contiguous():
        raise InPlaceUnsupported("live tensor is not contiguous")
    return tensor.reshape(-1).view(torch.uint8)


_SAME_SIZE_INTEGER = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """Return a same-size integer view of ``tensor``, valid for any strides."""
    integer = _SAME_SIZE_INTEGER.get(tensor.element_size())
    if integer is None:
        raise InPlaceUnsupported(f"cannot view {tensor.dtype} as integers")
    return tensor.view(integer)


def same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and torch.equal(bits(left), bits(right))


def xor_into(target: torch.Tensor, value: torch.Tensor) -> None:
    """XOR the bits of ``value`` into ``target`` in place."""
    if target.shape == value.shape and target.element_size() == value.element_size():
        bits(target).bitwise_xor_(bits(value))
        return
    # Processing may change the shape of a tensor but not its storage size.
    target_bytes = raw_bytes(target)
    value_bytes = value.contiguous().reshape(-1).view(torch.uint8)
    if target_bytes.numel() != value_bytes.numel():
        raise InPlaceUnsupported("processed tensor size differs from live storage")
    target_bytes.bitwise_xor_(value_bytes)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def layer_key(name: str) -> int | None:
    """Decoder layer index of a dotted tensor or module name, if any."""
    match = _LAYER.search(f".{name}.")
    return int(match[1]) if match else None


def is_per_expert(name: str) -> bool:
    return _PER_EXPERT.search(name) is not None


@contextmanager
def _quiet_readonly_buffers() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        yield


def device_offsets(positions: np.ndarray, device: torch.device) -> torch.Tensor:
    """Staged byte offsets as int64 on ``device``, sent at their stored width."""
    if positions.dtype == np.uint32:
        return torch.from_numpy(positions.view(np.int32)).to(device).to(torch.int64) & 0xFFFFFFFF
    return torch.from_numpy(positions.view(np.int64)).to(device)


def staged_bytes(
    nbytes: int,
    staged: StagedTensor | None,
    device: torch.device,
    *,
    chunk: int = 1 << 24,
) -> torch.Tensor:
    """A tensor's full XOR delta as bytes on ``device``; zeros when unstaged."""
    if device.type == "cpu":
        # Untouched pages of a zeroed host array stay unallocated.
        data = np.zeros(nbytes, dtype=np.uint8)
        if staged is not None:
            data[staged.positions.astype(np.intp, copy=False)] = staged.values
        return torch.from_numpy(data)
    data = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    if staged is not None:
        for start in range(0, staged.positions.size, chunk):
            offsets = device_offsets(staged.positions[start : start + chunk], device)
            data[offsets] = torch.from_numpy(staged.values[start : start + chunk]).to(device)
    return data


@dataclass(frozen=True)
class Peers:
    """Ranks that split reading staged bytes, then gather what each read."""

    rank: int
    size: int
    # Concatenates every peer's equally sized tensor along dimension 0, in rank order.
    all_gather: Callable[[torch.Tensor], torch.Tensor]


# Staged bytes per run that ranks read and gather, and per pass that places them.
_RUN_BYTES = 1 << 25
_PLACE_BYTES = 1 << 23
# Where placement writes the bytes outside this rank's shard.
_DISCARD_BYTES = 1 << 20
# A row length that keeps every offset of a tensor in row zero.
_ONE_ROW = 1 << 62


def staged_runs(
    staged: StagedDelta, begin: int, end: int, *, limit: int = _RUN_BYTES
) -> list[tuple[int, int]]:
    """Split rows ``[begin, end)`` into runs of about ``limit`` staged bytes.

    A run never mixes offset widths, so its offsets read as one integer type.
    """
    runs = []
    counts = staged.counts(begin, end)
    start, total = begin, 0
    for row, count in enumerate(counts.tolist(), begin):
        if row > start and (
            total + count > limit or staged.offset_width(row) != staged.offset_width(start)
        ):
            runs.append((start, row))
            start, total = row, 0
        total += count
    if end > begin:
        runs.append((start, end))
    return runs


@dataclass(frozen=True)
class StagedRun:
    """Offsets and values of consecutive staged rows, on the install device."""

    begin: int
    end: int
    offsets: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class _Share:
    """One rank's part of a run: equal slices of its offset and value bytes."""

    positions: tuple[int, int]
    values: tuple[int, int]
    # Bytes per peer of each file, padded so every peer sends the same size.
    positions_part: int
    values_part: int
    run_bytes: tuple[int, int]


class StagedRunReader:
    """Deliver runs of staged rows to a device in order, read ahead by threads.

    With peers, each rank reads an equal share of a run's bytes and an
    all-gather assembles the run, so each staged byte crosses host memory once
    per peer group rather than once per rank. Peers must read the same runs.
    """

    def __init__(
        self,
        staged: StagedDelta,
        runs: list[tuple[int, int]],
        device: torch.device,
        *,
        peers: Peers | None = None,
        threads: int = 8,
        depth: int = 4,
        piece: int = 1 << 21,
    ) -> None:
        self._staged = staged
        self._runs = runs
        self._device = device
        self._peers = peers
        self._piece = piece
        self._shares = [self._share(begin, end) for begin, end in runs]
        capacity = max(
            (share.positions_part + share.values_part for share in self._shares), default=0
        )
        pin = device.type == "cuda"
        self._buffers = [
            torch.empty(capacity, dtype=torch.uint8, pin_memory=pin)
            for _ in range(min(depth, len(runs)))
        ]
        # The copy out of each buffer, which a read into it must wait for.
        self._copies: list[torch.cuda.Event | None] = [None] * len(self._buffers)
        self._pool = ThreadPoolExecutor(threads, thread_name_prefix="mx-staged-read")
        self._reads: dict[int, list[Future]] = {}
        self._submitted = 0
        self._delivered = 0
        # Time spent waiting for reads, behind the device.
        self.wait_seconds = 0.0
        for _ in self._buffers:
            self._submit()

    def _share(self, begin: int, end: int) -> _Share:
        rank, size = (self._peers.rank, self._peers.size) if self._peers else (0, 1)
        positions, values = self._staged.byte_ranges(begin, end)
        mine, parts = [], []
        for low, high in (positions, values):
            part = -(-(high - low) // size // 16) * 16
            start = min(low + rank * part, high)
            mine.append((start, min(start + part, high)))
            parts.append(part)
        lengths = (positions[1] - positions[0], values[1] - values[0])
        return _Share(mine[0], mine[1], parts[0], parts[1], lengths)

    def _submit(self) -> None:
        index = self._submitted
        if index >= len(self._runs):
            return
        self._submitted += 1
        slot = index % len(self._buffers)
        share = self._shares[index]
        view = memoryview(self._buffers[slot].numpy())
        copy = self._copies[slot]
        reads = []
        for file, (low, high), start in (
            (POSITIONS_FILE, share.positions, 0),
            (VALUES_FILE, share.values, share.positions_part),
        ):
            for offset in range(low, high, self._piece):
                stop = min(offset + self._piece, high)
                target = view[start + offset - low : start + stop - low]
                reads.append(self._pool.submit(self._read, copy, file, offset, target))
        self._reads[index] = reads

    def _read(self, copy: Any, file: str, offset: int, target: memoryview) -> None:
        if copy is not None:
            copy.synchronize()
        self._staged.read_into(file, offset, target)

    def __iter__(self) -> StagedRunReader:
        return self

    def __next__(self) -> StagedRun:
        index = self._delivered
        if index >= len(self._runs):
            raise StopIteration
        started = time.perf_counter()
        for read in self._reads.pop(index):
            read.result()
        self.wait_seconds += time.perf_counter() - started
        self._delivered += 1
        slot = index % len(self._buffers)
        share = self._shares[index]
        width = share.positions_part + share.values_part
        begin, end = self._runs[index]
        if width == 0:
            # Every peer sees the same empty run, so none gathers it.
            self._submit()
            empty = torch.empty(0, dtype=torch.uint8, device=self._device)
            return StagedRun(begin, end, empty.view(torch.int32), empty)
        host = self._buffers[slot][:width]
        if self._device.type == "cuda":
            local = host.to(self._device, non_blocking=True)
            copy = torch.cuda.Event()
            copy.record()
            self._copies[slot] = copy
        else:
            local = host.to(self._device, copy=True)
        self._submit()
        rows = (self._peers.all_gather(local) if self._peers else local).view(-1, width)
        positions_bytes, values_bytes = share.run_bytes
        offsets = rows[:, : share.positions_part].reshape(-1)[:positions_bytes]
        values = rows[:, share.positions_part :].reshape(-1)[:values_bytes]
        if self._staged.offset_width(begin) == 4:
            offsets = offsets.view(torch.int32)
        else:
            offsets = offsets.view(torch.int64)
        return StagedRun(begin, end, offsets, values)

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._reads.clear()


def _to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy a small host array to ``device`` without waiting for queued work."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


class StagedPlacement:
    """Where the staged bytes of consecutive rows land in a layer's flat buffer.

    ``rules`` maps a row to ``(base, row_bytes, width, low)``: byte ``p`` of the
    row lands at ``base + p // row_bytes * width + c`` where ``c`` is
    ``p % row_bytes - low``, when ``0 <= c < width``. Rows without a rule, and
    bytes outside this rank's shard, go to the ``_DISCARD_BYTES`` after ``size``.
    """

    def __init__(
        self,
        staged: StagedDelta,
        rows: tuple[int, int],
        rules: dict[int, tuple[int, int, int, int]],
        size: int,
        device: torch.device,
    ) -> None:
        begin, end = rows
        table = np.array(
            [rules.get(row, (0, 1, 0, 0)) for row in range(begin, end)], dtype=np.int64
        )
        self._begin = begin
        self._size = size
        # One column per rule field, each indexed by row.
        self._columns = _to_device(table.reshape(-1, 4).T, device)
        self._counts = _to_device(staged.counts(begin, end), device)

    def place(self, flat: torch.Tensor, run: StagedRun, *, window: int = _PLACE_BYTES) -> None:
        low, high = run.begin - self._begin, run.end - self._begin
        total = run.values.numel()
        rows = torch.repeat_interleave(
            torch.arange(high - low, dtype=torch.int32, device=flat.device),
            self._counts[low:high],
            output_size=total,
        )
        columns = self._columns[:, low:high]
        for start in range(0, total, window):
            stop = min(start + window, total)
            ids = rows[start:stop]
            base, row_bytes, width, column_low = (columns[field][ids] for field in range(4))
            offsets = run.offsets[start:stop].to(torch.int64)
            if run.offsets.dtype == torch.int32:
                offsets &= 0xFFFFFFFF
            row = torch.div(offsets, row_bytes, rounding_mode="floor")
            column = offsets - row * row_bytes - column_low
            keep = (column >= 0) & (column < width)
            discard = self._size + (offsets & (_DISCARD_BYTES - 1))
            target = torch.where(keep, base + row * width + column, discard)
            flat.index_put_((target,), run.values[start:stop])


def _routed_experts_class() -> type | None:
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    except ImportError:
        return None
    return RoutedExperts


def routed_expert_layers(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """Map decoder layer indices to their routed-expert modules."""
    routed = _routed_experts_class()
    found: dict[int, torch.nn.Module] = {}
    for path, module in model.named_modules():
        if routed is not None and not isinstance(module, routed):
            continue
        if routed is None and type(module).__name__ != "RoutedExperts":
            continue
        key = layer_key(path)
        if key is not None:
            found.setdefault(key, module)
    return found


def _expert_param(meta: dict, prefix: str, kind: str) -> str | None:
    """Name of the load-time parameter holding one checkpoint tensor kind."""
    # Some methods load packed weights into ``w13_weight`` rather than ``w13_weight_packed``.
    names = [f"{prefix}_{kind}"] + ([f"{prefix}_weight"] if kind == "weight_packed" else [])
    return next((name for name in names if name in meta), None)


def expert_rows(
    staged: StagedDelta, expert_names: dict[tuple[int, str, str], str]
) -> tuple[int, int] | None:
    """Rows ``[begin, end)`` that span one layer's staged expert tensors."""
    rows = [staged.row(name) for name in expert_names.values() if name in staged]
    return (min(rows), max(rows) + 1) if rows else None


def arrange_expert_delta(
    layer: torch.nn.Module,
    staged: StagedDelta,
    expert_names: dict[tuple[int, str, str], str],
    metadata: dict[str, dict],
    *,
    device: torch.device,
    runs: Iterator[StagedRun] | None = None,
) -> dict[str, torch.Tensor]:
    """Lay out one layer's expert XOR bytes as its pre-processing parameters.

    Gate and up shard their output rows, down shards its input columns, and the
    loader transposes packed weights when the parameter says so, as vLLM's
    routed-expert weight loader does. ``runs`` delivers the rows of
    ``expert_rows``; without it, this reads them.
    """
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    meta, _buffers = get_layerwise_info(layer).restore_metadata
    parallel = layer.moe_config.moe_parallel_config
    if getattr(parallel, "ep_size", 1) != 1:
        raise InPlaceUnsupported("expert parallelism is not supported")
    tp_size, tp_rank = parallel.tp_size, layer.moe_config.tp_rank
    present = {kind for (_expert, _proj, kind) in expert_names}
    params = {
        kind: (w13, w2)
        for kind in _ARRANGED_KINDS
        if kind in present
        and (w13 := _expert_param(meta, "w13", kind)) is not None
        and (w2 := _expert_param(meta, "w2", kind)) is not None
    }
    changed = {kind for (_expert, _proj, kind), name in expert_names.items() if name in staged}
    if not params or changed - set(params):
        raise InPlaceUnsupported(f"unsupported expert tensors {sorted(changed - set(params))}")
    first = next(iter(params.values()))[0]
    transposed = bool(getattr(meta[first], "is_transposed", False))
    experts = meta[first].shape[0]
    # Every projection of every kind gets one slot of ``flat``, one shard per expert.
    slots: dict[tuple[str, str], tuple[int, tuple[int, int], torch.dtype]] = {}
    rules: dict[int, tuple[int, int, int, int]] = {}
    size = 0
    for kind in params:
        for proj in ("gate", "up", "down"):
            sample = expert_names.get((0, proj, kind))
            if sample is None or len(metadata[sample]["shape"]) != 2:
                raise InPlaceUnsupported(f"expert {proj} {kind} is not a 2-D tensor")
            rows, cols = metadata[sample]["shape"]
            dtype = _TORCH_DTYPES[metadata[sample]["dtype"]]
            itemsize = torch.empty((), dtype=dtype).element_size()
            if proj != "down":
                shard = (rows // tp_size, cols)
                shard_bytes = shard[0] * cols * itemsize
                # One byte range per expert, so every offset is in row zero.
                rule = (_ONE_ROW, shard_bytes, tp_rank * shard_bytes)
            else:
                shard = (rows, cols // tp_size)
                width = shard[1] * itemsize
                shard_bytes = rows * width
                rule = (cols * itemsize, width, tp_rank * width)
            size = -(-size // 256) * 256
            slots[(kind, proj)] = (size, shard, dtype)
            for expert in range(experts):
                name = expert_names.get((expert, proj, kind))
                if name in staged:
                    rules[staged.row(name)] = (size + expert * shard_bytes, *rule)
            size += experts * shard_bytes
    flat = torch.zeros(size + _DISCARD_BYTES, dtype=torch.uint8, device=device)
    span = expert_rows(staged, expert_names)
    if span is not None:
        placement = StagedPlacement(staged, span, rules, size, device)
        reader = None
        if runs is None:
            reader = StagedRunReader(staged, staged_runs(staged, *span), device)
            runs = reader
        try:
            for run in runs:
                placement.place(flat, run)
        finally:
            if reader is not None:
                reader.close()
    arranged: dict[str, torch.Tensor] = {}
    for kind, (w13_name, w2_name) in params.items():
        parts = {}
        for proj in ("gate", "up", "down"):
            base, shard, dtype = slots[(kind, proj)]
            nbytes = experts * shard[0] * shard[1] * torch.empty((), dtype=dtype).element_size()
            tensor = flat[base : base + nbytes].view(dtype).view(experts, *shard)
            parts[proj] = tensor.transpose(1, 2) if transposed else tensor
        w13 = torch.cat([parts["gate"], parts["up"]], dim=2 if transposed else 1).contiguous()
        for name, tensor in ((w13_name, w13), (w2_name, parts["down"].contiguous())):
            if tuple(tensor.shape) != tuple(meta[name].shape) or tensor.dtype != meta[name].dtype:
                raise InPlaceUnsupported(
                    f"{name} layout {tuple(tensor.shape)} differs from {tuple(meta[name].shape)}"
                )
            arranged[name] = tensor
    return arranged


def process_like_weights(
    layer: torch.nn.Module, arranged: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Run the layer's post-load processing on ``arranged`` and restore the layer.

    Mirrors vLLM's layerwise reload of one layer, but returns the processed
    tensors instead of copying them into live storage.
    """
    from vllm.model_executor.model_loader.reload.layerwise import (
        _place_kernel_tensors,
        get_layerwise_info,
    )
    from vllm.model_executor.model_loader.reload.meta import (
        SKIP_TENSORS,
        materialize_layer,
        restore_layer_on_meta,
    )
    from vllm.model_executor.model_loader.reload.utils import (
        get_layer_params_buffers,
        get_layer_tensors,
    )

    info = get_layerwise_info(layer)
    info.kernel_tensors = get_layer_params_buffers(layer)
    info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)
    # Processing rebuilds the kernel around the tensors it processes, but those
    # hold the processed delta here. Restore the layer's and method's state so
    # the kernel keeps reading the live storage this delta is XORed into.
    containers = ("_parameters", "_buffers", "_modules")
    layer_state = {name: value for name, value in vars(layer).items() if name not in containers}
    method = layer.quant_method
    method_state = dict(vars(method))
    try:
        restore_layer_on_meta(layer, info)
        materialize_layer(layer, info)
        for name, tensor in get_layer_tensors(layer).items():
            if name in arranged:
                tensor.data.copy_(arranged[name])
            elif name not in SKIP_TENSORS:
                # Live tensors that reload never moves to meta stay untouched.
                tensor.data.zero_()
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")
        method.process_weights_after_loading(layer)
        return {name: getattr(layer, name).data for name in arranged}
    finally:
        _place_kernel_tensors(layer, info)
        for name in [name for name in vars(layer) if name not in containers]:
            if name not in layer_state:
                delattr(layer, name)
        vars(layer).update(layer_state)
        vars(method).clear()
        vars(method).update(method_state)
        info.reset()


@dataclass(frozen=True)
class CheckpointCatalog:
    """A checkpoint's tensor names by decoder layer, built once per layout."""

    layers: dict[str, int | None]
    experts: dict[int, dict[tuple[int, str, str], str]]
    per_expert: frozenset[str]
    # Per-expert tensors whose projection or kind is not recognized.
    unmatched_experts: frozenset[str]
    # Every other tensor, in checkpoint order.
    generic: tuple[str, ...]

    @classmethod
    def of(cls, metadata: dict[str, dict]) -> CheckpointCatalog:
        layers: dict[str, int | None] = {}
        experts: dict[int, dict[tuple[int, str, str], str]] = defaultdict(dict)
        per_expert, unmatched, generic = set(), set(), []
        for name in metadata:
            layers[name] = layer_key(name)
            if not is_per_expert(name):
                generic.append(name)
                continue
            per_expert.add(name)
            match = _EXPERT.search(name)
            if match is None:
                unmatched.add(name)
                continue
            key = (int(match[2]), _PROJECTIONS[match[3]], match[4])
            experts[int(match[1])][key] = name
        return cls(layers, dict(experts), frozenset(per_expert), frozenset(unmatched), tuple(generic))


class InPlaceInstaller:
    """Install staged deltas into one rank's live model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        generic_reload: Callable[[Callable[[Any], None]], None],
        peers: Peers | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._generic_reload = generic_reload
        # Ranks that split reading staged expert bytes.
        self._peers = peers
        self._experts = routed_expert_layers(model)
        # Layers this rank does not serve: other pipeline stages' placeholders,
        # and checkpoint layers the model never builds, such as unused MTP layers.
        self._served_layers = {
            key
            for path, module in model.named_modules()
            if type(module).__name__ != "PPMissingLayer" and (key := layer_key(path)) is not None
        }
        # Processed zeros per expert layout; None when processing maps zero to zero.
        self._zero_terms: dict[tuple, dict[str, torch.Tensor] | None] = {}
        self._catalog: tuple[dict[str, dict], CheckpointCatalog] | None = None
        # Time spent building the reload's inputs in the current install.
        self._input_seconds = 0.0

    def install(self, checkpoint: str | Path, staging: Path) -> dict[str, float]:
        started = time.perf_counter()
        try:
            staged = StagedDelta(staging)
        except ValueError as error:
            raise InPlaceUnsupported(str(error)) from error
        try:
            with _quiet_readonly_buffers():
                metrics = self._install(checkpoint, staged)
        finally:
            staged.close()
        _synchronize(self._device)
        metrics["perf/mx_receive_inplace_time"] = time.perf_counter() - started
        return metrics

    def _install(self, checkpoint: str | Path, staged: StagedDelta) -> dict[str, float]:
        index_started = time.perf_counter()
        _paths, _locations, metadata = index_checkpoint_tensors(checkpoint)
        catalog = self._catalog_for(metadata)
        names = staged.names()
        if any(name not in metadata for name in names):
            raise InPlaceUnsupported("staged tensors are missing from the checkpoint")
        changed = [name for name in names if self._served(catalog.layers[name])]
        unmatched = [name for name in changed if name in catalog.unmatched_experts]
        if unmatched:
            raise InPlaceUnsupported(f"unrecognized expert tensors {unmatched[:3]}")
        expert_layers = sorted(
            {catalog.layers[name] for name in changed if name in catalog.per_expert}
        )
        missing = [index for index in expert_layers if index not in self._experts]
        if missing:
            raise InPlaceUnsupported(f"no routed-expert module for layers {missing[:5]}")
        index_seconds = time.perf_counter() - index_started

        expert_started = time.perf_counter()
        layer_runs = {}
        for index in expert_layers:
            span = expert_rows(staged, catalog.experts.get(index, {}))
            layer_runs[index] = staged_runs(staged, *span) if span else []
        # One reader for every layer, so reads run ahead across layers.
        reader = StagedRunReader(
            staged,
            [run for index in expert_layers for run in layer_runs[index]],
            self._device,
            peers=self._peers,
        )
        try:
            for index in expert_layers:
                self._install_experts(
                    self._experts[index],
                    staged,
                    catalog.experts.get(index, {}),
                    metadata,
                    itertools.islice(reader, len(layer_runs[index])),
                )
        finally:
            reader.close()
        _synchronize(self._device)
        expert_seconds = time.perf_counter() - expert_started

        generic_started = time.perf_counter()
        self._input_seconds = 0.0
        fed = self._install_generic(staged, changed, catalog, metadata)
        _synchronize(self._device)
        return {
            "perf/mx_receive_inplace_index_time": index_seconds,
            "perf/mx_receive_inplace_expert_layers": float(len(expert_layers)),
            "perf/mx_receive_inplace_expert_time": expert_seconds,
            "perf/mx_receive_inplace_expert_wait_time": reader.wait_seconds,
            "perf/mx_receive_inplace_generic_tensors": float(fed),
            "perf/mx_receive_inplace_input_time": self._input_seconds,
            "perf/mx_receive_inplace_generic_time": time.perf_counter() - generic_started,
        }

    def _catalog_for(self, metadata: dict[str, dict]) -> CheckpointCatalog:
        # The checkpoint index returns the same metadata while headers are unchanged.
        if self._catalog is None or self._catalog[0] is not metadata:
            self._catalog = (metadata, CheckpointCatalog.of(metadata))
        return self._catalog[1]

    def _served(self, key: int | None) -> bool:
        return key is None or key in self._served_layers

    def _install_experts(
        self,
        layer: torch.nn.Module,
        staged: StagedDelta,
        expert_names: dict[tuple[int, str, str], str],
        metadata: dict[str, dict],
        runs: Iterator[StagedRun],
    ) -> None:
        arranged = arrange_expert_delta(
            layer, staged, expert_names, metadata, device=self._device, runs=runs
        )
        signature = (
            type(layer.quant_method).__name__,
            tuple((name, tuple(tensor.shape), tensor.dtype) for name, tensor in sorted(arranged.items())),
        )
        if signature not in self._zero_terms:
            zeros = process_like_weights(
                layer, {name: torch.zeros_like(tensor) for name, tensor in arranged.items()}
            )
            nonzero = any(bool(bits(tensor).any()) for tensor in zeros.values())
            self._zero_terms[signature] = zeros if nonzero else None
        processed = process_like_weights(layer, arranged)
        zero = self._zero_terms[signature]
        for name, tensor in processed.items():
            live = getattr(layer, name).data
            xor_into(live, tensor)
            if zero is not None:
                xor_into(live, zero[name])

    def _install_generic(
        self,
        staged: StagedDelta,
        changed: list[str],
        catalog: CheckpointCatalog,
        metadata: dict[str, dict],
    ) -> int:
        """Reload every tensor of each changed non-expert group with XOR restore."""
        groups = {catalog.layers[name] for name in changed if name not in catalog.per_expert}
        if not groups:
            return 0
        fed = [
            name
            for name in catalog.generic
            if catalog.layers[name] in groups and self._served(catalog.layers[name])
        ]
        unknown = sorted({metadata[name]["dtype"] for name in fed} - _TORCH_DTYPES.keys())
        if unknown:
            raise InPlaceUnsupported(f"unsupported checkpoint dtypes {unknown}")
        staged_names = set(changed)
        # Some loaders write live storage directly: tensors reload leaves in place,
        # such as biases, and mirrors a loader keeps, such as fp32 decode copies.
        # They then hold the loaded value rather than an XOR of it.
        originals = {
            name: tensor.detach().clone()
            for name, tensor in self._live_tensors(groups).items()
        }

        device = self._input_device(max(metadata[name]["byte_size"] for name in fed))
        input_seconds = 0.0

        def tensors(zero: bool) -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal input_seconds
            for name in fed:
                info = metadata[name]
                started = time.perf_counter()
                delta = None if zero or name not in staged_names else staged.tensor(name)
                data = staged_bytes(info["byte_size"], delta, device)
                input_seconds += time.perf_counter() - started
                yield name, data.view(_TORCH_DTYPES[info["dtype"]]).view(info["shape"])

        live = self._live_tensors(groups)
        self._generic_reload(lambda _aliases: self._model.load_weights(tensors(zero=True)))
        # XOR the original back into every tensor the zero pass changed, after
        # both passes. A tensor a loader overwrote then ends as original ^ delta;
        # for one the XOR restore changed, the two extra XORs cancel.
        changed_live = {}
        for name, original in originals.items():
            if not same_bits(live[name].data, original):
                xor_into(live[name].data, original)
                changed_live[name] = original
        self._generic_reload(lambda _aliases: self._model.load_weights(tensors(zero=False)))
        for name, original in changed_live.items():
            xor_into(live[name].data, original)
        self._input_seconds = input_seconds
        return len(fed)

    def _input_device(self, nbytes: int) -> torch.device:
        """Build reload inputs on the GPU when it has room for a few of the largest.

        Host inputs cost page faults and a host-to-device copy of every byte.
        Asked once per reload, since querying free memory can be slow.
        """
        if self._device.type != "cuda":
            return torch.device("cpu")
        free, _total = torch.cuda.mem_get_info(self._device)
        cached = torch.cuda.memory_reserved(self._device) - torch.cuda.memory_allocated(
            self._device
        )
        return self._device if free + cached > 2 * nbytes + (1 << 30) else torch.device("cpu")

    def _live_tensors(self, groups: set[int | None]) -> dict[str, torch.Tensor]:
        """Parameters and buffers of the changed groups, except routed experts."""
        experts = {id(module) for module in self._experts.values()}
        found: dict[str, torch.Tensor] = {}
        for path, module in self._model.named_modules():
            if id(module) in experts or layer_key(path) not in groups:
                continue
            if any(part == "experts" for part in path.split(".")):
                continue
            for name, tensor in list(module.named_parameters(recurse=False)) + list(
                module.named_buffers(recurse=False)
            ):
                if tensor.device.type != "meta":
                    found.setdefault(f"{path}.{name}", tensor)
        return found


@contextmanager
def xor_restore() -> Iterator[None]:
    """Make layerwise reload XOR processed tensors into live storage instead of copying."""
    from vllm.model_executor.model_loader.reload import layerwise

    def restore(layer: torch.nn.Module, info: Any) -> None:
        parameters, buffers = info.kernel_tensors
        non_persistent = info.kernel_non_persistent_buffers
        loaded = {name for name, _bound in info.loaded_weights}
        for name, parameter in parameters.items():
            current = getattr(layer, name)
            # Tensors reload never replaced are live storage, not processed output.
            if current is not parameter:
                xor_into(parameter.data, current.data)
        for name, buffer in buffers.items():
            if name not in layer._buffers:
                continue
            if name in non_persistent and name not in loaded:
                continue
            current = getattr(layer, name)
            if current is not buffer:
                xor_into(buffer.data, current.data)
        layerwise._place_kernel_tensors(layer, info)

    original = layerwise._copy_and_restore_kernel_tensors
    layerwise._copy_and_restore_kernel_tensors = restore
    try:
        yield
    finally:
        layerwise._copy_and_restore_kernel_tensors = original


__all__ = [
    "CheckpointCatalog",
    "InPlaceInstaller",
    "InPlaceUnsupported",
    "Peers",
    "StagedPlacement",
    "StagedRun",
    "StagedRunReader",
    "arrange_expert_delta",
    "bits",
    "expert_rows",
    "process_like_weights",
    "raw_bytes",
    "routed_expert_layers",
    "staged_bytes",
    "staged_runs",
    "xor_into",
    "xor_restore",
]
