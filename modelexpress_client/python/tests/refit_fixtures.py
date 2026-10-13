# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delta snapshots in a fake S3, shared by the host checkpoint cache tests."""

import json

import safetensors.numpy
import torch
from modelexpress_rl import WeightPayloadFormat
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.utils import checksum_factory, compress_delta, compute_delta

PREFIX = "s3://weights/hotload"
INDEX = "model.safetensors.index.json"
SHARD = "model-00000-of-00001.safetensors"


def _bytes(values):
    return torch.tensor(values).contiguous().reshape(-1).view(torch.uint8).numpy()


def _checksum(value):
    checksum = checksum_factory("adler32")
    checksum.update(value)
    return checksum.hexdigest()


def _uri(identity):
    return f"{PREFIX}/{identity}/{INDEX}"


def _delta_snapshot(base, target, *, identity, base_identity, version=None):
    """The objects of a one-tensor XOR delta from ``base`` to ``target``."""
    delta, _ = compute_delta(_bytes(target), _bytes(base))
    shard = safetensors.numpy.save(
        {"weight": compress_delta(delta)},
        metadata={"weight": _checksum(_bytes(target))},
    )
    index = json.dumps(
        {
            "metadata": {
                "version": version or identity,
                "base_version": base_identity,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {"weight": SHARD},
        }
    ).encode()
    return {_uri(identity): index, f"{PREFIX}/{identity}/{SHARD}": shard}


def _delta(version_id, *, base, identity):
    return receiver_module._S3Version(
        version_id=version_id,
        base_version_id=base,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        uri=_uri(identity),
    )


class _FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.gets = []

    def get(self, uri):
        self.gets.append(uri)
        return self.objects[uri]

    def size(self, uri):
        return len(self.objects[uri])

    def exists(self, uri):
        return uri in self.objects

    def list_children(self, prefix_uri, *, start_after=None):
        prefix = prefix_uri.rstrip("/") + "/"
        names = {
            key[len(prefix) :].split("/", 1)[0]
            for key in self.objects
            if key.startswith(prefix) and "/" in key[len(prefix) :]
        }
        return sorted(name for name in names if start_after is None or name > start_after)

    def close(self):
        pass
