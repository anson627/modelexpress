# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare object-storage refits in the background, outside the serving pause.

One prefetcher process runs per host checkpoint cache, next to the inference
engine. It watches the ModelExpress registry for READY versions and, optionally,
the S3 snapshot prefix for deltas that are not registered yet. It prepares the
next version toward the newest registered one, or the next unregistered delta,
into the host-local cache through the same locked preparation the engine's
update uses, so a later update finds its target prepared and only installs it.
Once a staged head the engine serves reaches the layer limit, it writes the
head's deltas into its files.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import posixpath
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import grpc
from modelexpress import auth

from .. import refit_pb2, refit_pb2_grpc
from ..control import WeightVersion, WeightVersionState, _weight_version
from ..object_storage import ObjectStorageType
from ..s3 import S3Client
from ..train import WeightPayloadFormat
from .checkpoint_store import (
    PREFETCH_TRIGGER_FILE,
    CheckpointRecord,
    CheckpointState,
    LocalCheckpointStore,
)
from .receiver import (
    ObjectStorageGeneratorConfig,
    PreparedCheckpoint,
    _active_version,
    _LocalCheckpoint,
    _S3Version,
    speculative_version_id,
)
from .version_chain import resolve_replay_chain

logger = logging.getLogger("modelexpress_rl.inference.prefetch")

STATUS_FILE = "prefetch.json"
# How often the prefetcher checks for new versions.
POLL_SECONDS = 2.0
LOCK_FILE = ".prefetch.lock"

# How often a rewrite checks for an engine waiting on a following delta.
_WANTED_CHECK_SECONDS = 0.1

_REPLAY_FORMATS = frozenset(
    {WeightPayloadFormat.XOR_DELTA, WeightPayloadFormat.FULL_HF_CHECKPOINT}
)


@dataclass(frozen=True)
class PrefetchSettings:
    """Prefetcher configuration, passed to its process as JSON."""

    model_name: str
    cache_model_name: str
    server_url: str
    refit_checkpoint_dir: str
    initial_base_version_id: str
    seed_checkpoint_path: str | None
    refit_checkpoint_max_size_gb: int | None
    endpoint_url: str | None
    region_name: str | None
    max_replay_chain_length: int
    rpc_timeout_seconds: float
    s3_watch: bool
    parent_pid: int
    # Leave verified deltas staged on their base's files, rewritten later.
    staged: bool = False
    # The engine streams full checkpoints as it installs them.
    streamed: bool = False

    def object_storage(self) -> ObjectStorageGeneratorConfig:
        return ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id=self.initial_base_version_id,
            seed_checkpoint_path=self.seed_checkpoint_path,
            refit_checkpoint_dir=self.refit_checkpoint_dir,
            refit_checkpoint_max_size_gb=self.refit_checkpoint_max_size_gb,
        )

    def store(self) -> LocalCheckpointStore:
        return LocalCheckpointStore(
            root=self.refit_checkpoint_dir,
            model_name=self.cache_model_name,
        )


def launch_prefetcher(settings: PrefetchSettings, *, workers: int) -> subprocess.Popen:
    """Start the prefetcher process for one host checkpoint cache."""
    environment = dict(os.environ, MX_REFIT_DELTA_WORKERS=str(workers))
    return subprocess.Popen(
        [sys.executable, "-m", __name__, "--settings", json.dumps(asdict(settings))],
        env=environment,
    )


class RefitPrefetcher:
    """Prepare the next version toward the newest registered one in the host checkpoint cache."""

    def __init__(
        self,
        settings: PrefetchSettings,
        *,
        service: Any,
        checkpoint: _LocalCheckpoint,
    ) -> None:
        self._settings = settings
        self._service = service
        self._s3 = checkpoint.s3
        self._checkpoint = checkpoint
        self._store = checkpoint.store
        self._channel: grpc.Channel | None = None
        # Servers that predate version listing still support the S3 watch.
        self._registry_listing = True
        self._writer_versions: dict[str, str] = {}
        # The STAGED head and files a rewrite last tried, which stay unwritten
        # below the layer limit, or when it found no room or failed.
        self._rewrite_tried: tuple[str, str | None] | None = None
        # Versions whose preparation failed, with the head it started from. The
        # engine prepares those itself when it needs them.
        self._failed: set[tuple[str, str]] = set()

    @classmethod
    def connect(cls, settings: PrefetchSettings) -> RefitPrefetcher:
        channel = auth.with_auth(grpc.insecure_channel(settings.server_url))
        s3 = S3Client(
            endpoint_url=settings.endpoint_url,
            region_name=settings.region_name,
        )
        checkpoint = _LocalCheckpoint(
            model_name=settings.cache_model_name,
            config=settings.object_storage(),
            s3=s3,
            stage_deltas=settings.staged,
        )
        prefetcher = cls(
            settings,
            service=refit_pb2_grpc.RefitServiceStub(channel),
            checkpoint=checkpoint,
        )
        prefetcher._channel = channel
        return prefetcher

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._s3.close()

    def poll_once(self) -> str | None:
        """Prepare at most one version; return the cache ID it prepared."""
        state = self._store.state()
        if state is None or state.status is CheckpointState.UPDATING:
            return None
        if not self._may_move_head(state.version):
            return None
        head = self._registered_head(state.version)
        version = self._next_registered(head)
        if version is not None:
            return self._prepare(
                version.version_id,
                head=head,
                kind="registered",
                run=lambda: self._checkpoint.prepare_chain(
                    (version,), staged=self._settings.staged, announce=False
                ),
            )
        if self._settings.s3_watch:
            prepared = self._speculate(state)
            if prepared is not None:
                return prepared
        if (
            state.status is CheckpointState.STAGED
            and _active_version(self._store) == state.version
            and self._rewrite_tried != (state.version, state.base)
        ):
            # The engine serves the staged head, so at the layer limit write its
            # deltas into the files now. A delta that appears meanwhile joins.
            self._rewrite_tried = (state.version, state.base)
            return self._prepare(
                state.version,
                head=state.version,
                kind="rewrite",
                run=lambda: self._checkpoint.materialize_head(
                    staged=self._settings.staged,
                    following=self._following(state),
                ),
            )
        return None

    def _following(
        self, head: CheckpointRecord
    ) -> Callable[[], tuple[_S3Version, bool] | None]:
        """Return a check for a delta on top of ``head``, for a rewrite to join.

        An engine waiting to prepare such a delta is seen within
        ``_WANTED_CHECK_SECONDS``; registry and S3 checks run at most once per
        poll interval.
        """
        checked = {"wanted": 0.0, "remote": time.monotonic()}

        def following() -> tuple[_S3Version, bool] | None:
            now = time.monotonic()
            if now - checked["wanted"] >= _WANTED_CHECK_SECONDS:
                checked["wanted"] = now
                found = self._wanted_delta(head.version)
                if found is not None:
                    return found, False
            if now - checked["remote"] < POLL_SECONDS:
                return None
            checked["remote"] = now
            try:
                version = self._next_registered(head.version)
                if version is not None:
                    return version, False
                if self._settings.s3_watch:
                    found = self._next_speculative(head)
                    if found is not None:
                        return found, True
            except Exception as error:
                # A failed check must not abort the rewrite it runs inside.
                logger.warning("prefetch check during a rewrite failed: %s", error)
            return None

        return following

    def _wanted_delta(self, head: str) -> _S3Version | None:
        """Return the registered delta on top of ``head`` that an engine waits for."""
        wanted = self._store.wanted()
        if (
            wanted is None
            or not wanted.get("staged")
            or wanted.get("chain_length") != 1
            or wanted.get("base_version") != head
            or not isinstance(wanted.get("version"), str)
            or not isinstance(wanted.get("uri"), str)
            or (wanted["version"], head) in self._failed
        ):
            return None
        return _S3Version(
            version_id=wanted["version"],
            base_version_id=head,
            payload_format=WeightPayloadFormat.XOR_DELTA,
            uri=wanted["uri"],
        )

    def run(self) -> None:
        """Poll until the engine process that started this prefetcher exits."""
        trigger = self._trigger_mtime()
        while self._parent_alive():
            try:
                self.poll_once()
            except Exception:
                logger.exception("prefetch poll failed")
            deadline = time.monotonic() + POLL_SECONDS
            while time.monotonic() < deadline and self._parent_alive():
                current = self._trigger_mtime()
                if current != trigger:
                    trigger = current
                    break
                time.sleep(0.2)

    def _parent_alive(self) -> bool:
        return os.getppid() == self._settings.parent_pid

    def _trigger_mtime(self) -> int | None:
        try:
            return (self._store.cache / PREFETCH_TRIGGER_FILE).stat().st_mtime_ns
        except OSError:
            return None

    def _may_move_head(self, head: str) -> bool:
        """Whether preparing now cannot disturb an installation of the head.

        Engine ranks release the cache between preparing a target and installing
        it, and preparation may mutate a derived head in place. The head may move
        once the engine has activated it, or when it is a speculation on top of
        the active version, which the engine never installs under its own ID.
        """
        active = _active_version(self._store)
        if active == head:
            return True
        speculation = self._store.speculation()
        return (
            active is not None
            and speculation is not None
            and speculation.get("version") == head
            and speculation.get("base_version") == active
        )

    def _registered_head(self, head: str) -> str:
        """Return the head, or the registered base of a speculative head."""
        speculation = self._store.speculation()
        if speculation is not None and speculation.get("version") == head:
            return speculation.get("base_version", head)
        return head

    def _next_registered(self, head: str) -> _S3Version | None:
        """Return the registered version after ``head`` toward the newest one.

        Only the next version is prepared, so the cache head never passes a
        version the engine may still be asked to install.
        """
        target = self._newest_registered() if self._registry_listing else None
        if target is None or target.version_id == head:
            return None
        try:
            chain = resolve_replay_chain(
                target_version_id=target.version_id,
                fetch_ready_version=self._fetch_ready_version,
                max_chain_length=self._settings.max_replay_chain_length,
                stop_before_version_id=head,
            )
        except RuntimeError as error:
            logger.info("prefetch cannot resolve %s: %s", target.version_id, error)
            return None
        if any(
            version.object_storage.storage_type is not ObjectStorageType.S3
            for version in chain
        ):
            return None
        version = chain[0]
        if (version.version_id, head) in self._failed:
            return None
        if (
            self._settings.streamed
            and version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
        ):
            # Preparing it here would hold the cache the stream's writer waits on.
            return None
        return _S3Version(
            version_id=version.version_id,
            base_version_id=version.base_version_id,
            payload_format=version.payload_format,
            uri=version.object_storage.uri,
        )

    def _newest_registered(self) -> WeightVersion | None:
        try:
            response = self._service.ListWeightVersions(
                refit_pb2.ListWeightVersionsRequest(model_name=self._settings.model_name),
                timeout=self._settings.rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            if error.code() is grpc.StatusCode.UNIMPLEMENTED:
                self._registry_listing = False
                logger.info("MX server cannot list versions; prefetch watches S3 only")
            else:
                logger.warning("prefetch could not list versions: %s", error)
            return None
        candidates = []
        for message in response.versions:
            try:
                version = _weight_version(message)
            except RuntimeError:
                continue
            if (
                version.state is WeightVersionState.READY
                and version.payload_format in _REPLAY_FORMATS
                and version.object_storage is not None
                and version.object_storage.storage_type is ObjectStorageType.S3
            ):
                candidates.append(version)
        return max(candidates, key=lambda version: version.created_at_unix_ms, default=None)

    def _fetch_ready_version(self, version_id: str) -> WeightVersion:
        try:
            response = self._service.GetWeightVersion(
                refit_pb2.GetWeightVersionRequest(uid=version_id),
                timeout=self._settings.rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            raise RuntimeError(f"failed to resolve revision {version_id!r}") from error
        if not response.HasField("version"):
            raise RuntimeError(f"MX returned no revision {version_id!r}")
        version = _weight_version(response.version)
        if (
            version.state is not WeightVersionState.READY
            or version.model_name != self._settings.model_name
        ):
            raise RuntimeError(f"revision {version_id!r} is not a READY model version")
        return version

    def _speculate(self, head: CheckpointRecord) -> str | None:
        """Prepare the unregistered delta whose snapshot names the head as base."""
        version = self._next_speculative(head)
        if version is None:
            return None
        return self._prepare(
            version.version_id,
            head=head.version,
            kind="speculative",
            run=lambda: self._checkpoint.prepare_speculative(
                version, staged=self._settings.staged
            ),
            uri=version.uri,
        )

    def _next_speculative(self, head: CheckpointRecord) -> _S3Version | None:
        """Return the unregistered delta whose snapshot names the head as base.

        None when its preparation from the head already failed.
        """
        speculation = self._store.speculation()
        if speculation is not None and speculation.get("version") == head.version:
            return None
        head_uri = (head.source or {}).get("uri") or self._registered_uri(head.version)
        if head_uri is None:
            return None
        # Snapshots live at <prefix>/<identity>/<index>, and identities sort in
        # publish order. A delta names its base by the writer's version string.
        snapshot = posixpath.dirname(head_uri)
        identity = posixpath.basename(snapshot)
        prefix = posixpath.dirname(snapshot)
        index_name = posixpath.basename(head_uri)
        if not identity or not prefix.startswith("s3://"):
            return None
        head_version = self._writer_version(head_uri, default=identity)
        for child in self._s3.list_children(prefix, start_after=identity):
            if child <= identity or child.startswith(("_", ".")):
                continue
            uri = f"{prefix}/{child}/{index_name}"
            if not self._is_complete_child_delta(uri, base_version=head_version):
                continue
            version_id = speculative_version_id(uri)
            if (version_id, head.version) in self._failed:
                return None
            return _S3Version(
                version_id=version_id,
                base_version_id=head.version,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                uri=uri,
            )
        return None

    def _registered_uri(self, version_id: str) -> str | None:
        try:
            storage = self._fetch_ready_version(version_id).object_storage
        except RuntimeError:
            return None
        return None if storage is None else storage.uri

    def _writer_version(self, uri: str, *, default: str) -> str:
        """Return the version the snapshot's writer recorded, or ``default``."""
        if uri not in self._writer_versions:
            try:
                index = json.loads(self._s3.get(uri))
                version = index["metadata"]["version"]
            except Exception:
                version = None
            self._writer_versions[uri] = version if isinstance(version, str) else default
        return self._writer_versions[uri]

    def _is_complete_child_delta(self, uri: str, *, base_version: str) -> bool:
        try:
            index = json.loads(self._s3.get(uri))
        except Exception:
            return False
        metadata = index.get("metadata") if isinstance(index, dict) else None
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(weight_map, dict)
            or metadata.get("delta_encoding") != "xor"
            or metadata.get("base_version") != base_version
        ):
            return False
        # Writers upload the index last, but every shard must exist before reading.
        snapshot = posixpath.dirname(uri)
        return all(
            self._s3.exists(f"{snapshot}/{shard}") for shard in set(weight_map.values())
        )

    def _prepare(
        self,
        version_id: str,
        *,
        head: str,
        kind: str,
        run: Callable[[], PreparedCheckpoint | None],
        uri: str | None = None,
    ) -> str | None:
        started = time.monotonic()
        self._write_status(phase="preparing", version=version_id, kind=kind, uri=uri)
        logger.info("prefetch preparing %s version %s", kind, version_id)
        try:
            prepared = run()
        except Exception as error:
            logger.warning(
                "prefetch of %s failed; the engine will prepare it: %s", version_id, error
            )
            self._failed.add((version_id, head))
            self._write_status(
                phase="failed", version=version_id, kind=kind, uri=uri, error=str(error)
            )
            try:
                self._checkpoint.recover_incomplete_preparation()
            except Exception:
                logger.exception("prefetch recovery after a failed preparation failed")
            return None
        duration = time.monotonic() - started
        if prepared is None:
            self._write_status(phase="idle", version=version_id, kind=kind, uri=uri)
            return None
        logger.info(
            "prefetch prepared %s version %s in %.2fs", kind, prepared.target_version, duration
        )
        state = self._store.state()
        if uri is None and state is not None and state.version == prepared.target_version:
            # Only speculation passes a URI, so name the prepared head's source,
            # which for a rewrite can be the delta it joined.
            uri = (state.source or {}).get("uri")
        self._write_status(
            phase="prepared",
            version=prepared.target_version,
            kind=kind,
            uri=uri,
            duration_seconds=duration,
            metrics=prepared.metrics,
        )
        return prepared.target_version

    def _write_status(self, **fields: Any) -> None:
        status = dict(fields, pid=os.getpid(), updated_at_unix_ms=int(time.time() * 1000))
        path = self._store.cache / STATUS_FILE
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(status, sort_keys=True))
            temporary.replace(path)
        except OSError:
            logger.debug("prefetch status is unavailable", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", required=True, help="JSON PrefetchSettings")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = PrefetchSettings(**json.loads(args.settings))
    cache_dir = settings.store().cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / LOCK_FILE).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("another prefetcher already serves %s", cache_dir)
            return 0
        with contextlib.suppress(OSError):
            os.nice(10)
        prefetcher = RefitPrefetcher.connect(settings)
        try:
            prefetcher.run()
        finally:
            prefetcher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
