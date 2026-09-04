# CoreWeave Delta Chain Replay Plan

## Status

- Updated 2026-09-04 after rebasing `feature/coreweave-delta-replay` onto
  `main` at `e8761dd`.
- The initial target-version replay milestone shipped in
  `feat(refit): add target-version delta replay (#710)`.
- Full Hugging Face checkpoint anchors, immutable local artifacts, bounded
  cache management, and interrupted-update recovery are also present on
  `main`.
- Remaining work is CoreWeave fleet integration, external-format coverage, and
  production acceptance rather than the core replay algorithm.

## Outcome

A generator can now request an explicit target revision while serving an older
revision. ModelExpress resolves the complete READY lineage, prepares all missing
revisions in order, and installs only the final reconstructed checkpoint:

```text
serving v0 -> resolve v1, v2, v3 -> materialize v3 -> install v3 once
```

A full checkpoint may reset the lineage:

```text
serving v0 -> full v4 -> delta v5 -> materialize v5 -> install v5 once
```

The implementation does not select a deployment's desired revision. The caller
must provide the target identity, which preserves a clean boundary between the
ModelExpress data plane and the CoreWeave rollout orchestrator.

## Implemented design

### Chain resolution

`ModelExpressGeneratorClient._resolve_replay_chain()` walks backward from the
requested target to either the generator's serving revision or the nearest full
checkpoint. Before acquiring revision leases or preparing payloads, it checks:

- every revision exists, is READY, and belongs to the configured model;
- traversal is cycle-free and does not exceed `max_replay_chain_length`;
- every revision has a usable object-storage source;
- layout signatures remain compatible when present;
- a full checkpoint has no base revision;
- an XOR delta has an exact base revision; and
- every revision can use one compatible preparation method and installer.

The resolved revisions are reversed into base-to-target order. A request for an
already-serving, verified revision takes the warm no-op path.

### Atomic preparation and one final install

`WeightUpdateSession.stage_chain()` holds a lease for every revision in the
resolved chain and delegates the complete chain to one update method.
`CanonicalDeltaUpdateMethod.prepare_chain()` prepares an ordinary installable
checkpoint without mutating the inference engine.

`_LocalCheckpoint.prepare_chain()` then:

1. takes the host-local preparation and installation locks;
2. validates every index manifest before changing checkpoint bytes;
3. downloads or reuses immutable full and delta artifacts;
4. applies each XOR delta against its exact materialized base;
5. verifies reconstructed tensor checksums after each apply; and
6. returns the final checkpoint to the engine installer.

Only `WeightUpdateSession.apply()` installs weights. Intermediate revisions are
never exposed to the engine, and `active.json` advances only after the final
installation succeeds.

### Local checkpoint lifecycle

The cache separates canonical inputs from derived checkpoints:

```text
<refit_checkpoint_dir>/<model>/
  full/<version>/            immutable full checkpoints
  deltas/<version>/          immutable delta payloads
  chains/<version>.json      resolved full anchor plus ordered deltas
  materialized/<version>/    derived installable checkpoints
  state.json                 preparation state and file identity
  active.json                last successfully installed revision
```

The first delta after a full checkpoint copies the immutable full artifact.
Sequential deltas may rename and update the derived materialization in place,
avoiding another full-model copy. Cache capacity enforcement protects the
active lineage and current update while preferring eviction of rebuildable
materializations.

An interrupted or failed preparation leaves `state.json` in `UPDATING` unless
the failure is a pre-mutation validation or capacity error. Initialization does
not trust that state; it restores a verified launch seed before accepting a new
update. A failed engine installation leaves the prepared disk checkpoint READY
but keeps `active.json` on the prior revision. The generator marks the engine
state uncertain until a subsequent successful installation restores a known
revision.

### Full checkpoint anchors

`FULL_HF_CHECKPOINT` is an explicit payload format and a legal replay anchor.
Resolution stops at the first full checkpoint encountered while walking toward
the active revision. The receiver downloads and verifies its complete tensor
set, records it under `full/`, and uses it as the base for later deltas. This
allows bounded recovery without replaying all the way back to the launch seed.

## Implementation map

| Responsibility | Current location |
|---|---|
| Target lookup and backward chain resolution | `modelexpress_rl/inference/client.py` |
| Revision leases and chain staging | `modelexpress_rl/inference/session.py` |
| Method selection and chain capability | `modelexpress_rl/inference/plan.py` |
| Canonical S3 preparation adapter | `modelexpress_rl/inference/methods/canonical_delta.py` |
| Manifest validation, download, apply, and recovery | `modelexpress_rl/inference/receiver.py` |
| Immutable artifacts, chain records, locks, and capacity | `modelexpress_rl/inference/checkpoint_store.py` |
| vLLM checkpoint installation | `modelexpress_rl/inference/engines/vllm/` |
| User-facing configuration and operations | `docs/S3_DELTA_WEIGHT_REFIT.md` |

All code paths above are rooted at `modelexpress_client/python/`.

## Required invariants

These remain release-blocking invariants for any follow-up change:

1. Never apply a delta unless the local version exactly matches its declared
   base.
2. Resolve the entire control-plane chain and validate all index manifests
   before mutating checkpoint bytes.
3. Accept only READY revisions for the requested model.
4. Detect cycles, missing parents, invalid payload transitions, and excessive
   depth before payload preparation.
5. Preserve a stable layout signature across the chain when signatures are
   provided.
6. Verify each reconstructed tensor before advancing to the next delta.
7. Never install an intermediate revision.
8. Advance the active revision only after engine installation succeeds.
9. Never reuse an interrupted or externally modified materialization as a
   verified checkpoint.
10. Repeating a request for the active verified target is a no-op.

## Remaining CoreWeave work

### 1. Cold-start and scale-out orchestration (P0)

CoreWeave must persist or otherwise resolve the deployment's desired target and
pass that exact identity to each replacement or newly scaled worker. A new
worker must remain quarantined until preparation, installation, and target
identity verification succeed.

Required end-to-end flow:

```text
start and quarantine worker
  -> resolve deployment target
  -> initialize verified local seed/cache
  -> request explicit target
  -> replay from serving seed or nearest full anchor
  -> install target
  -> report per-replica target READY
  -> admit rollout traffic
```

This work owns desired-version persistence, retry policy, per-replica status,
and admission. ModelExpress continues to own exact-target reconstruction and
rank-local installation.

### 2. External checkpoint-format acceptance (P0)

The receiver consumes the canonical safetensors layout, per-tensor zstd frames,
and post-apply checksums. Production acceptance still needs fixtures produced
outside ModelExpress and compatibility coverage for the finalized CoreWeave
writer.

The current reader requires XOR encoding with `adler32`. The supplied format
document also names `xxh3-128` and `blake3`, with `xxh3-128` described as the
newer slime default. Either the CoreWeave writer must pin `adler32` for launch,
or ModelExpress must add and test the agreed checksum algorithms.

Required tests:

- consume externally generated full and delta checkpoints;
- verify shard numbering, tensor maps, zstd frame boundaries, and metadata;
- reject missing, duplicated, unsafe, or malformed shard references;
- reject wrong versions, bases, layouts, encodings, and checksums; and
- prove bit-for-bit equality with a direct full load at the target.

### 3. Fleet-level validation (P0)

Run the production topology with tensor parallelism and multiple replicas:

- a replacement worker reaches the deployment target before serving;
- a scale-out worker reaches the same target before serving;
- multiple ranks sharing a host perform one local reconstruction and one
  install per rank;
- a middle-delta download, decode, checksum, or disk-capacity failure never
  changes the active revision;
- restart from `UPDATING` restores a known seed and replays successfully;
- a failed engine install keeps the replica quarantined until a successful
  recovery install; and
- final parameters and generations match a direct full checkpoint load.

### 4. Engine and serving integration (P1)

The replay implementation produces a normal checkpoint directory and is
engine-independent up to installation, but the current documented production
path is vLLM. Validate SGLang installation separately if CoreWeave requires it.

Fleet readiness should ultimately expose the installed revision per replica.
Served revision identity in inference responses, HTTP 425 transition behavior,
prompt-cache reset policy, and hot-load cancellation/history remain separate
requirements and are not implied by chain replay.

## Delivery plan

1. **External fixtures:** agree on the launch checksum algorithm and pass
   CoreWeave-produced full/delta fixtures through the existing receiver tests.
2. **Fleet contract:** define desired-target lookup, trigger input, per-replica
   status, quarantine, timeout, and retry behavior with the orchestrator.
3. **Cold-start integration:** wire explicit-target replay into replacement and
   scale-out worker startup.
4. **Failure drills:** exercise corrupt objects, missing lineage, full-anchor
   fallback, interrupted preparation, failed installation, and cache pressure.
5. **Performance validation:** measure chain-resolution RPCs, S3 download,
   reconstruction, disk working set, and final installation at target scale.
6. **Launch acceptance:** demonstrate target identity, parameter equality, and
   generation parity across every replica before traffic admission.

## Acceptance criteria

The CoreWeave cold-start replay work is complete when:

- every new or replacement replica is given the deployment's explicit target;
- the replica remains out of service until that exact target is installed;
- any valid bounded chain from a verified seed or full anchor reconstructs the
  target with all relationships and tensor results verified;
- only the target revision is installed and reported active;
- interrupted preparation and failed installation recover deterministically
  without serving an unknown or stale revision;
- externally produced launch-format checkpoints pass conformance tests; and
- all ranks and replicas match a direct full load in weights and generation.

## Non-goals of replay

- Selecting the deployment's desired revision inside ModelExpress.
- Generator-to-generator RDMA fanout policy.
- HTTP 425 transition handling.
- Prompt-cache reset policy.
- Served revision identity in inference responses.
- Experience-payload transfer.
- Quantizing training-precision checkpoints during hot-load.
- Replaying an unbounded chain without a configured safety limit.
