# CoreWeave Delta Chain Replay Plan

## Status

- Planning document for `feature/coreweave-delta-replay`.
- Branch base: `hwoo/mx-delta-v4.1-vllm` at `2062cf9`.
- Scope: reconstruct an explicitly requested target revision from a verified local
  checkpoint by applying every missing delta in order, then install the final
  checkpoint into the inference engine once.

## Why replay is required

An XOR delta is valid only against its declared base. A worker at `v0` cannot
apply `delta(v2 -> v3)` directly. A replacement, restarted, or newly scaled
worker must either obtain a complete current checkpoint or reconstruct it:

```text
full v0 -> delta v1 -> delta v2 -> delta v3 -> install v3
```

The revision catalog already records `target_version`, `base_version`,
`target_digest`, `base_digest`, and `format_digest`. The receiver already
performs exact-base reconstruction for one delta. Replay composes those two
primitives without weakening their validation.

## Required invariants

1. Never apply a delta to a checkpoint whose version and digest do not exactly
   match the delta's declared base.
2. Resolve and validate the entire chain before mutating checkpoint bytes.
3. Every revision in the chain must belong to the requested model and be
   `READY` or `COMMITTED`.
4. Adjacent revisions must agree on version, base digest, and format digest.
5. Detect cycles, missing parents, self-parenting revisions, and excessive chain
   depth before apply.
6. Verify every reconstructed revision before advancing to the next delta.
7. Do not expose or install an intermediate revision into the inference engine.
8. Install the final reconstructed checkpoint once and report the final revision
   as active only after engine installation succeeds.
9. A failed replay must never be mistaken for a valid checkpoint. The local
   journal remains poisoned until the checkpoint is reseeded or recovered.
10. Repeating a request for an already verified target is a no-op.

## Contract boundary

The development branch currently uses custom delta-index and bucket objects with
SHA-256 digests. The finalized CoreWeave contract uses safetensors shards,
per-tensor zstd frames, and checksums stored in safetensors metadata.

Chain resolution must therefore operate only on revision relationships and
digests; it must not depend on the current bucket encoding. Payload application
stays behind the receiver's single-revision apply boundary. Before production
acceptance, that boundary must consume the finalized CoreWeave checkpoint
format. Replay should not make the development format a new public contract.

## Current building blocks

| Building block | Existing location |
|---|---|
| Exact revision lookup | `modelexpress/refit/catalog.py` |
| Parent and digest metadata | `modelexpress/refit/manifest.py` |
| Exact-base checkpoint validation and XOR apply | `modelexpress/refit/receiver.py` |
| Persistent checkpoint journal and poison marker | `modelexpress/refit/receiver.py` |
| vLLM final installation | `modelexpress/engines/vllm/refit/` |
| SGLang final installation | `modelexpress/engines/sglang/refit/` |
| Receiver status/result types | `modelexpress/refit/api.py` |

## Proposed design

### 1. Pure chain resolver

Add a small engine- and payload-independent resolver, preferably
`modelexpress/refit/replay.py`:

```python
@dataclass(frozen=True)
class ResolvedRevisionChain:
    model_id: str
    base_version: str
    target_version: str
    revisions: tuple[RevisionRecord, ...]  # forward order, base excluded


def resolve_revision_chain(
    catalog: RevisionCatalog,
    *,
    model_id: str,
    current_version: str,
    current_digest: str,
    target_version: str,
    format_digest: str,
    max_depth: int,
) -> ResolvedRevisionChain:
    ...
```

Resolution walks backward from the requested target using exact
`get_revision()` calls until it reaches `current_version`, then reverses the
records into apply order. It validates:

- model identity;
- allowed revision state;
- non-empty and distinct target/base versions;
- cycle-free parent traversal;
- the configured maximum depth;
- one stable format digest;
- target/base digest continuity between every adjacent pair; and
- agreement between the oldest delta and the caller's current digest.

The first implementation takes an explicit target version. Discovering the
deployment's latest active revision belongs to the CoreWeave orchestrator and
is not required to implement safe replay.

### 2. Receiver preparation API

Add `prepare_to_version(target_version)` to `ModelExpressWeightReceiver` and a
chain-oriented method to `_LocalCheckpoint`.

Expected lifecycle:

```text
resolve full chain
    -> lock local checkpoint
    -> revalidate journal version/digest
    -> apply and verify each revision in forward order
    -> return one PreparedRevision for the final target
    -> install final checkpoint once
    -> mark final revision VERIFIED
```

Do not implement replay by repeatedly calling the public
`start_weight_update()` and `update_weights()` pair: that would reload the full
model into the GPU after every intermediate delta. Intermediate progress is a
disk reconstruction concern; only the requested target is an engine update.

The checkpoint lock should cover the complete replay so another local rank
cannot observe or independently mutate an intermediate version. Followers that
acquire the lock afterward should see the final journal state and take the warm
no-op path.

### 3. Journal and recovery behavior

Before applying the first delta, persist a replay journal containing at least:

```json
{
  "poisoned": true,
  "replay_base_version": "v0",
  "replay_target_version": "v3",
  "last_verified_version": "v0"
}
```

After each delta verifies, update `last_verified_version` and its digest. Clear
the poisoned marker only after the final revision is verified and the normal
checkpoint state has been written atomically.

For the initial implementation, any interrupted or failed replay reseeds the
working checkpoint from the configured launch checkpoint and starts replay
again. Resuming from an in-place intermediate revision is unsafe unless the
journal and checkpoint bytes can be proven consistent. Copy-on-write checkpoint
publication may be evaluated separately because duplicating multi-terabyte
checkpoints has a substantial capacity and latency cost.

### 4. Initialization and cold start

Change initialization so it distinguishes:

- a verified reusable local checkpoint;
- a poisoned or externally modified checkpoint that must be reseeded;
- a launch checkpoint that is the configured replay anchor; and
- the explicit target revision requested by the orchestrator.

The current code reseeds whenever the journal version differs from
`initial_version`. That prevents reuse of a valid later revision and must be
changed. A later local checkpoint may be reused only after its version, digest,
format digest, and file state are validated against the catalog.

Cold-start orchestration is then:

```text
start/quarantine worker
    -> determine explicit deployment target
    -> initialize or reseed verified local anchor
    -> prepare_to_version(target)
    -> install target
    -> report VERIFIED
    -> admit rollout traffic
```

### 5. Full checkpoint anchors

Delta-only replay can begin using the configured launch checkpoint as the
anchor. Complete CoreWeave recovery also requires full checkpoints that reset
the chain.

The catalog needs an explicit payload kind, rather than inferring full versus
delta from missing fields. A follow-up protocol change should distinguish:

- metadata-only launch anchor;
- full Hugging Face checkpoint; and
- XOR delta checkpoint.

Once supported, chain resolution should stop at the nearest usable full
checkpoint, acquire and verify it, and replay only later deltas. Selection must
be deterministic and bounded; it must not silently fall back to an older model
after the orchestrator requested a newer target.

## Failure behavior

| Failure | Required result |
|---|---|
| Target revision missing | Fail before local mutation |
| Parent revision missing | Fail before local mutation and identify the missing version |
| Cycle or self-parent | Fail before local mutation |
| Chain exceeds limit | Fail before local mutation |
| Model or format mismatch | Fail before local mutation |
| Adjacent digest mismatch | Fail before local mutation |
| S3 download failure | Keep checkpoint poisoned; permit reseed and retry |
| Delta decompression/checksum failure | Keep checkpoint poisoned; never install |
| Engine install fails before mutation | Report `FAILED`; reconstructed disk target may remain reusable |
| Engine install may have mutated weights | Report `POISONED`; worker must remain quarantined/replaced |

Errors should include the model, requested target, failing revision, and replay
position without exposing credentials or signed object-store URLs.

## Test plan

### Chain resolver tests

- Direct parent update (`v0 -> v1`).
- Multi-hop replay (`v0 -> v1 -> v2 -> v3`).
- Already-at-target no-op.
- Missing target and missing intermediate parent.
- Self-parent and multi-node cycle.
- Maximum-depth rejection.
- Wrong model ID.
- Non-ready revision.
- Format-digest mismatch.
- Base-version and base-digest discontinuity.

### Checkpoint replay tests

- Apply three deltas and compare the final safetensors bytes with a full target
  checkpoint.
- Verify one engine install for a multi-delta replay.
- Confirm another rank sharing the cache performs no additional downloads.
- Fail the middle delta and prove no engine installation occurs.
- Restart after a poisoned middle delta, reseed, and successfully replay.
- Reuse a verified later local checkpoint without reseeding.
- Reject externally modified checkpoint files.
- Retry an already completed target without applying XOR twice.

### Engine integration tests

- vLLM lifecycle order remains `start -> receive/install -> finish`.
- Receiver becomes `VERIFIED` only after vLLM finalization succeeds.
- vLLM finalization failure leaves the receiver `POISONED`.
- TP greater than one uses one host-local replay and one install per rank.
- Parameter equality and generation parity against a full load of the target.
- Quantized and CUDA-graph-captured model coverage.

### CoreWeave-format tests

- Consume externally produced canonical safetensors delta fixtures.
- Enforce version/base metadata and checksum algorithm selection.
- Reject malformed shard numbering, missing tensors/checksums, wrong zstd
  frames, and out-of-order application.
- Start from a full checkpoint and replay subsequent deltas.

## Delivery sequence

1. **Contract alignment:** isolate payload decoding and add CoreWeave-format
   fixtures before extending replay around the legacy bucket encoding.
2. **Chain resolution:** implement and unit-test backward traversal and all
   pre-mutation validation.
3. **Disk replay:** apply a resolved chain under one lock, journal progress, and
   produce one final `PreparedRevision`.
4. **Engine integration:** expose `prepare_to_version()` through SGLang and
   vLLM while preserving one final engine install.
5. **Restart recovery:** reuse verified later checkpoints and automatically
   reseed poisoned checkpoints before replay.
6. **Full anchors:** add explicit full-checkpoint catalog records and nearest
   viable baseline selection.
7. **Fleet integration:** let the CoreWeave orchestrator provide the active
   target, quarantine new workers, and consume per-replica replay status.

## Acceptance criteria

Replay is complete for the initial delta-only milestone when:

- a worker starting at a verified configured anchor reaches an explicit target
  through any valid bounded delta chain;
- every chain relationship and reconstructed revision is verified;
- no intermediate version is installed into the inference engine;
- a failed or interrupted replay cannot serve or masquerade as the target;
- retry and restart behavior is deterministic; and
- final model bytes and generation match a direct full load of the target.

Full CoreWeave cold-start inheritance additionally requires full-checkpoint
anchors, active-target discovery by the deployment orchestrator, and externally
visible per-replica readiness.

## Explicit non-goals for the initial replay change

- Selecting the deployment's latest active revision.
- Generator-to-generator RDMA fanout.
- HTTP 425 transition handling.
- Prompt-cache reset policy.
- Served revision identity in inference responses.
- Experience-payload transfer.
- Replaying an unbounded chain without a configured safety limit.
