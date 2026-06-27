# Design: LoRA Adapter-Only Param Sync over the Disaggregated NCCL Path

**Status:** Draft / pending review
**Author:** (assisted)
**Target repo:** `verl-mirror`
**Date:** 2026-06-27
**Context job:** `raysubmit_DsBydRftLJSygEiW` (`remax_r2_punc_p0_7_s100_r2t6_sc4_icepop`, verl-n1-i11)

---

## 1. Problem statement

In the fully-async (disaggregated) policy, the trainer pool (e.g. 6 GPUs) and the
rollout pool (e.g. 2 GPUs) live in **separate processes** and exchange updated
weights every `async_training.trigger_parameter_sync_step` via the **NCCL
checkpoint engine**.

Even though training uses **LoRA** (`lora_rank=320`, `lora_alpha=640`), each param
sync broadcasts the **full ~9B base model** (~18 GB bf16), costing **~58 s per
sync**. Observed on the icepop run:

| Metric | Value |
|--------|-------|
| `timing_s/param_sync` | ~58 s |
| `timing_s/update_actor` | ~45 s |
| `timing_s/step` | ~103–106 s |
| `rollouter/idle_ratio` | ~0.83 (active 18.9 s / version 112.9 s) |
| `trainer/idle_ratio` | ~0.002 (trainer ~100% busy) |

The 58 s sync dominates the trainer step and is the main reason the 2-GPU rollout
pool sits idle (it fills its staleness-bounded queue fast, then waits for the slow
trainer step + full-model broadcast).

### Goal

Send **only the LoRA adapter** (A/B matrices, a few hundred MB) on each sync:
- **Base weights** are broadcast **once** (first sync after a fresh vLLM weight load).
- **Every subsequent sync** broadcasts only the adapter; the rollout applies it via
  vLLM `add_lora(TensorLoRARequest(...))`, replacing the previous adapter.

Expected effect: `param_sync` drops from ~58 s to a few seconds; rollout idle
ratio falls; overall throughput rises with **no change to learned behavior**.

---

## 2. Current behavior (as-is)

### 2.1 Two sync managers in the fully-async trainer

`verl/experimental/fully_async_policy/fully_async_trainer.py`:

| Manager | Backend | Used for | Code |
|---------|---------|----------|------|
| `checkpoint_manager` | **nccl** (from `rollout.checkpoint_engine.backend`) | **per-step** weight sync to the separate rollout pool | `_fit_update_weights()` → line ~537 |
| `hybrid_checkpoint_manager` | **naive** (forced) | trainer-side **validation** (hybrid replicas, colocated) | `_setup_hybrid_checkpoint_manager()` → line ~592 |

The **per-step** sync (the slow one) is the **nccl** path. The **naive** path is
only used during validation and is colocated (in-process IPC), so it cannot be
reused for the disaggregated 6t/2r layout.

### 2.2 The nccl branch discards LoRA info

`verl/workers/engine_workers.py::update_weights` (~lines 700-709):

```python
effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

# async training with disaggregated trainer/rollout
if effective_mode != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()   # <-- no args; peft_config DISCARDED
    await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
    return
```

- `get_per_tensor_param()` is called with **defaults** (`layered_summon=False`,
  `base_sync_done=False`).
- The returned `peft_config` is thrown away (`_`).

### 2.3 What `get_per_tensor_param(base_sync_done=False)` returns

`verl/workers/engine/fsdp/transformer_impl.py::get_per_tensor_param` (~818):

```python
peft_config = None
merge_lora = self.model_config.lora.get("merge", False)   # False for our run (lora dict empty)
peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
if hasattr(peft_model, "peft_config"):       # LoRA model
    if not merge_lora:
        peft_config = peft_model.peft_config.get("default", None)
        params = collect_lora_params(module=self.module,
                                     layered_summon=layered_summon,
                                     base_sync_done=base_sync_done)
        if not base_sync_done:
            params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
    else:  # merge lora
        with merged_lora_context(self.module, backup_adapters=True):
            params = normalize_peft_param_name(self.module.state_dict())
else:
    params = self.module.state_dict()
```

`verl/utils/fsdp_utils.py::collect_lora_params(base_sync_done=False)` returns the
**full base model** weights (skipping only `lora_`/`_flat_param` keys):

```python
if base_sync_done:
    lora_params = get_peft_model_state_dict(peft_model)         # SMALL: just adapter A/B
else:
    model = peft_model.base_model.model                         # FULL base model
    for name, param in model.state_dict().items():
        if any(x in name for x in ["_flat_param", "lora_"]): continue
        lora_params[name] = param.full_tensor()...              # ~9B base weights
```

So the nccl path sends the **full base** every time, with `base_sync_done` stuck
at the default `False`.

### 2.4 The naive path already does it right

`engine_workers.py::update_weights`, naive branch (~lines 711-754):

```python
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=True)        # adapter-only

do_lora_base_sync = False
if not self.peft_merge and peft_config is not None:
    self.rollout.sleep_level = 1
    do_lora_base_sync = not self.base_sync_done                     # base only on first sync

if do_lora_base_sync:
    base_params, peft_config = get_per_tensor_param(layered_summon=..., base_sync_done=False)
    await self.rollout.update_weights(base_params, peft_config=peft_config,
                                      base_sync_done=False, global_steps=...)

await self.rollout.update_weights(per_tensor_param, peft_config=peft_config,
                                  base_sync_done=True, global_steps=...)
self.base_sync_done = True
```

This is exactly the behavior we want — it is just **not wired into the nccl
(disaggregated) path**, because the nccl path doesn't go through
`self.rollout.update_weights(...)` (IPC) but through
`self.checkpoint_engine.send_weights(...)` (NCCL broadcast), which currently has
no way to carry `peft_config` / `base_sync_done`.

### 2.5 The receiver can already apply adapters

Receive funnel for the nccl path:

```
NCCLCheckpointEngine.receive_weights()                          (rollout CE process)
  -> CheckpointEngineWorker.update_weights()                    base.py:322-325
       weights = checkpoint_engine.receive_weights(...)
       await server_adapter.update_weights(weights, global_steps=...)   # <-- no peft_config today
  -> vLLMRollout.update_weights(weights, **kwargs)              vllm_rollout.py:170
       -> update_weights_from_ipc(**kwargs)                     utils.py:229
            if peft_config and base_sync_done:
                _update_weights -> add_lora(TensorLoRARequest(peft_config, lora_tensors=weights))
            else:
                model.load_weights(param_updates)               # standard full-weight load
```

- vLLM is confirmed launched with `--enable_lora` (so `lora_as_adapter=True`).
- `update_weights_from_ipc(peft_config, base_sync_done)` already implements both
  the adapter path (`add_lora`) and the base path (`load_weights`).
- The **only** missing link on the receive side is that
  `CheckpointEngineWorker.update_weights` calls `server_adapter.update_weights(...)`
  **without** `peft_config` / `base_sync_done`.

---

## 3. Proposed change (to-be)

Carry `peft_config` + `base_sync_done` through the NCCL transport so the
disaggregated path can replicate the naive path's base-once-then-adapter logic.

### 3.1 Touch point 1 — trainer send (`engine_workers.py`, nccl branch)

Replace the bare send with base-once-then-adapter, mirroring the naive branch:

```python
if effective_mode != "naive":
    per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon, base_sync_done=True)

    do_lora_base_sync = (not self.peft_merge) and (peft_config is not None) and (not self.base_sync_done)

    if do_lora_base_sync:
        base_params, _ = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=False)
        await self.checkpoint_engine.send_weights(
            base_params, global_steps=global_steps,
            peft_config=None, base_sync_done=False)

    await self.checkpoint_engine.send_weights(
        per_tensor_param, global_steps=global_steps,
        peft_config=peft_config, base_sync_done=(peft_config is not None))

    self.base_sync_done = True
    return
```

Notes:
- `self.base_sync_done` and `self.peft_merge` already exist (used by the naive branch).
- `self.layered_summon` already exists.
- For a **non-LoRA** checkpoint, `peft_config is None` → behaves exactly as today
  (single full-model send, `base_sync_done` argument ignored downstream).

### 3.2 Touch point 2 — NCCL transport metadata

`verl/checkpoint_engine/nccl_checkpoint_engine.py`:

- `send_weights(self, weights, global_steps=None, peft_config=None, base_sync_done=False)`:
  attach `peft_config` and `base_sync_done` to the **final** broadcast bucket's
  metadata dict (which already travels over the zmq PUB/SUB channel):

  ```python
  broadcast_op = BroadcastOperation(
      ...,
      metadata={"bucket_meta": bucket_meta, "is_last": True,
                "peft_config": peft_config, "base_sync_done": base_sync_done},
      ...)
  ```

- `receive_weights(...)`: surface the trailing `peft_config` / `base_sync_done`
  to the caller. Options:
  - **(B1)** stash on `self` (`self._last_peft_config`, `self._last_base_sync_done`)
    right before the generator finishes, so `CheckpointEngineWorker.update_weights`
    can read them after consuming the generator. Simplest; safe because one
    receive call == one update.
  - **(B2)** change the generator contract to also yield a final sentinel — more
    invasive, avoid.

  **Recommendation: B1.**

### 3.3 Touch point 3 — receiver plumbing (`base.py::CheckpointEngineWorker.update_weights`)

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
async def update_weights(self, global_steps: int = None):
    weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
    await self.server_adapter.update_weights(
        weights, global_steps=global_steps,
        peft_config=getattr(self.checkpoint_engine, "_last_peft_config", None),
        base_sync_done=getattr(self.checkpoint_engine, "_last_base_sync_done", False))
```

Because `receive_weights` is an async generator consumed inside
`server_adapter.update_weights`, the `_last_*` attributes must be **set while the
generator is being drained** (i.e. when the `is_last` bucket arrives), so they are
available by the time `update_weights_from_ipc` decides adapter-vs-base. Verify
the ordering: `server_adapter.update_weights` drains the generator first
(`BucketedWeightSender.async_send_weights(weights)`) and only then calls
`update_weights_from_ipc`. If `update_weights_from_ipc` is launched **before**
draining (it is started non-blocking in `vllm_rollout.update_weights`), we must
instead pass `peft_config`/`base_sync_done` **into** `server_adapter.update_weights`
as explicit args (as written above) rather than reading them post-hoc. **This is
the single most important ordering detail to get right.**

> Alternative that sidesteps the ordering issue entirely: transmit
> `peft_config`/`base_sync_done` on the **first** bucket instead of the last, so
> they are known before any weight is applied. Trade-off: must send peft_config
> even on base-only sync (cheap; it's a small dict, and on base sync we just send
> `None`). **Recommendation: send on the first bucket.**

### 3.4 Touch point 4 — signature compatibility

`verl/checkpoint_engine/base.py::CheckpointEngine.send_weights` (abstract) and the
sibling backends (`naive`, `nixl`, `mooncake`, `kimi`, `hccl`):

- Add `peft_config: dict | None = None, base_sync_done: bool = False, **kwargs`
  to the base abstract signature.
- Non-nccl backends **accept and ignore** (they are not used in this layout), so
  the blast radius stays minimal and they don't break import/abstract checks.

---

## 4. Open risks (resolve before / during implementation)

### R1 — Does the rollout keep base weights resident between syncs? (correctness-critical)
Adapter-only sync is only valid if the vLLM base weights persist across syncs.
- For `lora_as_adapter=True`, `_sleep_hybrid()` uses **sleep level 1** (adapter
  released, **weights kept**), so base should persist.
- **Must verify** for the **disaggregated 2-GPU rollout pool** (not just the
  hybrid/colocated validation pool): confirm that between two consecutive syncs
  the pool does not do a level-2 sleep or a full weight reload. If it does,
  `self.base_sync_done` must be reset to `False` on every such event (re-send base),
  which shrinks the savings.
- **How to check:** inspect the fully-async rollouter's sleep/wake calls around
  `update_weights`, and grep the live job for sleep-level / "resume weights" logs.

### R2 — Param name/key alignment
`collect_lora_params(base_sync_done=True)` uses `get_peft_model_state_dict` keys;
`TensorLoRARequest` expects PEFT-style names. The naive path already works with
these keys, and NCCL only moves named tensors verbatim, so keys should be
preserved. **Add an assert / debug log** of the first few adapter keys on both
sides during validation.

### R3 — `merge_lora=True` interaction
If `rollout.lora.merge=True` (NOT our case — `lora` dict is empty so merge=False),
`get_per_tensor_param` returns full merged weights and `peft_config=None`; the new
code path then behaves exactly like today (full-model send). No regression.

### R4 — base_sync_done lifecycle on resume / replica scale events
`base_sync_done` must be `False` after: (a) fresh start, (b) any rollout replica
restart, (c) elastic add/remove of replicas, (d) checkpoint resume that
re-instantiates the rollout. Audit each path that creates/replaces a rollout
replica and ensure the flag is reset (per-replica if necessary).

### R5 — Partial-rollout / staleness interaction
The nccl `CheckpointEngineManager.update_weights` aborts + resumes unfinished
requests around the sync. Confirm `add_lora` mid-stream (between abort and resume)
correctly affects only **new** requests and doesn't corrupt in-flight decode state.

---

## 5. Implementation plan (ordered)

1. **Verify R1** from the live job / rollouter code (no edits). Gate the whole
   change on base weights persisting between syncs.
2. **Touch point 4** (signatures) — additive, no behavior change.
3. **Touch point 2** (nccl metadata, first-bucket transmission) + **touch point 3**
   (receiver plumbing). Land together so the receive side can read the flags.
4. **Touch point 1** (trainer send base-once-then-adapter).
5. Reset logic for **R4** (`base_sync_done` lifecycle).
6. Guard everything behind a config flag, e.g.
   `async_training.lora_adapter_sync: true` (default **false**), so the full-model
   path remains the safe default and the change is opt-in / easily reverted.

---

## 6. Validation plan (non-negotiable for a weight-sync change)

1. **Speed:** short run (~20 steps) on `remax_r2_punc_p0_7_s100_r2t6_sc4_icepop`
   with the flag on; confirm `timing_s/param_sync` drops from ~58 s to a few
   seconds and `rollouter/idle_ratio` falls.
2. **Equivalence:** confirm `critic/score/mean`, `pg_loss`, `response_length`
   trajectories track an un-patched reference run for the same steps (proves the
   rollout is using the up-to-date adapter, not stale/base weights).
3. **End-to-end:** export the merged HF checkpoint and run the in-house
   `long_rollout_inhouse_2605_all_seg` eval — DTER must be in the normal **10–30%**
   range (a broken sync would show ~100% / garbage).
4. **A/B:** compare DTER of flag-on vs flag-off checkpoints at the same step; they
   should match within noise.

---

## 7. Decision log (to fill during review)

| ID | Decision | Choice | Rationale |
|----|----------|--------|-----------|
| A | `base_sync_done` reset triggers | TBD (depends on R1/R4) | correctness |
| B | Metadata transmission point | **first bucket** (recommended) | avoid ordering race in §3.3 |
| C | Sibling backend signatures | accept-and-ignore new kwargs | minimal blast radius |
| D | Opt-in config flag | `async_training.lora_adapter_sync` (default false) | safe rollout / easy revert |

---

## 8. Files to change (summary)

| File | Change |
|------|--------|
| `verl/workers/engine_workers.py` | nccl branch: base-once-then-adapter; thread `peft_config`/`base_sync_done`; set `self.base_sync_done` |
| `verl/checkpoint_engine/nccl_checkpoint_engine.py` | `send_weights`/`receive_weights`: carry `peft_config` + `base_sync_done` in zmq metadata (first bucket) |
| `verl/checkpoint_engine/base.py` | `CheckpointEngine.send_weights` abstract sig + `CheckpointEngineWorker.update_weights` receiver plumbing |
| `verl/checkpoint_engine/{naive,nixl,mooncake,kimi,hccl}_*.py` | accept-and-ignore new kwargs |
| `recipe/phimm/config/base/remax_asr.yaml` (or per-config) | add opt-in flag `async_training.lora_adapter_sync` |

---

## 9. Rollback

The change is opt-in (Decision D). Setting `async_training.lora_adapter_sync: false`
(default) restores the current full-model nccl sync with zero behavioral change.
