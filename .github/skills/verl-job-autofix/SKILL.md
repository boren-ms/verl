---
name: verl-job-autofix
description: Monitor and safely recover verl ASR Ray jobs on all Brix nodes. Use for fleet status, stalled-job recovery, Ray failure diagnosis, and latest-code resubmission.
---

# verl Job Monitor and Auto-Fix

Monitor all `verl-*` Brix nodes on `prod-westus2-cw-6`, report active and latest terminal submissions, and recover only confirmed failures or stalls.

## Guardrails

- Preserve job settings and resume behavior. Never add `enforce_eager`.
- Do not manually start Ray. For a replacement pod, wait until it is 2/2 Ready and `/tmp/ray/session_latest` plus `ray status` are available.
- Treat exactly one GPU at 100% utilization with every other GPU at 0% as a stalled rollout. Automatically rerun that job with its identical configuration.
- Use `ray job status` as the authoritative submission state. Do not infer status from GPU activity alone.
- Never show Ray submission IDs in user-facing status tables.

## Discover and Monitor

```bash
kubectl --context prod-westus2-cw-6 -n boren get pods -o \
  'custom-columns=NAME:.metadata.name,PHASE:.status.phase,READY:.status.containerStatuses[*].ready' \
  --no-headers | grep '^verl' | sort

kubectl --context prod-westus2-cw-6 -n boren exec verl-<node>-0 -- bash -l -c \
  'python /root/code/verl/ray_job.py list 2>/dev/null'
```

For each active submission collect:

```bash
ray job status <submission-id>
timeout 30 ray job logs <submission-id> 2>&1 | grep -oP 'step:\K[0-9]+' | tail -1
timeout 30 ray job logs <submission-id> 2>&1 | grep -oP 'critic/score/mean:\K[^ ]+' | tail -1
timeout 30 ray job logs <submission-id> 2>&1 | grep -oP 'val-aux/[^ ]*(?:dter_p_err|p_err|wer)/mean@1:\K[^ ]+' | tail -2
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
```

Report a table keyed by node and `trainer.experiment_name`: Ray status, progress, score/mean, p_err, phase, GPU utilization, and delta since the prior poll.

## Failed Jobs

For `FAILED`, obtain the traceback before acting:

```bash
ray job logs <submission-id> 2>&1 | tail -n 60
```

Keep the pipeline moving: diagnose the failure, apply a deterministic safe fix when one exists, sync current code, and resubmit only the same configuration. Confirm the replacement is RUNNING, then continue monitoring it through training or evaluation completion.

Examples:

- FSDP batch divisibility assertion: correct the invalid requested config value to a valid divisor, sync it, and restart that job.
- Brix Git resolver failure: use the direct workspace-sync fallback, then launch the same config.
- Pod replacement or Ray unavailable: wait for a 2/2 Ready replacement with platform Ray, then sync and resubmit the interrupted config.
- Missing module or deterministic code error: patch locally, sync, and resubmit the same job.

Do not change intentional training settings, switch configs, manually start Ray, or use `enforce_eager`. If the traceback shows an ambiguous data, checkpoint, or infrastructure failure, preserve the job's resume behavior and retry the same config once the underlying service is available.

## Single-GPU Stall Recovery

If a RUNNING job has exactly one GPU at 100% utilization while every other GPU is 0%:

1. Stop the submission with `ray job stop <submission-id>`.
2. Resubmit its identical configuration with the latest workspace.
3. Confirm the replacement Ray status is RUNNING.

## Submit and Fallback

Preferred:

```bash
bash submit_job.sh <node> recipe/phimm/config/<config>.yaml false true true
```

If Brix Git resolution fails (`no matching gits`), use the direct-sync fallback. For multi-pod nodes, sync both pods:

```bash
tar -C /home/boren/code/verl --exclude=.git --exclude=.venv --exclude=logs --exclude=tmp -cf - . \
  | kubectl --context prod-westus2-cw-6 -n boren exec -i verl-<node>-0 \
      -- tar -C /root/code/verl -xf -
```

Then submit:

```bash
kubectl --context prod-westus2-cw-6 -n boren exec verl-<node>-0 -- bash -l -c \
  'cd /root/code/verl && python ray_job.py cleanup <config-name> && \
   bash quick_run.sh recipe/phimm/config/<family>/<config-name>.yaml'
```

## Node Replacement

If `kubectl exec` reports no host assigned, a missing pod, or a platform replacement:

1. Wait for the replacement pod to be `Running true true`.
2. Wait for `/tmp/ray/session_latest` and successful `ray status`.
3. Sync current code and resubmit only the interrupted configuration.

If the pool is suspended, resume it with `brix resume verl-<node>`. To pause it on request, use `brix pause verl-<node>`.
