---
name: create-remote-node
description: 'Create or resume a remote Brix GPU devbox node using twdev/rcall-brix. Use when: create node, create remote node, spin up GPU node, new devbox, resume paused node, start a node, launch remote, create moe node, create verl node.'
argument-hint: 'Optional: prefix (default verl), N (default 1), i (auto-pick unused), num_gpu (default 8), priority (default team-critical)'
---

# Create Remote Node

Create or resume a remote Brix GPU devbox. By default creates `verl-n1-i{i}` nodes, auto-picking an unused `i` index.

## When to Use
- User wants to create a new remote GPU node / devbox
- User says "create node", "spin up a node", "new devbox", "start a node"
- User wants to resume a paused or suspended node
- User needs a GPU node for training, eval, or development

## Defaults

| Parameter | Default |
|-----------|---------|
| prefix | `verl` |
| N (num_pods) | `1` |
| i (index) | auto-pick unused |
| cluster | `prod-westus2-cw-6` |
| num_gpu | `8` |
| priority_class | `team-critical` |
| team | `team-moonfire-speech` |

## Procedure

### Step 1 — Parse arguments

Extract from user input (or use defaults):
- `PREFIX` — job name prefix (default `verl`)
- `N` — number of pods (default `1`)
- `I` — instance index (user-specified, or auto-pick in Step 2)
- `CLUSTER` — cluster name (default `prod-westus2-cw-6`)
- `NUM_GPU` — GPUs per node (default `8`)
- `PRIORITY` — priority class (default `team-critical`)
- `TEAM` — team name (default `team-moonfire-speech`)

The full job name is `{PREFIX}-n{N}-i{I}`.

### Step 2 — Check for existing pool

List pools matching the prefix to find existing nodes and determine their status:

```bash
rcall-brix ls '{PREFIX}-n*' 2>&1
```

**If user provided a specific `i`:**
- Check if `{PREFIX}-n{N}-i{I}` exists in the output.

**If user did NOT provide `i` (auto-pick):**
- Parse existing pool names matching `{PREFIX}-n{N}-i*`.
- Extract used `i` values.
- Pick the smallest unused `i` starting from 0.
- Example: if `i0` and `i1` exist, pick `i2`.

### Step 3 — Resume if paused/suspended

If the pool `{PREFIX}-n{N}-i{I}` already exists:

**Status is `Ready`:**
- Report that the node is already running. Done.

**Status is `Paused`, `Suspended`, or any non-Ready state:**
- Resume the pool:
  ```bash
  rcall-brix resume {PREFIX}-n{N}-i{I}
  ```
- Poll until it becomes Ready:
  ```bash
  rcall-brix ls '{PREFIX}-n{N}-i{I}' 2>&1
  ```
- If status is `Ready`, report success. Done.
- If status is `Assigning`, `Scheduled`, or other non-Ready state, wait 15 seconds and poll again. Keep polling until `Ready`.
- Report each poll result so the user sees progress (e.g. "Status: Assigning... waiting").

### Step 4 — Create new node

If the pool does NOT exist, create it with `twdev`:

```bash
twdev create-ray-devbox \
  cluster={CLUSTER} \
  num_pods={N} \
  num_gpu={NUM_GPU} \
  job_name={PREFIX}-n{N}-i{I} \
  priority_class={PRIORITY} \
  team={TEAM}
```

Wait for the command to complete. This may take 1-2 minutes.

### Step 5 — Verify

After create or resume, confirm the node is up:

```bash
rcall-brix ls '{PREFIX}-n{N}-i{I}' 2>&1
```

Optionally verify pod is Running:
```bash
kubectl --context {CLUSTER} -n boren get pods --no-headers | grep '{PREFIX}-n{N}-i{I}'
```

Report the final status to the user including:
- Job name: `{PREFIX}-n{N}-i{I}`
- Cluster: `{CLUSTER}`
- Status: Ready / Scheduled / etc.
- Size: `{N} x {NUM_GPU} GPU`

## Response Style
- Be concise. Report the job name, cluster, and status.
- If resuming, mention it was resumed (not newly created).
- If auto-picking `i`, mention which index was chosen and why.
- Surface errors immediately.
