---
name: rcall-brix-jobs
description: Use when the user wants to inspect Brix jobs or pools with their local `bls` workflow, where `bls` may resolve to `b ls` and `b` may resolve to `rcall-brix`. This skill lists current jobs with `rcall-brix ls`, identifies jobs in `Scheduled` state, restarts only those scheduled jobs with `rcall-brix restart`, and can change pool team allocation with `rcall-brix change-team`.
---

# Rcall Brix Jobs

Use this skill for routine Brix pool management from the terminal. Prefer the user's local alias flow when it exists, but fall back to the resolved underlying command when aliases are not loaded in the Codex shell.

## Workflow
1. Resolve the command chain before acting.
2. List jobs with `rcall-brix ls`.
3. Read the table and find rows whose `STATUS` is exactly `Scheduled`.
4. Restart only those scheduled jobs with `rcall-brix restart <NAME>`.
5. List jobs again to confirm the updated state.
6. If the user asks to move jobs to another team, use `rcall-brix change-team --to <TEAM> <PATTERN...>` and verify the `QUEUE` column afterward.

## Command Resolution
- If the user asks to use `bls`, first check whether `bls` exists with `type bls`.
- If `bls` is an alias, inspect the alias target with `alias bls`.
- If that expands to `b ls`, inspect `b` with `type b` or `alias b`.
- If `b` resolves to `rcall-brix`, use `rcall-brix` directly in Codex commands because interactive aliases may not be loaded in the non-interactive shell.
- If `rcall-brix` is not available, stop and report the missing command rather than guessing.

## Listing Jobs
Run:

```bash
rcall-brix ls
```

Guidance:
- This command may work in the sandbox, but if it blocks, hangs, or returns obvious network or DNS failures, rerun with escalation.
- Summarize the important rows instead of dumping raw terminal control codes.
- Call out `Ready`, `Scheduled`, `Assigning`, and `Unallocated` counts or notable jobs when useful.
- Note that `Unhealthy` rows can later transition into `Scheduled` after other actions; if the user asked to restart all currently scheduled jobs, verify again after the first pass.

## Restarting Scheduled Jobs
Only restart jobs whose `STATUS` is exactly `Scheduled`.

Run one command per job:

```bash
rcall-brix restart <NAME>
```

Guidance:
- Do not restart `Assigning`, `Ready`, or `Unallocated` jobs unless the user explicitly asks.
- If more than one scheduled job exists, restart them in parallel when the tool setup allows it.
- After each restart, report whether Brix acknowledged the restart and whether it used a runtime retry strategy such as `RetryRuntime`.
- If one restart fails with a cluster informer sync timeout or similar transient cluster lookup error, retry that pool individually before concluding it is stuck.

## Verification
After restarting scheduled jobs, run:

```bash
rcall-brix ls
```

Confirm:
- previously `Scheduled` jobs are no longer `Scheduled`
- whether they became `Ready`, `Assigning`, or another state
- whether any scheduled jobs remain
- whether any newly `Scheduled` jobs appeared during the restart window and need a follow-up restart

## Changing Team Allocation
Use this when the user asks to move pools from one team to another.

Inspect help first if needed:

```bash
rcall-brix change-team --help
```

Run:

```bash
rcall-brix change-team --to <TEAM> <PATTERN...>
```

Guidance:
- `change-team` changes the team allocation and recreates pods.
- In practice this often needs unrestricted network access from Codex. If you see `blobfile` DNS or storage-map lookup failures, rerun with escalation instead of waiting indefinitely.
- Use a quoted glob pattern when targeting a family of jobs, for example `rcall-brix change-team --to team-moonfire-speech 'bus-ats-prod-westus2-cw-6-*'`.
- If a broad pattern only partially applies, rerun `change-team` against the remaining explicit pool names.
- Verify team allocation using the `QUEUE` column from `rcall-brix ls`. The `PRIORITY` column is separate and may remain unchanged.
- After a team change, jobs may temporarily show `Scheduled`, `Assigning`, `Ready`, or even `Unhealthy` while pods are recreated. Treat that as a transitional state unless the user asks for additional remediation.

## Response Style
- Keep responses short and operational.
- When the user asks to "list the jobs", provide the current key states and highlight scheduled jobs explicitly.
- When the user asks to restart scheduled jobs, name exactly which jobs will be restarted before issuing the commands.
- When the user asks to change teams, say that Brix will recreate the pods and that verification will use the `QUEUE` column.
- If there are no scheduled jobs, say that clearly and do not restart anything.
