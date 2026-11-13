# Monitoring and scoped recovery

Track only jobs explicitly identified by the user or launched for this request;
never add other discovered jobs. Save exact experiment/config, cluster,
namespace, pool, submission ID, lineage, resource restrictions, recovery
boundary, desired final optimizer step `T`, verified resume step, last completed
optimizer step, remaining steps, and requested end condition. Rediscover current
Ray submissions on each pass and update identity after a scoped replacement.
A status-only or read-only request never authorizes recovery or other mutations.

For authorized execution, use one monitor in the current chat, defaulting to
30 minutes if no cadence was specified. Keep checking a specific job through
completion in this chat; do not create a scheduled task solely to watch that
job. If the user explicitly requests a recurring scheduled workflow and runtime
instructions permit it, search for `automation_update` first, inspect existing
schedules and ownership, and use the native tool with its current schema.
Never edit automation files or claim a schedule exists without a tool result.
For a pipeline use one monitor for all phases, not one per job. Its durable
state identifies this skill, the state path where applicable, tracked identities,
queue policy, recovery/read-only
boundary, target `T`, validation reporting cursor, exact resource pause targets
and pause status/exceptions, comparison run identities/reference, report output
root, last verified report/evidence fingerprint, report delivery cursor,
end and stop conditions. Update state when phase or submission changes.
For an explicitly requested schedule, confirm the saved cadence, client time
zone, chat destination and active state from the native tool result.
Keep unchanged passes quiet unless the user asks for an update every pass. On a
report, include each tracked job and its phase, live identity/pods, progress and
delta (including `no progress change`), latest per-dataset validation and
checkpoint evidence, GPU/process health, changed risks and required action.
Highlight meaningful progress, new evidence, failures or completion; do not
infer completion from absence or a stall from one quiet interval. Stop the
heartbeat only after all requested outcomes, including the training target and
completion pausing under SKILL.md and delivery of reports for all newly ready
validation (when applicable),
are verified and the final status is reported, or on explicit user cancellation.
Resolve each resource as verified paused or
an evidenced keep-running/shared-use exception; an unverified or failed pause
remains pending and keeps the monitor active.
A recovery blocker stops mutations requiring user direction; retain read-only
tracking on the same cadence and report the blocker once or when it changes.

## Every 30-minute pass

Perform the full pass even when no notification is warranted. Reconcile every
tracked lineage against live Ray, new logs, participating pods/processes/GPUs,
completed updates and checkpoint writes/uploads. Recheck outstanding blockers,
apply authorized minimal recovery below when evidence warrants it, and verify
any replacement submission before saving its identity. Advance only requested
pipeline phases, preserving unfinished training in the tracked set.
For completed work, apply SKILL.md's completion pause checks, reconcile pending
pause operations against the Brix control plane, and report a verified pause
or changed pause blocker. Never resume completed resources merely to poll them.

After the live health/recovery checks, check for the validation trigger below.
Invoke reporting only for newly ready validation or a pending report retry.
Reporting uses collected evidence and must not delay an urgent
recovery action or keep GPU resources running after durable collection is done.

Check for newly ready validation on every pass. Read final aggregate metrics
for each completed dataset at its actual validation step; do not treat an
in-progress batch or partial metric line as completed validation. Report newly
completed dataset results on the next pass, labeling any other requested
datasets still pending. Include separate `val-core/*/p_err/mean@1` and
`val-aux/*/p_err/mean@1` values where available, dataset coverage, current
training step/target and delta, checkpoint completeness, resource health and
recovery changes. Distinguish step-zero validation from optimizer progress.
If several validations completed between passes, retain each new step's
per-dataset results rather than silently replacing them with the latest one.

Persist observed and reported validation separately, keyed by lineage,
submission, validation step, dataset and metric namespace, with source/log
cursor or artifact fingerprint. Mark results reported only after the report
is delivered. On restart or recovery, reconcile these records so old or replayed
metrics are not presented as new; changed results at the same step must be
identified as reruns or corrections. Keep a compact durable record of every
pass even when unchanged. Stay quiet if no new validation, meaningful progress,
failure/recovery, changed blocker or verified completion is observed, unless
the user explicitly requested a report every pass.

## Invoke `report-experiment` when new validation is ready

Invoke [report-experiment](../../report-experiment/SKILL.md) only when any tracked
run has new or corrected completed validation not yet included in a delivered
report, unless the user opts out. One newly completed dataset at any actual
validation step is sufficient; do not wait for all datasets or an exact shared
step across runs. Batch results discovered in one pass into one report refresh,
preserving every new step and marking other datasets pending. Pass
the exact tracked and reference run identities, effective configs/revisions,
resume lineage, full metric histories, completed validation steps, collection
time, existing report root, and note/homepage identities when known. Preserve
the chosen reference, experiment order and colors across refreshes. Do not
discover extra experiments merely to fill the comparison; if fewer than two
runs are identified, record the missing comparison input, ask for it once,
and continue health checks and authorized recovery.

Deduplicate validation using lineage, submission, step, dataset, metric namespace
and artifact fingerprint; distinguish corrections from replayed observations.
If no new validation or pending retry exists, skip invoking `report-experiment`
and retain the last verified package. Training progress, new reward-only history,
config changes or completion alone do not trigger a report; include their latest
evidence with the next validation-triggered refresh. Retry pending rendering,
delivery or vault handoffs even when the source evidence is unchanged.
Use one stable comparison root with timestamped
refresh directories so a failed refresh cannot overwrite the last good report.
Label active-run reports interim and record their exact observed step coverage.
Never manufacture aligned steps or treat a report refresh as training progress.

Apply the report skill's PDF verification, inline validation/config presentation
and Obsidian handoff for each newly delivered package, respecting user opt-outs.
Update the same research note and daily entry rather than creating a new note
per pass. Advance the report delivery cursor only after verified artifacts are
delivered; preserve pending PDF, delivery or vault failures for retry on the next
pass. Keep experiment recovery in this skill; reporting remains read-only toward
experiments and never launches evaluations or changes training to obtain data.

Persist each report attempt, triggering validation identities/fingerprints,
validation coverage included in the last delivered report, observed metric coverage,
last verified artifact paths, delivery status and any blocker separately from
job health. A reporting failure must not stop monitoring or authorized recovery.
Before closing the monitor, reconcile final completed validation against the
last delivered report and refresh only for uncovered or corrected validation,
or retry outstanding report work, without resuming
completed GPU resources. An explicit report opt-out or user-directed stop is
recorded as such rather than claimed as a successful refresh.

## Automatic fixes within scope

When recovery is within the user's requested scope, diagnose and minimally
recover only tracked work without a separate approval. Respect explicit stop,
read-only and narrower boundaries. Establish failure from live Ray state and
the complete current traceback, or a stall from repeated polls plus current
process/GPU evidence. Identify code, data, environment, capacity or recipe
causes. Make and verify only the smallest code/runtime fix preserving the
effective recipe, data, reward, priority, pool, topology, checkpoint and resume
semantics; changes to these need authorization. Sync the exact revision, verify
no competing live submission, and resubmit with a new ID while recording the
original failure lineage and checkpoint/resume boundary. Do not call recovered
progress a continuation unless checkpoint loading proves it.

For unfinished training, treat a terminal job below `T` as an early exit, even
if Ray reports `SUCCEEDED`; inspect the complete exit logs and effective limits
before deciding on recovery. Keep the original target and lineage. Resume only
under the existing checkpoint/resume policy after verifying loadability and
restoration of the model, optimizer and other required trainer/dataloader state.
Do not silently restart from step zero or choose a different checkpoint policy
when compatible resume state is unavailable. Report that boundary for direction.
If an authorized recovery resumes step `S`, train toward the original absolute
target `T`, not a new target of `S + T`. Verify the loaded step and first new
completed update. Do not add step counters across submissions or count replayed
updates as new progress; show the recovered step, current completed step and
remaining `max(0, T - current_step)` separately. A report or queue finishing
cannot remove still-unfinished requested training from the shared monitor.

Do not impose a fixed retry count on authorized scoped recovery; re-establish
the failure and all preconditions for each attempt. If ownership or verification
is unavailable, or a required fix exceeds scope, stop recovery and report the
blocker. Continue monitoring a recovered job on the original cadence, with old
and new IDs, failure, fix, verification and genuinely new progress distinguished.
