# Monitoring and scoped recovery

Track only jobs explicitly identified by the user or launched for this request;
never add other discovered jobs. Save exact experiment/config, cluster,
namespace, pool, submission ID, lineage, target, resource restrictions, recovery
boundary, and requested end condition. Rediscover current Ray submissions on
each pass and update identity after a scoped replacement. A status-only or
read-only request never authorizes recovery or other mutations.

For requested periodic monitoring, use one chat-attached heartbeat, defaulting
to 30 minutes if no cadence was specified. Inspect existing schedules and
ownership first to avoid competing monitors. For a pipeline use one monitor for
all phases, not one per job. Its durable prompt identifies this skill, state
path where applicable, tracked identities, queue policy, recovery/read-only
boundary, end and stop conditions. Update it when phase or submission changes.
Keep unchanged passes quiet unless the user asks for an update every pass. On a
report, include each tracked job and its phase, live identity/pods, progress and
delta (including `no progress change`), latest per-dataset validation and
checkpoint evidence, GPU/process health, changed risks and required action.
Highlight meaningful progress, new evidence, failures or completion; do not
infer completion from absence or a stall from one quiet interval. Stop the
heartbeat after all requested outcomes are verified and reported, user
cancellation, or a terminal blocker requiring user action.

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

Do not impose a fixed retry count on authorized scoped recovery; re-establish
the failure and all preconditions for each attempt. If ownership or verification
is unavailable, or a required fix exceeds scope, stop recovery and report the
blocker. Continue monitoring a recovered job on the original cadence, with old
and new IDs, failure, fix, verification and genuinely new progress distinguished.
