# Agentic rollout support: source walkthrough

Inspected on 2026-11-12 at `origin/main`, commit
`662507f9582e16d09c12484387f27727b5d0f947` in `boren-ms/verl`.
This is a source review, not an end-to-end training validation.

## Recommended integration

Agentic rollout already exists. Use `ToolAgentLoop` for model-driven tool
calling; implement `AgentLoopBase.run()` for a custom environment, planner,
reflection loop, or external agent framework. Keep the existing inference,
weight synchronization, reward, and training infrastructure unless its contract
cannot represent the application.

## Current execution flow

The default configuration enables `trainer.use_v1=true` and
`trainer.v1.trainer_mode=sync`. The latter controls training scheduling;
`actor_rollout_ref.rollout.mode=async` controls inference requests. Concurrent
agent coroutines do not require asynchronous policy updates.

1. `verl/trainer/main_ppo.py:TaskRunnerV1.init_agent_loop_manager` creates
   `AgentLoopManagerTQ` with the trainer's LLM client and reward handles.
2. `verl/trainer/ppo/v1/agent_loop_tq.py:AgentLoopManagerTQ.generate_sequences`
   dispatches prompt chunks to workers. Each worker starts background tasks,
   including `rollout.n` sessions per training prompt.
3. The inherited `AgentLoopWorker._run_agent_loop` selects a registered loop by
   the sample's `agent_name`, instantiates it, and awaits `run()`.
4. The loop calls `LLMServerClient.generate` with token IDs. The shared load
   balancer routes by in-flight requests and preserves session affinity when
   possible. Reuse a trajectory's request ID across model calls.
5. The loop returns `AgentLoopOutput`. TQ postprocessing computes reward if
   needed, prepares model inputs, assigns `loss_mask=response_mask`, and writes
   trajectories plus status/version tags to TransferQueue for the trainer.

The legacy `trainer.use_v1=false` path still exists. Its `AgentLoopManager`
returns padded `DataProto` batches directly. A custom manager must implement
the contract of the selected trainer; replacing the default TQ manager with a
legacy batch-returning manager is insufficient.

## Built-in tool calling

`verl/experimental/agent_loop/tool_agent_loop.py:ToolAgentLoop` runs:

```text
build prompt -> generate assistant -> parse tool calls
                    ^                       |
                    |                       v
             append observations <- execute tools
```

It terminates when the model emits no tool call, a configured turn limit is
reached, or the response budget is reached. Tool calls within a turn run
concurrently up to `max_parallel_calls`; the implementation takes only the
first configured number of parsed calls.

Two tool APIs are available:

- `verl/tools/function_tool.py:@function_tool`: typed Python functions with
  Google-style docstrings, loaded through `multi_turn.function_tool_path`.
  Async functions are awaited and sync functions run in a thread. These tools
  do not receive per-trajectory state injection.
- `verl/tools/base_tool.py:BaseTool`: explicit schemas and lifecycle hooks,
  loaded through `multi_turn.tool_config_path`. In the built-in loop,
  `create -> execute -> release` happens **per tool invocation**. Persist
  episode state explicitly, for example through `agent_data.extra_fields`, or
  own the environment lifecycle in a custom loop.

Tools are loaded once per worker. Per-sample `extra_info.tool_selection` can
filter them. The dataset forwards `extra_info.tools_kwargs` as `tools_kwargs`.

The following overrides illustrate the integration into an otherwise complete
training configuration; the paths are application-provided placeholders:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm  # or a supported SGLang configuration
    mode: async
    agent:
      default_agent_loop: tool_agent
      num_workers: 8
    multi_turn:
      enable: true
      function_tool_path: /absolute/path/to/tools.py
      format: hermes  # match the model's actual tool-call format
      max_assistant_turns: 8
      max_user_turns: 8
      max_parallel_calls: 2
      max_tool_response_length: 2048
```

Alternatively set dataset `agent_name: tool_agent` to select tools per sample.
Setting `multi_turn.enable=true` alone does not select `ToolAgentLoop`.
The parser registry includes `qwen3_coder`, documented in source for
Qwen3-Coder/Qwen3.5; parser choice must match actual model output.

## Custom loop contract

Subclass `AgentLoopBase` and implement
`async run(self, sampling_params, **kwargs) -> AgentLoopOutput`.
Read `kwargs["raw_prompt"]` and application fields, then alternate model calls
and environment actions until termination. Use the base helpers:

- `ct_build_initial_tokens` for the initial conversation;
- `ct_merge_assistant_token` for model-returned token IDs and log probabilities;
- `ct_merge_non_assistant_msg` for appended observations and their metadata.

Register importable implementations through an agent YAML file:

```yaml
- name: environment_agent
  _target_: my_package.agent_loop.EnvironmentAgentLoop
```

Set `actor_rollout_ref.rollout.agent.agent_loop_config_path` to that YAML and
select `environment_agent` in the dataset or `default_agent_loop`. The worker
imports the target, so application code and dependencies must be available on
every relevant Ray node. Decorator registration also exists, but its module
must actually be imported in worker processes.

Return initial `prompt_ids`, the full post-prompt `response_ids`, aligned
`response_mask`, `metrics`, and optional `response_logprobs`, `reward_score`,
multimodal data, and `extra_fields`.

## Correctness and practical limits

- Preserve the generated token stream. Do not reconstruct the training
  trajectory by decoding and re-tokenizing the final chat history. The current
  Continuous Token builder also handles model-specific boundaries.
- Assign mask 1 to generated assistant tokens, including generated tool-call
  syntax; assign mask 0 to tool observations, inserted boundaries, and padding.
  Observation tokens remain visible as attention context. Align log
  probabilities to the same response positions.
- Treat `response_length` as a whole-trajectory budget including tool
  observations. The built-in loop caps its output; custom loops must enforce
  their own lengths. Validate the inference context limit as well.
- `tool_rewards` is collected in `extra_fields`; the loop does not automatically
  add it to the policy reward. Supply `reward_score` or explicitly aggregate
  these fields in the reward function. The default reward path can decode the
  whole response, including observations, so task scoring must extract the
  intended final answer.
- The TQ adapter accepts lists of trajectory outputs, but contains remaining
  TODOs around multi-output support. Start with one output per session; test
  reward propagation, grouping, and distillation before relying on branches.
- Add environment timeouts and cleanup in custom code. The built-in tool
  dispatch has no general timeout wrapper. For asynchronous policy training,
  preserve version fields used by staleness filtering.
- Input audio is forwarded by the loop/client, but compatibility still depends
  on the concrete processor, Continuous Token builder, inference backend, and
  training model. The built-in tool-result path handles text/images, explicitly
  rejects video results, and does not append tool-produced audio.
- The TQ worker currently disables its per-sample trajectory tracing with a
  TODO. Do not assume the legacy tracing documentation applies unchanged.

## Existing evidence and next validation

Relevant existing tests include `tests/experimental/agent_loop/test_basic_agent_loop.py`,
`test_call_tool_on_cpu.py`, `test_tool_call_id_on_cpu.py`,
`test_audio_server_contract_on_cpu.py`, `tests/tools/test_function_tool_on_cpu.py`,
and `tests/utils/test_continuous_token_on_cpu.py`. They were located and reviewed
selectively, not executed for this walkthrough.

For an implementation, first validate a deterministic two-turn episode with
one tool: generated IDs unchanged, aligned masks/logprobs, observation visible
on turn two, known final reward, length termination, and cleanup after failure.
Then validate through the default TQ manager and one optimizer update before
scaling concurrency or selecting an asynchronous trainer.

Some older docs reference `examples/sglang_multiturn` and
`recipe/langgraph_agent`, which are absent from this checkout. `recipe/phimm`
is also absent from this `main`; ASR recipe integration is separate work.

## Base3a Qwen ASR adaptation in this worktree

The [ASR package](../plugins/phimm/README.md) applies this integration through
`ASRAgentLoop`, a TransferQueue worker, and the synchronous V1 trainer. The
loop preserves generated training tokens/log probabilities, shares one greedy
ReMax baseline across each prompt's eight samples, and groups validation
segments before scoring parent transcripts. The adapter imports the selected
development recipe's model, data, and scoring dependencies into an editable
namespace package so main's `recipe` submodule stays intact.

Runtime checks require the original 55,664-row mix and 868-update target.
Unit-level loss and reward equivalence checks establish individual contracts;
training, checkpoint reloadability, and dataset-level quality require the
separate reproduction pipeline. Its acceptance threshold is at most 0.1
percentage-point regression per dataset with matched coverage and denominators.

## Completed Qwen ASR integration

See the [Base3a code walkthrough and model integration workflow](experiments/base3a_main_reproduction_20261113/CODE_WORKFLOW.md)
and the [completed experiment report](experiments/base3a_main_reproduction_20261113/README.md)
for the implementation built on this source review. The full pipeline completed;
the report records the failed strict quality parity gates.
