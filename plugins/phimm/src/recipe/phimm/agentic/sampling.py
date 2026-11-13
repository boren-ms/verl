"""ASR request sampling settings, independent of GPU worker initialization."""


def sampling_params_for_row(rollout_config, sampling_params, *, validate):
    """Apply ASR limits and the configured no-repeat window to a vLLM request."""
    custom = rollout_config.custom
    params = dict(sampling_params)
    params.update({k: list(custom[k]) for k in ("stop", "stop_token_ids") if k in custom})
    ngram_size = custom["validation_no_repeat_ngram_size"] if validate else custom["no_repeat_ngram_size"]
    # vLLM enables this processor when ngram_size is present and positive.
    # Its window parameter is named window_size; ngram_window is ignored.
    if ngram_size > 0:
        window_key = "validation_no_repeat_ngram_window_size" if validate else "no_repeat_ngram_window_size"
        params["extra_args"] = {"ngram_size": ngram_size, "window_size": custom[window_key]}
    params["max_tokens"] = custom["validation_response_length"] if validate else rollout_config.response_length
    return params
