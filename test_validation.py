import os
import sys
import hydra
from omegaconf import OmegaConf

# 1. Compose recipe/phimm/config/eval/eval_ls_2607v1.yaml using hydra.initialize_config_dir
# Needs absolute path for initialize_config_dir usually.
config_dir = os.path.abspath("recipe/phimm/config")
print(f"Initializing hydra with config_dir: {config_dir}")

try:
    # Clear any active Hydra instance to prevent overrides/reinitialization issues
    hydra.core.global_hydra.GlobalHydra.instance().clear()
except Exception:
    pass

hydra.initialize_config_dir(config_dir=config_dir, version_base=None)
cfg = hydra.compose(config_name="eval/eval_ls_2607v1")

print("Validating configuration assertions:")
print("path:", cfg.val_reward.custom_reward_function.path)
print("name:", cfg.val_reward.custom_reward_function.name)

assert cfg.val_reward.custom_reward_function.path == 'recipe/phimm/reward/asr_bias.py', f"Unexpected path: {cfg.val_reward.custom_reward_function.path}"
assert cfg.val_reward.custom_reward_function.name == 'eval_score', f"Unexpected name: {cfg.val_reward.custom_reward_function.name}"

# Check datasets: two lang_asr zero-distractor datasets (ls_bias with cache_name: auto_clean_0_lang_asr / auto_other_0_lang_asr)
# Let's inspect the resolved datasets
datasets = cfg.data.val_data
print(f"Loaded {len(datasets)} datasets:")
for d in datasets:
    print(f" - dataset_name: {d.dataset_name}, cache_name: {d.cache_name}, task: {d.add_task_info.task}")

assert len(datasets) == 2, f"Expected 2 datasets, got {len(datasets)}"
for d in datasets:
    assert d.add_task_info.task == 'lang_asr', f"Expected lang_asr task, got {d.add_task_info.task}"
    assert '0_lang_asr' in d.cache_name, f"Expected 0-distractor (0_lang_asr), got {d.cache_name}"

print("Step 1 PASSED.")

# 2. Import eval_score and test
from recipe.phimm.reward.asr_bias import eval_score
print("Imported eval_score successfully.")

ref = 'the quick brown fox'
hyp = 'the quick blue fox'
extra_info = {'keywords': ['brown']}

res = eval_score(solution_str=hyp, ground_truth=ref, extra_info=extra_info)
print("eval_score result:", res)

expected = {
    'n_err': 1,
    'n_ref': 4,
    'nu_err': 0,
    'nu_ref': 3,
    'nb_err': 1,
    'nb_ref': 1
}

for k, val in expected.items():
    assert res[k] == val, f"For {k}: expected {val}, got {res[k]}"

print("Step 2 PASSED.")

# 3. Feed minimal var2metric2val to update_var2metric2val
from verl.trainer.ppo.metric_utils import update_var2metric2val
print("Imported update_var2metric2val successfully.")

# Let's check update_var2metric2val signature/functionality
import inspect
print(inspect.getsource(update_var2metric2val))

