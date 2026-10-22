from pathlib import Path
import json, csv, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/boren/code/verl')
WORK = ROOT / 'tmp/experiment_comparisons/20261022T1811Z_bcr_error_analysis'
OUT = ROOT / 'reports/bcr_error_analysis_20261022T1811Z'
RUNS = json.loads((WORK / 'scalar_runs.json').read_text())
DATASETS = ['earning_chunk_verb', 'openml_bad', 'ls_clean', 'ls_other']
COLORS = {'Fix': '#223f69', 'R16': '#d36239', 'R64': '#138477'}
LABELS = {'earning_chunk_verb': 'Earnings (6 parents)', 'openml_bad': 'OpenML bad (393 utterances)', 'ls_clean': 'LibriSpeech clean (2620 utterances)', 'ls_other': 'LibriSpeech other (2939 utterances)'}

def vals(run, key):
    return {x['step']: x['metrics'][key] for x in RUNS[run]['val'] if key in x['metrics']}

def trainvals(run, key):
    return {x['step']: x['metrics'][key] for x in RUNS[run]['train'] if key in x['metrics']}

summary, matched, trends = [], [], []
for ds in DATASETS:
    key = f'val-core/{ds}/p_err/mean@1'
    maps = {r: vals(r, key) for r in RUNS}
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    common = [s for s in common if s % 20 == 0]
    for r, m in maps.items():
        steps = sorted(m)
        summary.append({'run': r, 'dataset': ds, 'key': key,
                        'initial': {'step': steps[0], 'value': m[steps[0]]},
                        'latest': {'step': steps[-1], 'value': m[steps[-1]]},
                        'best': {'step': min(steps, key=lambda s: m[s]), 'value': min(m.values())},
                        'observation_count': len(steps)})
        for s in steps:
            trends.append({'run': r, 'dataset': ds, 'step': s, 'p_err_fraction': m[s]})
    for target in ['R16', 'R64']:
        shared = sorted(set(maps[target]) & set(maps['Fix']))
        shared = [s for s in shared if s % 20 == 0]
        s = shared[-1]
        diff = [(p, 100 * (maps[target][p] - maps['Fix'][p])) for p in shared if p >= 80]
        matched.append({'comparison': target + '_vs_Fix', 'dataset': ds, 'step': s,
                        'fix': maps['Fix'][s], 'target': maps[target][s],
                        'delta_pp': 100 * (maps[target][s] - maps['Fix'][s]),
                        'relative_error_increase_percent': 100 * (maps[target][s] / maps['Fix'][s] - 1),
                        'common_scheduled_steps': shared,
                        'worse_checks_step80_plus': sum(v > 1e-12 for _, v in diff),
                        'better_checks_step80_plus': sum(v < -1e-12 for _, v in diff),
                        'equal_checks_step80_plus': sum(abs(v) <= 1e-12 for _, v in diff),
                        'check_denominator_step80_plus': len(diff)})

keys = sorted({key for r in RUNS.values() for row in r['train'] for key in row['metrics'] if key.startswith(('reward-core/', 'reward-delta/', 'reward-vote/')) and key.endswith('/mean')})
reward = []
for key in keys + ['actor/grad_norm', 'actor/kl_loss', 'response_length/clip_ratio', 'response/aborted_ratio', 'train/gen_kept_frac']:
    for run in RUNS:
        m = trainvals(run, key)
        observed = [s for s in range(401, 441) if s in m]
        values = [m[s] for s in observed]
        reward.append({'run': run, 'key': key, 'step_start': 401, 'step_end': 440,
                       'n': len(values), 'mean': statistics.mean(values) if values else None,
                       'stdev': statistics.stdev(values) if len(values) > 1 else None,
                       'reason': None if values else 'metric absent; not zero', 'steps': observed})

analysis = {'snapshot_policy': 'Frozen retained model lineages; no interpolation. R16/Fix collection 17:34-17:36 UTC; R64 source through 17:49 UTC. Latest matched pair steps580 and440.',
            'runs': {r: {'latest_observed_optimizer_step': max(x['step'] for x in v['train']), 'train_rows': len(v['train']), 'validation_rows': len(v['val'])} for r, v in RUNS.items()},
            'validation_summary': summary, 'pairwise_latest': matched, 'reward_window': reward,
            'metric_inventory': keys, 'cross_dataset_average': None,
            'limits': ['These are pre-horizon single-trajectory snapshots; current Ray liveness must be observed separately. No independent seed replication.', 'Equal optimizer steps are not equal fresh prompt exposure.', 'R16 reward-core/lang is zero in this window; the deployed parser requires a target-language header. This cannot be interpreted as all transcription text having the wrong spoken language.', 'Best values are descriptive per dataset and can occur at different checkpoints.']}
(OUT / 'scalar_analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
(OUT / 'sources/scalar_runs.json').write_text(json.dumps(RUNS, ensure_ascii=False))
for name, rows in [('validation_history.csv', trends), ('matched_validation.csv', matched), ('reward_window.csv', reward)]:
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15})
for mode in ['validation', 'difference', 'difference_recent']:
    is_diff = mode.startswith('difference')
    fig, axs = plt.subplots(2, 2, figsize=(13.6, 8.2), constrained_layout=True)
    for ax, ds in zip(axs.flat, DATASETS):
        key = f'val-core/{ds}/p_err/mean@1'
        base = vals('Fix', key)
        for run in RUNS:
            m = vals(run, key)
            if is_diff:
                if run == 'Fix': continue
                steps = sorted(s for s in set(m) & set(base) if s >= (300 if mode == 'difference_recent' else 80) and s % 20 == 0)
                y = [100 * (m[s] - base[s]) for s in steps]
            else:
                steps = sorted(s for s in m if s >= 80 and s % 20 == 0)
                y = [100 * m[s] for s in steps]
            ax.plot(steps, y, color=COLORS[run], lw=1.7, marker='o', ms=3.1, label=run)
            if steps:
                ax.annotate(f'{y[-1]:+.3f}' if is_diff else f'{y[-1]:.3f}', (steps[-1], y[-1]), xytext=(6, 4 if run != 'R64' else -12), textcoords='offset points', color=COLORS[run], fontsize=9)
        if is_diff: ax.axhline(0, color=COLORS['Fix'], lw=1.0)
        ax.set_title(LABELS[ds], loc='left', fontweight='bold')
        ax.set_xlabel('Retained optimizer step')
        ax.set_ylabel('Target - Fix (percentage points)' if is_diff else 'Validation p_err (%)')
        ax.margins(x=.09)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('BCR vs direct Fix baseline | lower is better' if mode == 'validation' else 'BCR regression relative to Fix | above zero is worse', fontsize=14, fontweight='bold')
    fig.savefig(OUT / f'{mode}.png', dpi=170)
    fig.savefig(OUT / f'{mode}.svg')
    plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
for ax, key, title in zip(axs, ['reward-core/word/mean', 'reward-core/score/mean', 'actor/kl_loss'], ['Raw sampled word accuracy', 'Sampled - greedy weighted reward', 'Actor KL loss']):
    rows = [next(r for r in reward if r['key'] == key and r['run'] == run) for run in RUNS]
    ax.bar(list(RUNS), [r['mean'] for r in rows], color=[COLORS[r] for r in RUNS], width=.58)
    ax.set_title(title, loc='left', fontweight='bold')
    for i, r in enumerate(rows): ax.annotate(f"{r['mean']:.4f}", (i, r['mean']), xytext=(0, 5 if r['mean'] >= 0 else -12), textcoords='offset points', ha='center')
    ax.margins(y=.25)
fig.suptitle('Training diagnostics | exact shared steps401-440, n=40/run\nDifferent fresh/replay mixtures; not a fixed-data quality test', fontsize=13)
fig.savefig(OUT / 'reward.png', dpi=170)
fig.savefig(OUT / 'reward.svg')
print(json.dumps({'out': str(OUT), 'matched': matched}, ensure_ascii=False, indent=2))
