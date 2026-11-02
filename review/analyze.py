"""Audit downloaded E007 JSON records and plot completion results; no training."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'
RUN = E / 'runs/rl'
load = lambda p: json.loads(p.read_text())
summary = load(RUN / 'final_summary.json')
config = load(RUN / 'config.json')
digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
assert summary['completed_updates'] == 1000
assert digest(config) == summary['identity']['config']
assert config['initial_step'] == 6500
assert config['initial_checkpoint_sha256'] == summary['identity']['initial_model']
assert config['sft_parent_identity'] == load(E / 'runs/sft/final_summary.json')['identity']
assert load(E / 'data/rl/manifest.json')['sha256'] == summary['identity']['data']
assert load(E / 'artifacts/pipeline.json')['status'] == 'complete'
assert load(E / 'artifacts/rl_train.json')['exit'] == 0
experiment = E / 'source_snapshot'
deployment = load(E / 'deployment.json')
for name, expected in deployment['sources'].items():
    path = E / name if name in ['config.json', 'rl_config.json'] else experiment / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
assert digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(experiment.glob('*.py'))}) == summary['identity']['source']

budgets = config['test_budgets']
selected = summary['selected_step']
all_rows = {}
for label, step in [('student', selected), ('frozen', 0), ('teacher_policy', 950)]:
    folder = 'test_teacher' if label == 'teacher_policy' else 'test_' + label
    records = []
    for rank in range(4):
        records.extend(load(RUN / f'evaluation/{folder}/per_graph_step{step:06d}_rank{rank}.json'))
    indexed = {(r['graph'], r['budget']): r for r in records}
    assert len(indexed) == len(records) == 4000
    assert set(indexed) == {(g, b) for g in range(1000) for b in budgets}
    for b in budgets:
        rows = [indexed[g, b] for g in range(1000)]
        for key, skey in [('cost', 'cost'), ('initial', 'initial'),
                          ('first4_cost', 'first4_cost'), ('single_cost', 'single_cost')]:
            assert math.isclose(st.mean(r[key] for r in rows), summary[label]['by_budget'][str(b)][skey], abs_tol=1e-9)
        for g in range(1000):
            costs = [indexed[g, k]['cost'] for k in budgets]
            assert costs == sorted(costs, reverse=True)
    all_rows[label] = indexed

heur = load(RUN / 'evaluation/test_heuristics.json')
assert [r['graph'] for r in heur['records']] == list(range(1000))
for key in ['initial', 'teacher', 'BFS', 'DFS']:
    assert math.isclose(st.mean(r[key] for r in heur['records']), summary['heuristics'][key], abs_tol=1e-9)
for key in all_rows:
    assert all(all_rows[key][g, b]['initial'] == heur['records'][g]['initial'] for g in range(1000) for b in budgets)

paired = {}
for b in budgets:
    paired[str(b)] = {}
    student = [all_rows['student'][g, b]['cost'] for g in range(1000)]
    for ref in ['teacher_policy', 'frozen', 'teacher', 'BFS', 'DFS']:
        reference = ([all_rows[ref][g, b]['cost'] for g in range(1000)]
                     if ref in all_rows else [r[ref] for r in heur['records']])
        improvements = [(r-s)/r for r, s in zip(reference, student)]
        mean = st.mean(improvements)
        se = st.stdev(improvements) / math.sqrt(1000)
        result = dict(relative_improvement=mean, ci95=[mean-1.96*se, mean+1.96*se],
                      wins=sum(s < r for s, r in zip(student, reference)), graphs=1000)
        paired[str(b)][ref] = result
        if ref in ['teacher_policy', 'frozen']:
            original = summary['paired_teacher' if ref == 'teacher_policy' else 'paired'][str(b)]
            assert result['wins'] == original['wins']
            assert math.isclose(mean, original['relative_improvement'], abs_tol=1e-12)

metrics = [load(p) for p in sorted((RUN / 'metrics').glob('step*.json'))]
assert [m['step'] for m in metrics] == list(range(1, 1001))
assert all(math.isfinite(m[k]) for m in metrics for k in ['loss', 'grad_norm', 'recompute_logp_abs_delta'])
assert all(m['global_actions'] == 131072 for m in metrics)
validation = [load(p) for p in sorted((RUN / 'evaluation/validation').glob('summary_step*.json'))]
assert [v['step'] for v in validation] == list(range(0, 1001, 50))
best = min(validation, key=lambda v: v['by_budget']['1024']['cost'])
assert best['step'] == selected == 1000
result = dict(paired=paired, selected_step=selected, graphs=1000,
              median_update_seconds=st.median(m['step_seconds'] for m in metrics),
              peak_allocated_bytes=max(m['peak_vram_bytes'] for m in metrics),
              max_logp_replay_error=max(m['recompute_logp_abs_delta'] for m in metrics),
              training_seconds=summary['elapsed_training_seconds'],
              final_model_evaluation_seconds=sum(summary[k]['seconds'] for k in all_rows),
              initial_validation_cost=validation[0]['by_budget']['1024']['cost'],
              selected_validation_cost=best['by_budget']['1024']['cost'],
              caveat='Graph-level intervals conditional on one trained seed; no continued-RL-only control or decoder timing.')
(ROOT / 'audited_results.json').write_text(json.dumps(result, indent=2))
table = []
for b in budgets:
    table.append(dict(swaps_per_rollout=b, new_policy=summary['student']['by_budget'][str(b)]['cost'],
                      new_SFT=summary['frozen']['by_budget'][str(b)]['cost'],
                      old_learned_teacher=summary['teacher_policy']['by_budget'][str(b)]['cost'],
                      spectral=summary['heuristics']['teacher'], BFS=summary['heuristics']['BFS'], DFS=summary['heuristics']['DFS']))
with (ROOT / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=table[0].keys())
    writer.writeheader(); writer.writerows(table)
(ROOT / 'evidence_sha256.json').write_text(json.dumps({str(p.relative_to(E)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(E.rglob('*.json'))}, indent=2))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
orange = '#e47e26'
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1.5]})
x = [v['step'] for v in validation]
y = [v['by_budget']['1024']['cost'] for v in validation]
ax.plot(x, y, color=orange, marker='o', markersize=3, linewidth=2)
ax.axhline(y[0], color='#6d7781', linestyle='--', linewidth=1, label='SFT initialization')
ax.scatter([selected], [y[-1]], s=70, color=orange, zorder=4)
ax.annotate(f'{y[-1]:,.2f}', (selected, y[-1]), xytext=(-8, 13), textcoords='offset points', ha='right', color='#9b4c0a')
ax.set(xlabel='RL update', ylabel='Mean recursive calls ↓')
ax.set_title('RL improves the pretrained student', pad=29, fontsize=12)
ax.text(0, 1.02, '256 graphs · 4 rollouts × 1,024 swaps', transform=ax.transAxes, fontsize=9, color='#65717c')
ax.legend(loc='upper right', frameon=False, fontsize=9)
ax.grid(alpha=.15)

initial = summary['heuristics']['initial']
for key, label, color, ls in [('frozen', 'New SFT before RL', '#5783a5', '--'),
                               ('teacher_policy', 'Previous learned teacher', '#222b33', '-'),
                               ('student', 'New SFT + RL', orange, '-')]:
    bx.plot(budgets, [summary[key]['by_budget'][str(b)]['cost']/initial for b in budgets],
            label=label, color=color, linestyle=ls, marker='o', linewidth=2)
for key, label, color in [('teacher', 'Spectral bisection', '#498b69'), ('BFS', 'BFS', '#92709d'), ('DFS', 'DFS', '#689b9a')]:
    bx.axhline(summary['heuristics'][key]/initial, label=label, color=color, linestyle=':', linewidth=1.5)
bx.set_xscale('log', base=2)
bx.set_xticks(budgets, labels=['256', '512', '1,024', '2,048'])
bx.set(xlabel='Swaps per rollout (learned policies)',
       ylabel='Mean cost / initial mean cost ↓', ylim=(.832, .915))
bx.set_title('Fresh test: student surpasses teacher', pad=29, fontsize=12)
bx.text(0, 1.02, '1,000 graphs · learned policies use 16 rollouts each', transform=bx.transAxes, fontsize=9, color='#65717c')
bx.grid(alpha=.15)
bx.legend(loc='upper center', bbox_to_anchor=(.5, -.23), ncol=2, frameon=False, fontsize=9)
fig.text(.5, .015, 'BFS, DFS and spectral bisection produce one ordering each. This compares ordering quality, not equal compute.',
         ha='center', fontsize=9, color='#58636d')
fig.subplots_adjust(top=.82, bottom=.31, left=.075, right=.985, wspace=.30)
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(ROOT / f'results.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(json.dumps(result, indent=2))
