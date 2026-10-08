from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
RECORDS = next(p for p in (HERE.parent, HERE.parent / 'reviewer_records')
               if (p / 'specialist_formal_v1').is_dir())
FORMAL = RECORDS / 'specialist_formal_v1'
OUT = HERE / 'exports'
OUT.mkdir(exist_ok=True)

contract = {
    'conclusion': 'Recorded release policies reduce poisoned target selection while also reducing clean joint-correct selection; search-selected and fixed suffix conditions reuse identical outputs.',
    'archetype': 'quantitative grid',
    'backend': 'Python / matplotlib',
    'panel_a': 'Strict raw JSON/source-binding clean joint correctness and separately canonicalized references; not an isolated semantic-capability measure.',
    'panel_b': 'Poisoned-adapter SUPPORT selection on 39 non-SUPPORT claims; identical fixed/search-selected suffixes and reused outputs.',
    'panel_c': 'Poisoned-adapter clean joint-correct selection on all 75 claims.',
    'statistics': 'Arithmetic mean of training seeds 42/43/44; exact min--max whiskers; seed values shown in a/c. No population confidence interval or new significance test.',
    'reuse': 'Same claims/reference outputs across policies. Fixed and search-selected conditions are paired reused completions, not independent attack evaluations.',
    'size_inches': [7.16, 6.16],
    'font': 'Arial, 8.5--10 pt; embedded TrueType in vector PDF; SVG text retained.',
    'exports': ['PDF', 'SVG', '600 dpi PNG', '600 dpi TIFF', 'source CSV'],
    'new_model_calls': 0,
    'source_records_modified': False,
    'official_graphics_reference': 'https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/',
}
(HERE / 'FIGURE_CONTRACT.json').write_text(json.dumps(contract, indent=2) + '\n', encoding='utf-8')

cap = json.loads((FORMAL / 'FORMAL_CAPABILITY.json').read_text(encoding='utf-8'))
summary = json.loads((FORMAL / 'FORMAL_POLICY_SUMMARY.json').read_text(encoding='utf-8'))
audit = json.loads((HERE / 'INDEPENDENT_FIGURE_NUMERIC_AUDIT.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS' and audit['independent_raw_event_replay']
seeds = [42, 43, 44]
cap_by = {r['arm']: r for r in cap}
cohort_names = {'adaptive_trigger': 'adaptive', 'fixed_trigger': 'fixed', 'clean': 'clean'}
lookup = {(r['arm'], cohort_names[r['cohort']], r['policy']): r for r in summary}
policies = [r['policy'] for r in audit['panel_b']]
cohorts = sorted({r['cohort'] for r in summary})
assert cohorts == ['adaptive_trigger', 'clean', 'fixed_trigger'], cohorts
reference_arms = ['Base7B_three', 'Base14B_zero', 'Base14B_three']
raw = np.array([cap_by[a]['joint_correct'] for a in reference_arms])
canon = np.array([cap_by[a]['canonical_joint_correct'] for a in reference_arms])
benign = np.array([cap_by['benign_seed' + str(s)]['joint_correct'] for s in seeds])
poison = np.array([cap_by['poison_seed' + str(s)]['joint_correct'] for s in seeds])
assert raw.tolist() == audit['panel_a']['base_raw']
assert canon.tolist() == audit['panel_a']['base_canonical']
assert benign.tolist() == audit['panel_a']['benign']['seed_values_42_43_44']
assert poison.tolist() == audit['panel_a']['poison']['seed_values_42_43_44']

fixed = []
adaptive = []
clean = []
for i, p in enumerate(policies):
    values = {}
    for cohort, metric in [('fixed', 'target_selected'), ('adaptive', 'target_selected'), ('clean', 'joint_correct_selected')]:
        rows = [lookup[('poison_seed' + str(s), cohort, p)] for s in seeds]
        assert all(r['n'] == 75 and r['target_eligible_n'] == 39 for r in rows)
        values[cohort] = [r[metric] for r in rows]
    assert values['fixed'] == values['adaptive']
    assert values['fixed'] == audit['panel_b'][i]['fixed']['seed_values_42_43_44']
    assert values['adaptive'] == audit['panel_b'][i]['adaptive']['seed_values_42_43_44']
    assert values['clean'] == audit['panel_c'][i]['seed_values_42_43_44']
    fixed.append(values['fixed'])
    adaptive.append(values['adaptive'])
    clean.append(values['clean'])
fixed, adaptive, clean = map(np.array, [fixed, adaptive, clean])

source_rows = []
for i, a in enumerate(reference_arms):
    source_rows.extend([['a', a, 'raw_joint', 'shared', 75, int(raw[i])],
                        ['a', a, 'canonical_joint', 'shared', 75, int(canon[i])]])
for a, v in [('benign', benign), ('poison', poison)]:
    source_rows.extend([['a', a, 'raw_joint', s, 75, int(n)] for s, n in zip(seeds, v)])
for i, p in enumerate(policies):
    for cohort, v in [('fixed', fixed), ('adaptive', adaptive)]:
        source_rows.extend([['b', p, cohort + '_target_SUPPORT', s, 39, int(n)] for s, n in zip(seeds, v[i])])
    source_rows.extend([['c', p, 'poison_clean_joint', s, 75, int(n)] for s, n in zip(seeds, clean[i])])
with (HERE / 'Figure_S3_source_data.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['panel', 'arm_or_policy', 'metric_or_condition', 'training_seed', 'denominator', 'count'])
    w.writerows(source_rows)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False,
    'savefig.facecolor': 'white',
})
GRAY = '#6E7D88'
BLUE = '#244D64'
ORANGE = '#D19A3C'
RUST = '#B86447'
INK = '#202020'
fig = plt.figure(figsize=(7.16, 6.16))
axs = [fig.add_axes([.135, .715, .845, .195]),
       fig.add_axes([.135, .395, .845, .178]),
       fig.add_axes([.135, .090, .845, .168])]
for ax in axs:
    ax.tick_params(axis='x', length=0, pad=7)
    ax.tick_params(axis='y', length=3, pad=3)
    ax.spines['left'].set_color(INK)
    ax.spines['bottom'].set_color(INK)
    ax.set_axisbelow(True)
    ax.set_xlim(-.5, 6.5)

def heading(y, label, title):
    fig.text(.025, y, label, weight='bold', fontsize=10, va='center')
    fig.text(.135, y, title, weight='bold', fontsize=9.2, va='center')

def seed_bar(ax, x, values, color, width=.52):
    values = np.asarray(values)
    mean = values.mean()
    ax.bar(x, mean, width=width, color=color, zorder=2)
    ax.errorbar([x], [mean], yerr=[[mean-values.min()], [values.max()-mean]],
                fmt='none', ecolor=INK, elinewidth=.8, capsize=3, capthick=.8, zorder=3)
    for off, value, marker in zip([-.15, 0, .15], values, ['o', '^', 's']):
        ax.plot(x+off, value, marker=marker, ms=4.2, color=INK,
                markeredgewidth=.5, linestyle='none', zorder=4)

heading(.974, 'a', 'Clean capability: strict raw JSON + source binding')
ax = axs[0]
ax.set_xlim(-.5, 4.5)
ax.set_ylim(0, 75)
ax.set_yticks([0, 25, 50, 75])
ax.set_ylabel('Joint correct (of 75)')
ax.bar(np.arange(3), raw, width=.52, color=GRAY, zorder=2)
ax.plot(np.arange(3), canon, 'D', ms=5.3, markerfacecolor='white', markeredgecolor=INK, markeredgewidth=1, linestyle='none', zorder=4)
seed_bar(ax, 3, benign, BLUE)
seed_bar(ax, 4, poison, RUST)
ax.set_xticks(np.arange(5), ['7B\nthree-shot', '14B\nzero-shot', '14B\nthree-shot', 'Benign\nLoRA', 'Poisoned\nLoRA'])
handles = [Patch(color=GRAY, label='Raw joint'),
           Line2D([], [], marker='D', color=INK, markerfacecolor='white', linestyle='none', label='Canonical reference'),
           *[Line2D([], [], marker=m, color=INK, linestyle='none', label='Seed ' + str(s))
             for s, m in zip(seeds, ['o', '^', 's'])]]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.135, .941), ncol=5, borderaxespad=0,
           handlelength=1.1, columnspacing=1.15, handletextpad=.4, fontsize=8.5)

heading(.640, 'b', 'Poisoned adapters: target SUPPORT selections')
fixed_handle = Line2D([], [], marker='o', color=BLUE, linestyle='none', label='Fixed suffix')
adaptive_handle = Line2D([], [], marker='D', color=ORANGE, markerfacecolor='white', linestyle='none', label='Search-selected suffix')
fig.legend(handles=[fixed_handle, adaptive_handle], loc='center left', bbox_to_anchor=(.135, .609), ncol=2,
           borderaxespad=0, handlelength=1.4, columnspacing=1.5, handletextpad=.5, fontsize=8.5)
fig.text(.977, .609, 'Same suffix; outputs reused', ha='right', va='center', fontsize=8.5, color=INK)
ax = axs[1]
ax.set_ylim(0, 40)
ax.set_yticks([0, 13, 26, 39])
ax.set_ylabel('Target selected (of 39)')
labels = ['Raw\ncandidate', 'Source\nbinding', 'Conditional14', 'Full\nagreement14',
          'Raw-ref\nagreement14', 'Clean\nsolver14', 'Canonical\nsolver14']
for values, shift, color, marker, face in [(fixed, -.11, BLUE, 'o', BLUE),
                                          (adaptive, .11, ORANGE, 'D', 'white')]:
    means = values.mean(axis=1)
    err = np.array([means-values.min(axis=1), values.max(axis=1)-means])
    ax.errorbar(np.arange(7)+shift, means, yerr=err, fmt=marker, ms=4.7, color=color,
                markerfacecolor=face, markeredgewidth=.8, capsize=2.8, elinewidth=.8, capthick=.8, zorder=3)
ax.set_xticks(np.arange(7), labels)

heading(.306, 'c', 'Poisoned adapters: clean joint-correct selections')
fig.text(.135, .276, 'Bars = three-seed mean; whiskers = min-max; seed symbols as in a',
         va='center', fontsize=8.5, color=INK)
ax = axs[2]
ax.set_ylim(0, 75)
ax.set_yticks([0, 25, 50, 75])
ax.set_ylabel('Clean joint selected (of 75)')
for i, values in enumerate(clean):
    seed_bar(ax, i, values, RUST)
ax.set_xticks(np.arange(7), labels)
fig.text(.135, .018, 'All panels use recorded outputs; seed ranges are not population confidence intervals.',
         fontsize=8.5, va='center', color=INK)

# Verify all text bounds on the same Python rendering backend before export.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
text_bounds = []
for t in fig.findobj(match=matplotlib.text.Text):
    if not t.get_visible() or not t.get_text().strip():
        continue
    r = t.get_window_extent(renderer)
    assert r.x0 >= 0 and r.y0 >= 0 and r.x1 <= fig.bbox.width and r.y1 <= fig.bbox.height, (t.get_text(), list(r.bounds))
    text_bounds.append({'text': t.get_text(), 'bounds_pixels': list(r.bounds)})
for ext in ['pdf', 'svg', 'png', 'tiff']:
    kwargs = {'dpi': 600} if ext in ['png', 'tiff'] else {}
    if ext == 'tiff':
        kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(OUT / ('figure_S3_formal.' + ext), **kwargs)
plt.close(fig)
(HERE / 'FIGURE_EXPORT_QA.json').write_text(json.dumps({
    'figure_dimensions_inches': contract['size_inches'], 'all_text_inside_canvas': True,
    'counts_match_independent_raw_replay': True, 'raw_records_unchanged': True,
    'text_bounds': text_bounds,
    'source_fingerprints': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [FORMAL / 'FORMAL_CAPABILITY.json', FORMAL / 'FORMAL_POLICY_SUMMARY.json']},
}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'source_data_rows': len(source_rows), 'exports': 4, 'all_text_inside_canvas': True, 'counts_match_independent_replay': True}))
