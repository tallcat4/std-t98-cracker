"""Plot FIRST accepted answers, including wrong answers and abstentions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from benchmark_guided_ai import aggregate, classification


def first_time(row, processing=False):
    first = row['first_confirmation']
    if first is None:
        return None
    if not processing:
        return first['end_burst'] * .08
    finish = 0.
    for i, milliseconds in enumerate(row['step_ms'][:first['end_burst']]):
        finish = max(finish, (i + 1) * .08) + milliseconds / 1000
    return finish


def correct_first(row):
    first = row['first_confirmation']
    return first is not None and (first['series'], first['key']) == (row['series'], row['key'])


def at_deadline(rows, deadline, processing=False):
    counts = dict(correct=0, wrong=0, unresolved=0)
    for row in rows:
        time = first_time(row, processing)
        outcome = 'unresolved' if time is None or time > deadline else 'correct' if correct_first(row) else 'wrong'
        counts[outcome] += 1
    return dict(**counts, **classification(counts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, help='Completed paired benchmark directory')
    p.add_argument('--output', type=Path, help='New output basename; defaults to directory/first-confirmation')
    p.add_argument('--clock', choices=('processing', 'traffic'), default='processing')
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.directory/'trials.jsonl').read_text(encoding='utf-8').splitlines()]
    plan = json.loads((args.directory/'plan.json').read_text(encoding='utf-8'))
    if len(rows) != len(plan) * 2 or len({(r['case'], r['encoder']) for r in rows}) != len(rows):
        raise ValueError('Benchmark is incomplete or contains duplicate cases')
    output = args.output or args.directory/'first-confirmation'
    targets = [output.with_suffix(ext) for ext in ('.png', '.svg', '.json')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Choose a new output basename')
    for path in ('C:/Windows/Fonts/meiryo.ttc', 'C:/Windows/Fonts/YuGothM.ttc'):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    groups = {name: [r for r in rows if r['encoder'] == name] for name in ('thumbdv', 'ambe2-c')}
    processing = args.clock == 'processing'
    xmax = max(3.2, max((first_time(r, processing) or 0.) for r in rows) + .24)
    edges = np.arange(0, np.ceil(xmax / .08) * .08 + .001, .08)
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    fig.subplots_adjust(top=.96, bottom=.16, left=.09, right=.97)
    total = at_deadline(rows, float('inf'))
    good = [first_time(r, processing) for r in rows if r['first_confirmation'] and correct_first(r)]
    bad = [first_time(r, processing) for r in rows if r['first_confirmation'] and not correct_first(r)]
    ax.hist([good, bad], bins=edges, stacked=True, color=['#2776ad', '#cd4c4c'],
            edgecolor='white', linewidth=.7, label=[f"正解  {len(good)}件", f"誤答  {len(bad)}件"])
    ax.set_xlabel('最初に回答を確定するまでの時間（秒）', labelpad=11)
    ax.set_ylabel('頻度（試行数）', labelpad=9)
    ax.set_xlim(0, edges[-1])
    ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.text(.97, .71, f"F1 = {total['f1']:.3f}", transform=ax.transAxes, ha='right', fontsize=19,
            color='#233341', bbox=dict(boxstyle='round,pad=.5', facecolor='#f1f4f7', edgecolor='none'))
    report = dict(first_answer_metrics=total, summary={name: aggregate(group) for name, group in groups.items()},
                  clocks={}, definition='First accepted answer, whether correct or wrong; never replace it with a later correct answer.')
    for simulated in (False, True):
        report['clocks']['processing_simulation' if simulated else 'traffic'] = {
            name: {str(t): at_deadline(group, t, simulated) for t in (1., 3., 5.)}
            for name, group in groups.items()}
    report.update(bin_edges=edges.tolist(), plotted_clock=args.clock,
                  bin_correct=np.histogram(good, edges)[0].tolist(), bin_wrong=np.histogram(bad, edges)[0].tolist(),
                  timing_assumption='80 ms burst arrivals and measured serial processing; excludes cold start, RF and USB delays')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(targets[0], dpi=180)
    fig.savefig(targets[1])
    targets[2].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(total), flush=True)
    print(targets[0], flush=True)


if __name__ == '__main__':
    main()
