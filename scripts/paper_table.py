import json
import sys
from pathlib import Path

# prints the StreamLens paper's Table 2 layout for every scenario: baseline linear buffer -> nonlinear buffer + prefetch
# usage: python scripts/paper_table.py results/paper_all
ABR_ORDER = ['bola', 'bolae', 'dynamic', 'dynamicdash', 'throughput', 'comyco']


def main():
    root = Path(sys.argv[1])
    scenarios = sorted(d for d in root.iterdir() if d.is_dir())
    print('%-34s %-12s %-12s %-16s %-16s %-14s' % ('Scenario', 'ABR', 'Rebuf events', 'Rebuf time (s)', 'Utility', 'QoE'))
    totals = {}
    for sc in scenarios:
        for abr in ABR_ORDER:
            f = sc / ('comparison_%s.json' % abr)
            if not f.exists():
                continue
            d = json.load(open(f))
            a, b = d['without_buffer_py']['summary'], d['with_buffer_py']['summary']
            print('%-34s %-12s %-12s %-16s %-16s %-14s' % (
                sc.name, abr,
                '%d -> %d' % (a['rebuffer_count'], b['rebuffer_count']),
                '%.2f -> %.2f' % (a['total_rebuffer_time'], b['total_rebuffer_time']),
                '%.1f -> %.1f' % (a['played_utility'], b['played_utility']),
                '%.2f -> %.2f' % (a['qoe_score'], b['qoe_score'])))
            t = totals.setdefault(abr, [0, 0, 0.0, 0.0, 0.0, 0.0, 0])
            t[0] += a['rebuffer_count']
            t[1] += b['rebuffer_count']
            t[2] += a['total_rebuffer_time']
            t[3] += b['total_rebuffer_time']
            t[4] += a['qoe_score']
            t[5] += b['qoe_score']
            t[6] += 1
    print()
    print('mean over %d scenarios' % len(scenarios))
    print('%-12s %-14s %-18s %-14s' % ('ABR', 'Rebuf events', 'Rebuf time (s)', 'QoE'))
    for abr, t in totals.items():
        n = t[6]
        print('%-12s %-14s %-18s %-14s' % (abr, '%.1f -> %.1f' % (t[0] / n, t[1] / n),
                                          '%.2f -> %.2f' % (t[2] / n, t[3] / n), '%.3f -> %.3f' % (t[4] / n, t[5] / n)))


if __name__ == '__main__':
    main()
