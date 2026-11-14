import json
import os
import runpy
import sys
from pathlib import Path

# runs sabre.py with scripted (replays $SCRIPTED_ACTIONS) registered at runtime and comyco wrapped to
# record its decisions; $COMYCO_WEIGHTS picks the weights, a session summary goes to $SABRE_SUMMARY
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('STREAMLENS_SRC', HERE.parent.parent / 'StreamLens/src'))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE.parent))

import abr_algorithms as A  # noqa: E402
from global_state import gs  # noqa: E402


class Scripted(A.Abr):
    def __init__(self, config):
        self.actions = json.load(open(os.environ['SCRIPTED_ACTIONS']))
        self.records = []
        Scripted.instance = self

    def get_first_quality(self):
        return self.actions[0]

    def get_quality_delay(self, segment_index):
        bl = A.get_buffer_level(gs.manifest.segment_time, gs.buffer_contents, gs.buffer_fcc)
        self.records.append({'seg': segment_index, 'decision_buffer_ms': bl})
        return (self.actions[segment_index], 0)

    def report_download(self, metrics, is_replacment):
        r = self.records[-1]
        r.update(quality=metrics.quality, size=metrics.size, time=metrics.time, ttfb=metrics.time_to_first_bit)


A.abr_list['scripted'] = Scripted

class Comyco(A.Comyco):
    def __init__(self, config):
        super().__init__(config)
        self.actions = [0]
        Comyco.instance = self

    def get_quality_delay(self, segment_index):
        out = super().get_quality_delay(segment_index)
        self.actions.append(out[0])
        return out


A.abr_list['comyco'] = Comyco

sys.argv = [str(SRC / 'sabre.py')] + sys.argv[1:]
try:
    runpy.run_path(str(SRC / 'sabre.py'), run_name='__main__')
finally:
    out = os.environ.get('SABRE_SUMMARY')
    if out and gs.manifest is not None:
        T = gs.total_play_time
        seg = gs.manifest.segment_time
        mean_util = gs.played_utility / (T / seg)
        rebuf_ratio = gs.rebuffer_time / T
        switch_rate = gs.switch_count / (T / 1000.0)
        summ = dict(
            qoe=mean_util - 10 * rebuf_ratio - 1 * switch_rate,
            mean_utility=mean_util,
            rebuf_ratio=rebuf_ratio,
            rebuf_s=gs.rebuffer_time / 1000.0,
            rebuf_events=gs.rebuffer_event_count,
            switches=gs.switch_count,
            switch_rate=switch_rate,
            total_play_s=T / 1000.0,
            startup_ms=gs.startup_time,
        )
        if hasattr(Scripted, 'instance'):
            summ['records'] = Scripted.instance.records
        if hasattr(Comyco, 'instance'):
            summ['comyco_actions'] = Comyco.instance.actions
        json.dump(summ, open(out, 'w'))
