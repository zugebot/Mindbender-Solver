# Sums a Gecko profile's CPU per thread over its last N seconds (ff_eval.mjs PROF=...), so the
# idle page's cost can be split into the main thread, the renderer, the compositor, canvas.
# usage: python prof_threads.py prof.json [seconds]   (the last 2 s, Firefox closing, left out)
import json
import sys

prof = json.load(open(sys.argv[1], encoding='utf-8'))
window = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
SKIP = 2.0      # the close at the end (browser.close) is left out
rows = []
ends = []


def lastTime(p):
    """The last sample of any thread, in ms since the parent process started."""
    off = p['meta']['startTime'] - prof['meta']['startTime']
    for t in p['threads']:
        d = t['samples']['data']
        if d:
            ends.append(off + d[-1][t['samples']['schema']['time']])
    for sub in p.get('processes', []):
        lastTime(sub)


lastTime(prof)
END = max(ends)


def walk(p, proc):
    off = p['meta']['startTime'] - prof['meta']['startTime']
    unit = p['meta'].get('sampleUnits', {}).get('threadCPUDelta', 'µs')
    for t in p['threads']:
        s = t['samples']
        schema = s['schema']
        if 'threadCPUDelta' not in schema:
            continue
        ti, ci = schema['time'], schema['threadCPUDelta']
        data = s['data']
        if not data:
            continue
        lo, hi = END - (window + SKIP) * 1000 - off, END - SKIP * 1000 - off
        cpu = sum((d[ci] or 0) for d in data if len(d) > ci and lo <= d[ti] < hi)
        rows.append((cpu, unit, proc, t.get('processType', ''), t['name']))
    for sub in p.get('processes', []):
        walk(sub, sub['meta'].get('product', '') + str(sub['threads'][0].get('pid', '')) if sub['threads'] else '?')


walk(prof, 'parent')
total = sum(r[0] for r in rows if r[1] in ('µs', 'ns'))
for cpu, unit, proc, ptype, name in sorted(rows, reverse=True)[:18]:
    scale = 1000 if unit == 'µs' else 1e6 if unit == 'ns' else 1
    print(f'{cpu / scale / window / 10:6.1f}% of a core  {ptype:10s} {name}  ({unit})')
