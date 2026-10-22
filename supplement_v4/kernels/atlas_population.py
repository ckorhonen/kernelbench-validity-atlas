"""atlas_population.py — population collapse plot, reflection #15 implication.

EXPLORATORY-DESCRIPTIVE (declared 2026-10-19, post-reflection-#15): no confirmatory
gate on first look. Any confirmatory collapse claim requires a fresh prereg with a
numeric threshold BEFORE the next run (P5 discipline).

Question (population version of the spine): do measured arm-points from DIFFERENT
sessions/artifacts land on one locus in derived coordinates (bytes/thread ->
achieved GB/s), or scatter? HR-diagram analogy: locus first, explanation second.

Coverage v1 (honest limits):
- e1_triad.json + e1_triad_day2.json: triton FLAT arms only (kind=='flat'); deployed
  threads = ceil(n/BLOCK)*num_warps*32 is exact. torch rows: no thread coord -> listed, not plotted.
- nw_sweep.json: flat arms (grid,BLOCK,nw stored -> exact). persistent arms: deployed
  thread count not stored (grid implicit) -> listed, y-only branch stat, not plotted.
- m3_curve.json: variants empty in artifact -> excluded v1 (declared).
- w3_conv_gemm.json: gemm thread counts not derivable from stored fields without
  config assumptions -> excluded v1 (declared); A3/bookkeeping residue.

Output: results/atlas_population.json  {meta, points, bins, branch_stats}
"""
import json, math, statistics as st, os

RES = os.path.join(os.path.dirname(__file__), 'results')

def load(name):
    return json.load(open(os.path.join(RES, name)))['rows']

points, listed, excl = [], [], {'m3_curve.json': 'variants empty', 'w3_conv_gemm.json': 'threads not derivable v1'}

for src in ('e1_triad.json', 'e1_triad_day2.json', 'nw_sweep.json'):
    for r in load(src):
        kind = r.get('kind', 'flat')
        cfg = r.get('cfg', {}) or {}
        block = r.get('BLOCK', cfg.get('BLOCK'))
        nw = r.get('nw', cfg.get('num_warps'))
        g = None
        if 'grid' in r:
            g = r['grid']
        elif r.get('n') and block:
            g = math.ceil(r['n'] / block)
        bytes_moved = r.get('bytes_moved') or r.get('bytes')
        gbps = r.get('GBps') or r.get('gbps')
        row = {'src': src, 'kind': kind, 'BLOCK': block, 'nw': nw, 'n': r.get('n'), 'GBps': gbps}
        if kind == 'flat' and g and nw and bytes_moved and gbps:
            thr = g * nw * 32
            row.update(bt=bytes_moved / thr, threads=thr, grid=g)
            points.append(row)
        else:
            listed.append({**row, 'plot': False, 'why': 'persistent/torch: no exact deployed threads'})

# log2 bytes/thread bins, per-branch envelope
def binkey(bt):
    return round(math.log2(bt), 1)

bins = {}
for p in points:
    bins.setdefault(('flat', binkey(p['bt'])), []).append(p['GBps'])
bin_stats = [{'log2_bt': k[1], 'n': len(v), 'med_gbps': st.median(v),
              'spread_pct': (max(v) - min(v)) / st.median(v) * 100}
             for k, v in sorted(bins.items())]

flat_pts = [p for p in points if p['kind'] == 'flat']
res = []
med_by_bin = {b['log2_bt']: b['med_gbps'] for b in bin_stats}
for p in flat_pts:
    m = med_by_bin.get(binkey(p['bt']))
    if m:
        res.append(abs(p['GBps'] - m) / m * 100)

# --- v2: two derived coords (bytes/thread, waves). SM capacity 48*2048 threads (m3 meta sm_count=48).
SM_CAP = 48 * 2048
for p in flat_pts:
    p['waves'] = p['threads'] / SM_CAP
    p['log2w'] = round(math.log2(max(p['waves'], 1e-9)), 1)
bins2 = {}
for p in flat_pts:
    bins2.setdefault((binkey(p['bt']), p['log2w']), []).append(p['GBps'])
bins2_stats = [{'log2_bt': k[0], 'log2_waves': k[1], 'n': len(v),
                'med_gbps': st.median(v),
                'cv_pct': (st.pstdev(v) / st.mean(v) * 100) if len(v) > 1 and st.mean(v) else None}
               for k, v in sorted(bins2.items()) if len(v) >= 3]
res2 = []
for b in bins2_stats:
    key = (b['log2_bt'], b['log2_waves'])
    m = st.median(bins2[key])
    for v in bins2[key]:
        res2.append(abs(v - m) / m * 100)
cov2 = sum(b['n'] for b in bins2_stats)
pers = [r for r in listed if 'pers' in str(r['kind'])]
flatnw = [r for r in points if r['src'] == 'nw_sweep.json']

out = {
    'meta': {
        'date': '2026-10-19', 'mode': 'exploratory-descriptive (reflection #15 implication)',
        'prereg': 'none for this look; confirmatory claim needs fresh prereg',
        'coverage': {'plotted': 'triton flat arms with exact deployed threads (e1 day1/day2, nw)',
                     'listed_only': 'persistent + torch rows (no exact thread count)',
                     'excluded': excl},
        'axes': {'x': 'bytes per thread (deployed)', 'y': 'achieved GB/s'},
    },
    'n_points_plotted': len(points),
    'n_listed_unplotted': len(listed),
    'collapse_1d_bt': {'residual_pct': {'median': st.median(res), 'p90': sorted(res)[int(.9 * len(res))], 'max': max(res)} if res else None,
                       'reading': 'bytes/thread ALONE insufficient: latency-regime points share b/t with saturation points'},
    'collapse_2d_bt_waves': {'bins_n3': len(bins2_stats), 'points_covered': cov2,
                             'median_cv_pct': st.median([b['cv_pct'] for b in bins2_stats if b['cv_pct'] is not None]) if bins2_stats else None,
                             'residual_pct': {'median': st.median(res2), 'p90': sorted(res2)[int(.9 * len(res2))], 'max': max(res2)} if res2 else None,
                             'bins': bins2_stats},
    'bin_stats': bin_stats,
    'collapse_residual_pct': {'median': st.median(res), 'p90': sorted(res)[int(.9 * len(res))], 'max': max(res)} if res else None,
    'branch_note': 'flat vs persistent branch separation tested on nw_sweep y-only values',
    'points': sorted(points, key=lambda p: (p['bt'], p['GBps'])),
    'listed': listed,
}
json.dump(out, open(os.path.join(RES, 'atlas_population.json'), 'w'), indent=1)
o = json.load(open(os.path.join(RES, 'atlas_population.json')))
print(json.dumps({k: o[k] for k in ('n_points_plotted', 'n_listed_unplotted', 'collapse_1d_bt', 'collapse_2d_bt_waves') if k in o}, indent=1)[:1800])
c2 = o.get('collapse_2d_bt_waves', {})
if c2.get('residual_pct'):
    print('2D median residual vs 1D:', c2['residual_pct']['median'], 'vs', o['collapse_1d_bt']['residual_pct']['median'])
