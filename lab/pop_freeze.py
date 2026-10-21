"""Population-collapse CONFIRMATORY prereg — freeze BEFORE the fresh-point run.

Tick #31 execution of standing commitment #15 (2026-10-19): "population collapse
confirmatory prereg (numeric threshold fixed from atlas_population.json sampling
spread BEFORE the fresh-point run)".

Claim under test (ATLAS A3, exploratory): flat-elementwise GB/s collapses in the
2D derived coords (bytes/thread, waves); committed within-bin residual vs bin
median: median 0.55% / p90 14.68% (1136 pts, 204 bins n>=3).

CONFIRMATORY RULE (frozen here, no post-hoc changes):
  Run EVERY committed flat cell (BLOCK, num_warps, n) once, fresh session
  (day-3), e1 harness verbatim (same kernel, warmup=3 reps=15, flush 256MB,
  med). For each fresh point: bin = (round(log2 bt,1), round(log2 waves,1))
  with bt = bytes_moved/(ceil(n/BLOCK)*nw*32), waves = threads/98304.
  PRIMARY endpoint: P90_fresh = 90th pct of |GBps_fresh - med_committed(bin)|
  / med_committed * 100, over fresh points whose bin has committed n>=3.
  PASS iff P90_fresh <= T, where T = q95 of the bootstrap sampling
  distribution of that same statistic under H_collapase: for B=20000 draws,
  for each bin draw c_b values with replacement from the bin's COMMITTED
  values (c_b = number of fresh cells mapping to that bin), residual vs the
  FROZEN med, pool, take p90. T is computed HERE, before any fresh number
  exists.
  FAIL branch (P5, no rescue): P90_fresh > T => population collapse NOT
  confirmed at the preregistered bar; A3 stays exploratory with the sampling
  caveat restated; no threshold tuning, no subgroup rescue.
  Registered descriptives (no pass/fail): fresh median resid, max, per-bin
  union CV median, count excluded by the n>=3 rule.
  Infra note: a run aborted by GPU/env error may be rerun wholesale (log kept);
  numbers-driven reruns are not allowed.

Single run, day-3 session, spark GB10, torch 2.14.1+cu130. This file + freeze
JSON committed BEFORE the runner executes on spark.
"""
import json, math, os, hashlib, statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
POP = os.path.join(RES, 'atlas_population.json')
OUT = os.path.join(RES, 'pop_freeze.json')

d = json.load(open(POP))
pts = [p for p in d['points'] if p['kind'] == 'flat']
SM_CAP = 48 * 2048

def binkey(bt):
    return round(math.log2(bt), 1)

bins = {}
for p in pts:
    bins.setdefault((binkey(p['bt']), p['log2w']), []).append(p['GBps'])

# cells: distinct (BLOCK, nw, n) among committed flat points
cells = sorted({(p['BLOCK'], p['nw'], p['n']) for p in pts})

# per-cell derived coords, verbatim atlas formulas
def cell_bt(B, nw, n):
    g = math.ceil(n / B)
    thr = g * nw * 32
    return 12.0 * n / thr, thr / SM_CAP   # bytes_moved = 3*4*n (e1 verbatim)

cell_bin = {}
for (B, nw, n) in cells:
    bt, waves = cell_bt(B, nw, n)
    cell_bin[(B, nw, n)] = (binkey(bt), round(math.log2(max(waves, 1e-9)), 1))

# coverage: bins with committed n>=3 (the A3 collapse domain)
cov = {k: v for k, v in bins.items() if len(v) >= 3}
med = {k: st.median(v) for k, v in cov.items()}

# fresh counts per bin = number of CELLS mapping there (1 fresh run per cell)
c_b = {}
for (B, nw, n), k in cell_bin.items():
    if k in cov:
        c_b[k] = c_b.get(k, 0) + 1
n_covered_cells = sum(c_b.values())
n_excluded_cells = len(cells) - n_covered_cells

# bootstrap the p90 sampling distribution -> threshold T
B_BOOT = 20000
rng = __import__('random').Random(1531)
p90s = []
cov_items = [(k, v, c_b.get(k, 0)) for k, v in cov.items() if c_b.get(k, 0)]
for _ in range(B_BOOT):
    pool = []
    for k, v, c in cov_items:
        pool.extend(abs(v[rng.randrange(len(v))] - med[k]) / med[k] * 100 for _ in range(c))
    p90s.append(sorted(pool)[int(.9 * len(pool))])
p90s.sort()
T = p90s[int(.95 * len(p90s))]

committed_res = []
for k, v in cov.items():
    committed_res.extend(abs(x - med[k]) / med[k] * 100 for x in v)

out = {
    'date_utc': '2026-10-21T03:10Z',
    'prereg': 'see lab/pop_freeze.py docstring; frozen BEFORE fresh run',
    'population_sha256': hashlib.sha256(open(POP, 'rb').read()).hexdigest(),
    'rule': 'PASS iff P90_fresh(%) <= T; resid vs frozen committed bin median; bins committed n>=3 only',
    'T_pct': round(T, 4),
    'bootstrap': {'B': B_BOOT, 'seed': 1531, 'q50': round(p90s[len(p90s)//2], 4),
                  'q95': round(T, 4), 'max': round(p90s[-1], 4)},
    'committed_in_sample_p90': round(sorted(committed_res)[int(.9*len(committed_res))], 4),
    'committed_median_res': round(st.median(committed_res), 4),
    'n_bins_cov': len(cov), 'n_cells_total': len(cells),
    'n_cells_covered': n_covered_cells, 'n_cells_excluded_bin_lt3': n_excluded_cells,
    'cell_bin_map': {f'{B}|{nw}|{n}': k for (B, nw, n), k in sorted(cell_bin.items())},
    'med_map': {f'{k[0]}|{k[1]}': v for k, v in sorted(med.items())},
}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('T_pct','bootstrap','committed_in_sample_p90','committed_median_res','n_bins_cov','n_cells_total','n_cells_covered','n_cells_excluded_bin_lt3')}, indent=1))
