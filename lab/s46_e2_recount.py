"""s46_e2_recount.py — canonical E2-v3 win count for paper section 3.2.

RULE (frozen before run, 2026-10-23, CPU, records only — no GPU):
  Source: results/e2_splitk_v3.json rows (60 (K,M) cells, 12 reps, med/iqr per arm).
  Hand arms: every variant whose name starts 'tsplit'. Reference: 'torch' arm.
  BEAT  := min over hand arms of iqr_hi  <  torch iqr_lo   (strict combined-IQR
           separation; the diary-3 'beyond combined-IQR noise' rule, made explicit)
  TIE   := intervals overlap but chosen-hand med < torch med
  LOSE  := otherwise (chosen hand = min-med hand arm overall)
  Chosen hand per cell: min med_ms among BEATING arms if any beat, else min med overall.
  margin := torch.med/chosen.med - 1 (pp), reported for beats.
  Also recorded per cell: chosen (S,KT), total CTAs = ceil(M/BM)*S (BM from meta),
  waves = CTAs/48 — to test the 'wins live in 32..128 CTAs' band claim, and the M<=8
  stratum, and torch's own margin when torch wins.

Emits results/s46_e2_recount.json. No kernel is run; timings are as-recorded.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'results', 'e2_splitk_v3.json')
OUT = os.path.join(HERE, 'results', 's46_e2_recount.json')

d = json.load(open(SRC))
BM = d['meta']['BM']
rows_out = []
counts = {'beat': 0, 'tie': 0, 'lose': 0}
for r in d['variants'] if False else d['rows']:
    K, M = r['K'], r['M']
    t = r['variants']['torch']
    tlo, thi = sorted(t['iqr_ms'])[0], sorted(t['iqr_ms'])[-1]
    hands = {n: v for n, v in r['variants'].items() if n.startswith('tsplit')}
    beats = {n: v for n, v in hands.items() if sorted(v['iqr_ms'])[-1] < tlo}
    pool = beats if beats else hands
    best_n = min(pool, key=lambda n: pool[n]['med_ms'])
    best = pool[best_n]
    blo, bhi = sorted(best['iqr_ms'])[0], sorted(best['iqr_ms'])[-1]
    if beats:
        cls = 'beat'
    elif best['med_ms'] < t['med_ms'] and bhi >= tlo:
        cls = 'tie'
    else:
        cls = 'lose'
    counts[cls] += 1
    S = int(best_n.split('_S')[1].split('_')[0])
    KT = int(best_n.split('KT')[1])
    ctas = math.ceil(M / BM) * S
    rows_out.append({
        'K': K, 'M': 2 * M if False else M, 'class": ' if False else 'class': cls,
        'hand': best_n, 'S': S, 'KT': KT, 'ctas': ctas, 'waves': round(ctas / 48, 2),
        'hand_med_ms': best['med_ms'], 'torch_med_ms': t['med_ms'],
        'margin_pp': round((t['med_ms'] / best['med_ms'] - 1) * 100, 2) if cls == 'beat' else None,
    })

out = {
    'meta': {
        'date': '2026-10-23 (records-only recount)',
        'source': 'results/e2_splitk_v3.json',
        'rule': __doc__.split('RULE', 1)[1].split('Emits')[0].strip(),
        'BM': BM,
    },
    'counts': counts,
    'beats_max_margin_pp': max((r['margin_pp'] for r in rows_out if r['margin_pp'] is not None), default=None),
    'beats_margin_pp_min': min((r['margin_pp'] for r in rows_out if r['margin_pp'] is not None), default=None),
    'beat_ctas_range': [min((r['ctas'] for r in rows_out if r['class'] == 'beat'), default=None),
                        max((r['ctas'] for r in rows_out if r['class'] == 'beat'), default=None)],
    'lose_ctas_range': [min((r['ctas'] for r in rows_out if r['class'] == 'lose'), default=None),
                        max((r['ctas'] for r in rows_out if r['class'] == 'lose'), default=None)],
    'm_le_8': {c: sum(1 for r in rows_out if r['M'] <= 8 and r['class'] == c) for c in counts},
    'rows': rows_out,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
