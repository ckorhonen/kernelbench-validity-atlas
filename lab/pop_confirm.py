"""Population-collapse CONFIRMATORY runner (tick #31). Prereg: lab/pop_freeze.py
docstring + results/pop_freeze.json (frozen BEFORE this run; T=16.1783 pct).

Protocol, verbatim e1: same add_kernel (imported), evt_time warmup=3 reps=15,
flush 256MB zero_ per rep, med; cells = every committed flat (BLOCK,nw,n) from
the freeze manifest, n-asc/BLOCK-asc/nw-asc order; tensors alloc/free per n.
NO other arms, NO persistent, NO torch baseline (population is flat-only).

Verdict (frozen): PASS iff P90_fresh <= T where resid uses the FROZEN committed
bin medians, bins committed-n>=3 only. Descriptives: median/max resid, union-CV
median, excluded count. Clocks logged start/mid/end (no lock, standing practice).
"""
import json, math, os, sys, time, subprocess, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_triad as E          # module-level: loads calibration.json, defines kernels/evt_time
import torch, triton

RES = os.path.join(HERE, 'results')
FRZ = json.load(open(os.path.join(RES, 'pop_freeze.json')))
T = FRZ['T_pct']
cell_bin = {}
for k, v in FRZ['cell_bin_map'].items():
    B, nw, n = (int(x) if x.isdigit() else float(x) for x in k.split('|'))
    n = int(n) if float(n).is_integer() else n
    cell_bin[(B, nw, n)] = tuple(v.split('|')) if isinstance(v, str) else tuple(v)
med = {}
for k, v in FRZ['med_map'].items():
    a, b = k.split('|')
    med[(float(a), float(b))] = v
cells = sorted(cell_bin, key=lambda c: (c[2], c[0], c[1]))
by_n = {}
for c in cells:
    by_n.setdefault(c[2], []).append(c)

def smclock():
    r = subprocess.run('nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader',
                       shell=True, capture_output=True, text=True)
    return r.stdout.strip()

def run():
    dev = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    flush = torch.empty(E.FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    rows, t0 = [], time.time()
    print('clocks start:', smclock(), flush=True)
    for ni, (n, cs) in enumerate(sorted(by_n.items())):
        a = torch.randn(n, device=dev); b = torch.randn(n, device=dev); c = torch.empty_like(a)
        bytes_moved = 3 * 4 * n
        for (B, nw, _) in sorted(cs, key=lambda x: x[0] or 0):
            grid = (triton.cdiv(n, B),)
            def go(B=B, nw=nw, grid=grid):
                E.add_kernel[grid](a, b, c, n, BLOCK=B, num_warps=nw)
            try:
                r = E.evt_time(go, warmup=3, reps=15, flush=flush)
            except Exception as ex:
                rows.append({'op_class': 'elementwise_triada_confirm', 'BLOCK': B, 'nw': nw, 'n': n,
                             'error': str(ex)[:200]}); continue
            d = E.row_dict('triton_flat', {'kind': 'flat', 'BLOCK': B, 'num_warps': nw},
                           n, bytes_moved, r, props.multi_processor_count, grid[0])
            d['confirm_bin'] = '|'.join(map(str, cell_bin[(B, nw, n)]))
            rows.append(d)
        if ni % 8 == 0:
            print(f'n={n:.3g} ({ni+1}/{len(by_n)}) {time.time()-t0:.0f}s clocks={smclock()}', flush=True)
        del a, b, c
    print('clocks end:', smclock(), flush=True)

    # --- frozen verdict, no post-hoc anything ---
    res, excl, union = [], 0, {}
    for d in rows:
        if 'error' in d:
            continue
        k = tuple(float(x) for x in d['confirm_bin'].split('|'))
        if k not in med:
            excl += 1; continue
        m = med[k]
        r = abs(d['GBps'] - m) / m * 100
        d['resid_pct'] = round(r, 4)
        res.append(r)
        union.setdefault(k, []).extend([d['GBps']])  # union CV computed in summary load
    P90 = sorted(res)[int(.9 * len(res))] if res else None
    verdict = 'PASS' if (P90 is not None and P90 <= T) else 'FAIL'
    summary = {
        'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'verdict': verdict, 'P90_fresh_pct': round(P90, 4) if P90 else None, 'T_pct': T,
        'freeze_sha': FRZ['population_sha256'][:12], 'n_eval': len(res), 'n_excl_binlt3': excl,
        'median_res_pct': round(st.median(res), 4) if res else None,
        'max_res_pct': round(max(res), 4) if res else None,
        'prereg': 'lab/pop_freeze.py docstring; threshold frozen pre-run',
        'runtime_s': round(time.time() - t0, 1),
    }
    json.dump({'summary': summary, 'rows': rows},
              open(os.path.join(RES, 'pop_confirm.json'), 'w'), indent=1)
    with open(os.path.join(RES, 'pop_confirm.jsonl'), 'w') as f:
        for d in rows:
            f.write(json.dumps(d) + '\n')
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    run()
