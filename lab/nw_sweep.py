"""nw sweep: the num_warps residue, one-variable test under controlled geometry.

Pre-registered 2026-10-19T00:58Z BEFORE any run (reflection #13 decision, DIARY session 16).
Context: e1v3 flat B256 rows showed 238-280 GB/s (nw8 med 250.4) -- the "copy-ladder excess
246.5"; e3 controlled geometry at nw=4 gave flat B256 226.7/225.7 (two n) -- excess NOT
reproduced at nw4. e3 ruled out mapping (~4%) and clocks (pinned); nw is the live hypothesis.

Arms (identical kernels to e3_gridgeom.py, only num_warps varies; flush-charged, 15 reps, med):
  flat B256, n in {100663296, 50331648} (exact-wave), nw in {4(anchor), 8, 16}
  pers192 B256 (the >=2-wave plateau arm), same n, nw in {4, 8, 16}
Pre-registered reads:
  P1 (primary): flat B256 nw8 med >= 240 GB/s at n=100663296 AND same direction at 50331648
      => the reference-BW anchor gains a DECLARED nw term (A1 validity-domain note).
  P2 (dose): nw16 vs nw8 monotonicity recorded, not gate-fitted.
  P3 (symmetry): if flat gains >=5% (nw8 vs nw4) and pers192 gains <2%, the nw effect is
      geometry-linked (flat-only), recorded as such.
  S (sanity): nw4 re-run within 2% of e3 (flat 226.7/225.7; pers192 215.7/215.3) else
      harness drift declared and P1 reads taken against the in-run nw4 anchor instead.
Falsifier / vacuity: nw8 < 240 GB/s at the primary point => the e1v3 excess closes as
harness-order artifact (uncontrolled geometry), anchor stays nw4. NO re-running after seeing
the verdict clauses. No gate-fitting, no new arms; A2 stays bookkeeping-parked.
Schema P2 rows: op_class, rule, domain_predicate, derived_coords, variance, n_reps, date.
"""
import json, os, time
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "nw_sweep.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20

@triton.jit
def copy_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, a + b, mask=mask)

@triton.jit
def copy_persistent(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num = tl.num_programs(0)
    NB = tl.cdiv(n, BLOCK)
    for i in range(pid, NB, num):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(x_ptr + offs, mask=mask)
        b = tl.load(y_ptr + offs, mask=mask)
        tl.store(o_ptr + offs, a + b, mask=mask)

def evt_time(fn, warmup=3, reps=15, flush=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); n = len(ts)
    lo, hi = ts[(n - 1) // 4], ts[3 * (n - 1) // 4]
    return {"med_ms": ts[n // 2], "iqr_ms": hi - lo, "n_reps": n}

def clocks():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        p = [x.strip() for x in out.split(",")]
        def num(i):
            try: return float(p[i].replace("[N/A]", "nan").replace("N/A", "nan"))
            except Exception: return None
        return {"sm_mhz": num(0), "mem_mhz": num(1), "util_pct": num(2), "watts": num(3)}
    except Exception as ex:
        return {"error": str(ex)}

def main():
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    x = torch.randn(2**27, device="cuda"); y = torch.randn_like(x); o = torch.empty_like(x)
    rows = []
    def run_arm(kind, W, nw, n_el):
        if kind == "flat":
            nb = n_el // 256
            go = lambda: copy_kernel[(nb,)](x, y, o, n_el, BLOCK=256, num_warps=nw)
            grid = nb
        else:
            grid = sm_count * W
            go = lambda: copy_persistent[(grid,)](x, y, o, n_el, BLOCK=256, num_warps=nw)
        r = evt_time(go, flush=flush)
        c1 = clocks(); c2 = clocks()
        bytes_moved = n_el * 4 * 3
        gbps = bytes_moved / (r["med_ms"] / 1e3) / 1e9
        rows.append({"arm": f"{kind}_nw{nw}", "kind": kind, "BLOCK": 256, "nw": nw, "W": W,
                     "n": n_el, "grid": grid, "bytes_moved": bytes_moved, "gbps": gbps, **r,
                     "clk_pre": c1, "clk_post": c2, "op_class": "elementwise_copy", "rule": "none",
                     "domain_predicate": f"n={n_el},BLOCK=256,nw={nw},kind={kind},exact-wave",
                     "derived_coords": {"bytes_per_thread": 2.0, "waves_total": grid / sm_count},
                     "variance_iqr_ms": r["iqr_ms"], "n_reps": r["n_reps"], "date": DATE})
        print(f"{kind:8s} nw={nw:2d} n={n_el:>9d} {gbps:7.1f} GB/s iqr {r['iqr_ms']:.4f}", flush=True)

    for n_el in (100663296, 50331648):
        for nw in (4, 8, 16):
            run_arm("flat", None, nw, n_el)
        for nw in (4, 8, 16):
            run_arm("pers", 4, nw, n_el)   # pers192 plateau arm

    json.dump({"meta": {"date": DATE, "sm_count": sm_count, "torch": torch.__version__,
                        "device": props.name, "prereg": "docstring, written 00:58Z pre-run"},
               "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
