"""e3: grid-geometry mechanism — why does saturated BW depend on grid geometry?

Pre-registered 2026-10-18 ~16:55Z BEFORE any run. Context (DIARY 10-17 session 2, E1 v3):
flat BLOCK=256 exceeds the copy ladder (246.5 vs ~220 GB/s) while persistent kernels sit at
copy parity; falsifiable form was "persistent vs flat same BLOCK same n differ >5% saturated".
Question: WHICH mechanism? Candidates:
  (a) address phase / CTA->data mapping (mem-controller distribution),
  (b) resident-wave structure (occupancy / wave quantization),
  (c) clock state differences across launch geometries.

Arms — identical kernel body, same BLOCK, same n, same bytes/thread:
  flat        grid=NB,  block pid            (the anomalous geometry)
  flat_perm   grid=NB,  block (pid*5)%NB     same geometry, permuted mapping -> tests (a) alone
  pers{W}     grid=48*W persistent strided  W in {1,2,4,8}            -> tests (b) dose-response
  pers192_perm W=2 with permuted order                              -> (a) inside persistent
Pre-registered reads:
  R1 (a): flat_perm vs flat saturated-BW gap > 3% AND > pooled IQR => mapping matters.
  R2 (b): if BW(pers) rises monotonically with W toward flat and closes to <3% at some W
          while flat==flat_perm, geometry effect is occupancy, not mapping.
  R3 (c): per-arm sm/mem clocks recorded (pynvml if present); >2% cross-arm clock spread
          with BW rank-tracking clocks => clock confound; else ruled out for the recorded run.
Decision: mechanism = the R whose intervention moves >= half of the flat-vs-persW=1 gap,
replicated at both n. No re-running after seeing the verdict clause above.
Schema P2: op_class, rule, domain predicate, derived coords, variance, n_reps, date per row.
"""
import json, os, time, sys
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e3_gridgeom.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20

@triton.jit
def copy_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, PERM: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    src = pid
    if PERM:
        src = (pid * P) % tl.num_programs(0)
    offs = src * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, a + b, mask=mask)

@triton.jit
def copy_persistent(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, PERM: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    num = tl.num_programs(0)
    NB = tl.cdiv(n, BLOCK)
    for i in range(pid, NB, num):
        src = i
        if PERM:
            src = (i * P) % NB
        offs = src * BLOCK + tl.arange(0, BLOCK)
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
        parts = [p.strip() for p in out.split(",")]
        def num(i):
            try: return float(parts[i].replace("[N/A]", "nan").replace("N/A", "nan"))
            except Exception: return None
        return {"sm_mhz": num(0), "mem_mhz": num(1), "util_pct": num(2), "watts": num(3)}
    except Exception as ex:
        return {"error": str(ex)}

def main():
    dev = "cuda"
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    x = torch.randn(2**28, device=dev); y = torch.randn_like(x); o = torch.empty_like(x)

    rows = []
    def run_arm(tag, kind, BLOCK, n_el, W=None, perm=False):
        nb = n_el // BLOCK
        if kind == "flat":
            go = lambda: copy_kernel[(nb,)](x, y, o, n_el, BLOCK=BLOCK, PERM=perm, P=5, num_warps=4)
        else:
            go = lambda: copy_persistent[(sm_count * W,)](x, y, o, n_el, BLOCK=BLOCK, PERM=perm, P=5, num_warps=4)
        r = evt_time(go, flush=flush)
        clk_pre = clocks()
        bytes_moved = n_el * 4 * 3
        gbps = bytes_moved / (r["med_ms"] / 1e3) / 1e9
        clk_post = clocks()
        row = {"arm": tag, "kind": kind, "BLOCK": BLOCK, "n": n_el, "grid": nb if kind == "flat" else sm_count * W,
               "waves_flat": nb / sm_count, "perm": perm, "bytes_moved": bytes_moved,
               "gbps": gbps, **r, "clk_pre": clk_pre, "clk_post": clk_post,
               "op_class": "elementwise_copy", "rule": "none",
               "domain_predicate": f"n={n_el},BLOCK={BLOCK},num_warps=4,exact multiple grid",
               "derived_coords": {"bytes_per_thread": 4 * BLOCK / 32 / 4, "ctas": nb if kind == 'flat' else sm_count * W,
                                   "waves_total": (nb if kind == 'flat' else sm_count * W) / sm_count},
               "date": DATE}
        rows.append(row); print(f"{tag:16s} B={BLOCK:5d} n=2^{n_el.bit_length()-1:2d} {gbps:7.1f} GB/s iqr {r['iqr_ms']:.3f}ms", flush=True)

    for BLOCK in (256, 1024):
        for n_el in (100663296, 50331648):   # both divisible by 48*BLOCK (exact waves, no tails)
            assert n_el % (sm_count * BLOCK) == 0
            run_arm("flat", "flat", BLOCK, n_el)
            run_arm("flat_perm", "flat", BLOCK, n_el, perm=True)
            for W in (1, 2, 4, 8):
                run_arm(f"pers{sm_count*W}", "pers", BLOCK, n_el, W=W)
            run_arm("pers192_perm", "pers", BLOCK, n_el, W=2, perm=True)

    json.dump({"meta": {"date": DATE, "sm_count": sm_count, "torch": torch.__version__,
                        "device": props.name, "prereg": "see module docstring, written pre-run"},
               "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
