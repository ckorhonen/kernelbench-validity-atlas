"""E1b v2: order/clock-state instrument test. Debt fixes vs v1:
  - smi query power.instant -> power.draw (v1 returned ERR rows)
  - explicit unit roles; v1 misfiled the end anchor as phase=main
    (v1 data recovered in results/e1b_order_v2.json by e1_recover.py)
Protocol unchanged otherwise: sane configs; sizes 2^14..2^28 step .5;
orderings asc / seeded-shuffle(42); anchors torch@2^28 start+end+mid;
per-unit clock/power sampling before+after (untimed).
"""
import json, os, time, subprocess, random
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e1b_order_v2.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20

def smi(q="clocks.sm,temperature.gpu,power.draw,clocks.mem"):
    try:
        return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                                       text=True, timeout=5).strip()
    except Exception:
        return "ERR"

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)

@triton.jit
def add_pers(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    num = tl.num_programs(0)
    for i in range(pid, tl.cdiv(n, BLOCK), num):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(o_ptr + offs, x + y, mask=mask)

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
    return {"med_ms": ts[n // 2],
            "iqr_ms": [round(ts[int(0.25 * n)], 5), round(ts[min(n - 1, int(0.75 * n))], 5)], "n_reps": n}

def main():
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)

    sizes, k = [], 14.0
    while k <= 28.0:
        sizes.append(int(2 ** k)); k += 0.5
    cfgs = [("torch", {}), ("flat", {"BLOCK": 1024, "num_warps": 4}),
            ("flat", {"BLOCK": 4096, "num_warps": 4}),
            ("flat", {"BLOCK": 256, "num_warps": 2}),
            ("pers", {"BLOCK": 4096, "cap": 4 * sm_count, "num_warps": 4})]
    ANCHOR_N = 2 ** 28

    def make_op(kind, cfg, a, b, c, n):
        if kind == "torch":
            return lambda: torch.add(a, b, out=c)
        grid = (triton.cdiv(n, cfg["BLOCK"]),)
        if kind == "flat":
            return lambda: add_kernel[grid](a, b, c, n, BLOCK=cfg["BLOCK"], num_warps=cfg["num_warps"])
        g = (min(triton.cdiv(n, cfg["BLOCK"]), cfg["cap"]),)
        return lambda: add_pers[g](a, b, c, n, BLOCK=cfg["BLOCK"], NPROG=g[0], num_warps=cfg["num_warps"])

    maxn = 2 ** 28
    A = torch.randn(maxn, device=dev); B = torch.randn(maxn, device=dev); C = torch.empty_like(A)

    def work_units(order):
        units = []
        for n in sizes:
            for kind, cfg in cfgs:
                units.append(("main", n, kind, dict(cfg)))
        if order == "shuffled":
            random.Random(42).shuffle(units)
        anchor_start = ("anchor_start", ANCHOR_N, "torch", {})
        anchor_end = ("anchor_end", ANCHOR_N, "torch", {})
        anchor_mid = ("anchor_mid", ANCHOR_N, "torch", {})
        return [anchor_start] + units + [anchor_end, anchor_mid]

    results = {"meta": {"date": DATE, "device": props.name, "sm_count": sm_count,
                        "sizes": len(sizes), "cfg_count": len(cfgs), "version": 2}, "runs": {}}
    for order in ("asc", "shuffled"):
        rows = []
        units = work_units(order)
        for idx, (role, n, kind, cfg) in enumerate(units):
            a, b, c = A[:n], B[:n], C[:n]
            op = make_op(kind, cfg, a, b, c, n)
            ck0 = smi()
            r = evt_time(op, warmup=3, reps=15, flush=flush)
            ck1 = smi()
            rows.append({"role": role, "phase": "anchor" if role != "main" else "main",
                         "anchor_kind": role if role != "main" else "",
                         "order_idx": idx, "n": n, "kind": kind, "cfg": cfg,
                         "med_ms": r["med_ms"], "iqr_ms": r["iqr_ms"],
                         "GBps": round(12 * n / (r["med_ms"] / 1e3) / 1e9, 1),
                         "clocks": [ck0, ck1]})
        results["runs"][order] = rows

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    for order, rows in results["runs"].items():
        anch = [r for r in rows if r["role"].startswith("anchor")]
        print(order, "anchors:", [(r["role"], r["med_ms"], r["clocks"][0]) for r in anch])
    print("wrote", OUT)

if __name__ == "__main__":
    main()
