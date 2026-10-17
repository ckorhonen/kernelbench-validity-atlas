"""E1b: order/clock-state instrument test. The 2x cross-run gap (calib 14.4ms vs E1 7.1ms,
same op+size) must be explained before any boundary is trusted.

Protocol: reduced sane-config set; sizes 2^14..2^28 step 0.5; two orderings (asc, seeded-shuffled);
per-work-unit clock sampling (nvidia-smi, untimed); anchor config (torch @ 2^28) measured at
start, middle, end of EACH ordering -> within-run drift signature + order-effect size.
"""
import json, os, time, subprocess, random
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e1b_order.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
FLUSH_BYTES = 256 * 2**20

def smi(q="clocks.sm,temperature.gpu,power.instant"):
    try:
        return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True, timeout=5).strip()
    except Exception as e:
        return f"ERR"

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
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
    return {"med_ms": ts[n // 2], "iqr_ms": [round(ts[int(0.25 * n)], 5), round(ts[min(n - 1, int(0.75 * n))], 5)], "n_reps": n}

def main():
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)

    sizes, k = [], 14.0
    while k <= 28.0:
        sizes.append(int(2 ** k)); k += 0.5
    # sane configs only (per E1 analysis)
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

    # preallocate max buffers, slice views per n (contiguous rows)
    maxn = 2 ** 28
    A = torch.randn(maxn, device=dev); B = torch.randn(maxn, device=dev); C = torch.empty_like(A)

    def work_units(order):
        units = []
        for n in sizes:
            for kind, cfg in cfgs:
                units.append((n, kind, cfg))
        if order == "asc":
            pass
        else:
            random.Random(42).shuffle(units)
        # anchors: torch@2^28 at start/mid/end
        anchor = (ANCHOR_N, "torch", {})
        units = [anchor] + units + [anchor, ("mid", None, None)] + []
        return units

    results = {"meta": {"date": DATE, "device": props.name, "sm_count": sm_count,
                        "sizes": len(sizes), "cfg_count": len(cfgs)}, "runs": {}}
    for order in ("asc", "shuffled"):
        rows = []
        units = work_units(order)
        half = len(units) // 2
        for idx, (n, kind, cfg) in enumerate(units):
            if n == "mid":  # mid anchor
                n = ANCHOR_N; kind = "anchor_mid"; cfg = {}
            nn = ANCHOR_N if kind == "anchor_mid" else n
            a, b, c = A[:nn], B[:nn], C[:nn]
            op = make_op("torch" if "anchor" in kind else kind, cfg if "anchor" not in kind else {}, a, b, c, nn)
            ck0 = smi()
            r = evt_time(op, warmup=3, reps=15, flush=flush)
            ck1 = smi()
            bytes_moved = 12 * nn
            rows.append({"phase": "anchor" if ("anchor" in kind or idx in (0, len(units) - 1)) else "main",
                         "anchor_kind": kind if "anchor" in kind else ("torch_end" if idx == len(units) - 1 else ("torch_start" if idx == 0 else "")),
                         "order_idx": idx, "n": nn, "kind": kind, "cfg": cfg,
                         "med_ms": r["med_ms"], "iqr_ms": r["iqr_ms"],
                         "GBps": round(bytes_moved / (r["med_ms"] / 1e3) / 1e9, 1),
                         "clocks": [ck0, ck1]})
        results["runs"][order] = rows

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    # quick console: anchor stability per run
    for order, rows in results["runs"].items():
        anch = [r for r in rows if r["phase"] == "anchor"]
        print(order, "anchors:", [(r["anchor_kind"], r["med_ms"], r["clocks"][0].split(",")[0]) for r in anch])
    print("wrote", OUT)

if __name__ == "__main__":
    main()
