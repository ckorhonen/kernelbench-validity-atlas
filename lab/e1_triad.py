"""E1: elementwise triad boundary — the instrument's first real measurement.

Question: does the latency->bandwidth transition of a trivial elementwise kernel
collapse in derived coordinates (fixed-overhead x bandwidth product), with the
fixed overhead anchored by INDEPENDENT calibration (launch overhead, copy BW),
not fitted from the same data?

Design: c = a + b over log-lattice sizes; Triton kernels across BLOCK/num_warps/
persistent-grid variants + torch add baseline. Per row: timing + derived coords.
Analysis (same run): per-config two-parameter fit t(B) = c0 + B/BW_eff on large-B
region; compare c0 to calibration launch overhead; knee B* = c0 * BW_eff vs measured
transition; verdict on collapse.

Results schema (P2): every row carries op_class, rule, derived coords, variance, n_reps, date.
"""
import json, os, time, sys
import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, "results", "calibration.json")))
OUT = os.path.join(HERE, "results", "e1_triad.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

FLUSH_BYTES = 256 * 2**20

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)

@triton.jit
def add_persistent(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    num = tl.num_programs(0)
    for i in range(pid, tl.cdiv(n, BLOCK), num):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(o_ptr + offs, x + y, mask=mask)

def evt_time(fn, warmup=5, reps=25, flush=None):
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
    iqr = [ts[int(0.25 * n)], ts[min(n - 1, int(0.75 * n))]]
    return {"med_ms": ts[n // 2], "iqr_ms": iqr, "n_reps": n}

def run():
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)

    sizes = []
    k = 10.0
    while k <= 28.0:
        sizes.append(int(2 ** k)); k += 0.5
    configs = []
    for BLOCK in (256, 1024, 4096, 16384):
        for warps in (1, 2, 4, 8):
            if BLOCK // 32 < warps:  # warps can't exceed block threads
                continue
            configs.append({"kind": "flat", "BLOCK": BLOCK, "num_warps": warps})
    for BLOCK in (1024, 4096):
        for cap_mult in (1, 4):
            configs.append({"kind": "persistent", "BLOCK": BLOCK, "cap": cap_mult * sm_count, "num_warps": 4})

    rows = []
    for n in sizes:
        a = torch.randn(n, device=dev); b = torch.randn(n, device=dev); c = torch.empty_like(a)
        bytes_moved = 3 * 4 * n
        # torch baseline
        r = evt_time(lambda: torch.add(a, b, out=c), flush=flush)
        rows.append(row_dict("torch_add", {}, n, bytes_moved, r, sm_count))
        for cfg in configs:
            grid = (triton.cdiv(n, cfg["BLOCK"]),) if cfg["kind"] == "flat" else (min(triton.cdiv(n, cfg["BLOCK"]), cfg["cap"]),)
            def go():
                if cfg["kind"] == "flat":
                    add_kernel[grid](a, b, c, n, BLOCK=cfg["BLOCK"], num_warps=cfg["num_warps"])
                else:
                    add_persistent[grid](a, b, c, n, BLOCK=cfg["BLOCK"], NPROG=grid[0], num_warps=cfg["num_warps"])
            try:
                r = evt_time(go, warmup=3, reps=15, flush=flush)
            except Exception as ex:
                rows.append(row_err(cfg, n, ex)); continue
            rows.append(row_dict(f"triton_{cfg['kind']}", {k2: v2 for k2, v2 in cfg.items()}, n, bytes_moved, r, sm_count, grid[0]))
        del a, b, c
    meta = {"date": DATE, "device": props.name, "sm_count": sm_count,
            "calibration_launch_single_ms": CAL["launch_overhead"]["single_shot_med_ms"],
            "calibration_launch_b2b_ms": CAL["launch_overhead"]["back_to_back_ms"],
            "note": "c0 anchor from calibration, not fitted; see analysis"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, len(rows), "rows")

def row_dict(kind, cfg, n, bytes_moved, r, sm_count, grid=None):
    t = r["med_ms"] / 1e3
    d = {"op_class": "elementwise_triad", "rule": kind, "cfg": cfg,
         "n": n, "bytes_moved": bytes_moved,
         "med_ms": round(r["med_ms"], 5), "iqr_ms": [round(x, 5) for x in r["iqr_ms"]],
         "n_reps": r["n_reps"], "date": r.get("date", None) or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "GBps": round(bytes_moved / t / 1e9, 1)}
    if grid is not None:
        d["derived"] = {"grid": grid, "nominal_waves": round(grid / sm_count, 3),
                        "threads": None}
    return d

def row_err(cfg, n, ex):
    return {"op_class": "elementwise_triad", "rule": "triton_%s" % cfg["kind"], "cfg": cfg,
            "n": n, "error": str(ex)[:200], "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

if __name__ == "__main__":
    run()
