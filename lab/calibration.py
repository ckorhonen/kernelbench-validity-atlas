"""W1.1 GB10 box calibration. Every number the atlas will use, measured not assumed.

Outputs results/calibration.json with provenance (date, host, torch, driver, util at start/end).
Sections: device props, clock lock attempt, launch overhead (single + back-to-back),
D2D copy bandwidth ladder (fp32/fp16, L2-flushed), triad bandwidth, drift probe.
"""
import json, subprocess, time, os, sys
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "calibration.json")
FLUSH_BYTES = 256 * 2**20  # > L2 on any plausible config

def evt_time(fn, warmup=5, reps=25, flush=None):
    """Median cuda-event time of fn() in ms, L2-flush optional before each rep."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    med = ts[n // 2]
    lo, hi = ts[int(0.25 * n)], ts[min(n - 1, int(0.75 * n))]
    return {"med_ms": med, "iqr_ms": [lo, hi], "n_reps": n}

def main():
    dev = torch.device("cuda:0")
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    cal = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": props.name, "cc": f"{props.major}.{props.minor}",
           "sm_count": props.multi_processor_count}
    for attr in ("L2_cache_size", "total_memory"):
        cal[attr] = getattr(props, attr, None)
    cal["total_memory_GiB"] = round(props.total_memory / 2**30, 1)

    def smi(q):
        try:
            return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).strip()
        except Exception as ex:
            return f"ERR {ex}"
    cal["clocks_at_start"] = smi("clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu,power.draw")

    # clock lock attempt (record-only; drift protocol adapts if unavailable)
    lock = subprocess.run(["nvidia-smi", "-lgc", "1500,1500"], capture_output=True, text=True)
    cal["clock_lock"] = {"attempted": "1500", "ok": lock.returncode == 0,
                         "msg": (lock.stdout + lock.stderr).strip()[:200]}
    cal["clocks_after_lock"] = smi("clocks.sm,clocks.min.sm,clocks.max.sm")

    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)

    # launch overhead: single-shot and back-to-back, minimal kernel (1-elem fill)
    tiny = torch.zeros(1, device=dev)
    single = evt_time(lambda: tiny.fill_(1.0), warmup=20, reps=200, flush=None)
    # back-to-back: N launches inside one event pair
    def b2b(n=1000):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        tiny.fill_(0.5)
        torch.cuda.synchronize(); s.record()
        for _ in range(n): tiny.fill_(0.5)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / n
    for _ in range(3): b2b(100)  # warm
    cal["launch_overhead"] = {"single_shot_med_ms": single["med_ms"], "back_to_back_ms": b2b()}

    # bandwidth ladder: contiguous D2D copy, fp32 and fp16, same byte sizes
    ladder = {}
    for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        rows = []
        for nbytes in [2**k for k in range(10, 29)]:
            esize = 4 if dt == torch.float32 else 2
            n = max(1, nbytes // esize)
            a = torch.empty(n, dtype=dt, device=dev); b = torch.empty_like(a)
            a.uniform_()
            r = evt_time(lambda: b.copy_(a), warmup=5, reps=25, flush=flush)
            gbs = 2 * n * esize / (r["med_ms"] / 1e3) / 1e9  # read+write
            rows.append({"bytes": n * esize, "med_ms": round(r["med_ms"], 4),
                         "iqr_ms": [round(x, 4) for x in r["iqr_ms"]], "GBps": round(gbs, 1)})
        ladder[name] = rows
    cal["copy_bandwidth_ladder"] = ladder

    # triad c = a + b (2r+1w) at large size => effective bandwidth
    for n in (2**28,):
        a = torch.randn(n, device=dev); b = torch.randn(n, device=dev); c = torch.empty_like(a)
        r = evt_time(lambda: torch.add(a, b, out=c), warmup=5, reps=25, flush=flush)
        cal["triad_fp32_1GiB"] = {"bytes_moved": 3 * 4 * n, "med_ms": r["med_ms"],
                                  "GBps": round(3 * 4 * n / (r["med_ms"] / 1e3) / 1e9, 1),
                                  "iqr_ms": [round(x, 4) for x in r["iqr_ms"]]}

    # drift probe: same 64MiB copy, 5 probes over ~50s
    a = torch.empty(16 * 2**20, dtype=torch.float32, device=dev); b = torch.empty_like(a)
    drift = []
    for i in range(5):
        r = evt_time(lambda: b.copy_(a), warmup=3, reps=15, flush=flush)
        drift.append({"t": time.strftime("%H:%M:%S"), "med_ms": round(r["med_ms"], 4),
                      "clocks": smi("clocks.sm,temperature.gpu,utilization.gpu")})
        if i < 4: time.sleep(10)
    cal["drift_probe_64MiB_copy"] = drift
    cal["clocks_at_end"] = smi("clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu,power.draw")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(cal, f, indent=1)
    print("wrote", OUT)
    print(json.dumps({k: cal[k] for k in ("device", "sm_count", "L2_cache_size", "launch_overhead", "clock_lock")}, indent=1)[:800])

if __name__ == "__main__":
    main()
