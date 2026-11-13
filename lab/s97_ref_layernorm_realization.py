#!/usr/bin/env python3
"""s97: E1 of the exploitation plan — is the reference LayerNorm realization stable across draws?

Records, per shape and per draw, the realized CUDA kernel set (names+counts) from torch.profiler,
the launch geometry proxy (kernel name carries vectorization/block hints), output checksums, and a
contended-GPU timing (no speed claim). Preregistered in lab/s97_prereg.md.
"""
import json, os, sys, statistics, hashlib
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SHAPES = [
    ("official_L1_40", (16, 64, 256, 256)),
    ("reduced_w524k", (16, 32, 128, 128)),
    ("reduced_w65k", (16, 16, 64, 64)),
]
DRAWS = 8
OUT = os.environ.get("S97_OUT", "results/s97_ref_layernorm_realization.jsonl")

rows = []
for name, shape in SHAPES:
    rows_per = shape[1] * shape[2] * shape[3]
    for d in range(DRAWS):
        torch.manual_seed(9700 + d)
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        w = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
        b = torch.randn(shape[1:], device="cuda", dtype=torch.float32)
        # warmup
        for _ in range(3):
            y = F.layer_norm(x, shape[1:], w, b, 1e-5)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            y = F.layer_norm(x, shape[1:], w, b, 1e-5)
            torch.cuda.synchronize()
        evts = {}
        for e in prof.key_averages():
            if e.device_type == torch.autograd.DeviceType.CUDA or (e.self_device_time_total > 0):
                if e.key not in ("cudaLaunchKernel",):
                    evts[e.key] = {"count": int(e.count), "cuda_us": round(float(e.self_device_time_total), 2)}
        # timing: contended GPU, labelled, no claim
        times = []
        for _ in range(20):
            torch.cuda.synchronize(); t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
            t0.record(); y = F.layer_norm(x, shape[1:], w, b, 1e-5); t1.record(); torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        med = statistics.median(times)
        rows.append({
            "shape_name": name, "shape": list(shape), "elements_per_row": rows_per,
            "draw": d, "seed": 9700 + d,
            "kernels": evts,
            "n_distinct_kernels": len(evts),
            "out_checksum": hashlib.sha256(y.detach().cpu().numpy().tobytes()).hexdigest()[:16],
            "out_mean": float(y.mean()), "out_std": float(y.std()),
            "median_ms": round(med, 4), "min_ms": round(min(times), 4), "max_ms": round(max(times), 4),
            "gpu_contended": True,
        })
        print(f"{name} draw {d}: kernels={sorted(evts)} med_ms={med:.3f}", flush=True)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# analysis: kernel-set stability per shape
summary = {"shapes": {}, "env": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                                 "cc": torch.cuda.get_device_capability(0)}}
for name, shape in SHAPES:
    rs = [r for r in rows if r["shape_name"] == name]
    sets = [tuple(sorted(r["kernels"].keys())) for r in rs]
    summary["shapes"][name] = {
        "n_draws": len(rs),
        "distinct_kernel_sets": len(set(sets)),
        "kernel_sets": [list(s) for s in sorted(set(sets))],
        "kernels_per_draw": [len(s) for s in sets],
        "out_checksum_distinct": len(set(r["out_checksum"] for r in rs)),
        "median_ms_range": [min(r["median_ms"] for r in rs), max(r["median_ms"] for r in rs)],
    }
    print(name, json.dumps(summary["shapes"][name])[:400], flush=True)

verdict = "REFERENCE_STABLE" if all(v["distinct_kernel_sets"] == 1 for v in summary["shapes"].values()) else "REFERENCE_UNSTABLE"
summary["verdict"] = verdict
with open(os.path.splitext(OUT)[0] + "_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("VERDICT", verdict)
