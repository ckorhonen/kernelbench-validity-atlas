"""Tuple-surface measurement — per-tuple engine-class surface at fixed shape (round-11).

WHY THIS RUN EXISTS
Round-11 review finding 3 says the round-8 tuple test cannot establish "no simple boundary"
because the observed classes are already consistent with the rule "W when stride and dilation
both equal one; T otherwise", and finding 2 says the NHWC row never actually ran channels-last
convolution (the branch permuted back to contiguous NCHW before F.conv2d, so the recorded
x_strides describe an intermediate tensor, not the convolution input).

This script measures the surface instead of arguing about it, under a prereg frozen BEFORE the
first run:
  * every cell runs the tensor layout that is declared (channels_last input is passed to
    F.conv2d directly; the strides recorded are those of the tensors actually handed to conv2d,
    for both flag settings);
  * each cell repeats twice, so class determinism is measured, not assumed;
  * class, rho_max, frac_over and flag response are recorded per cell, plus the fp64 anchor's
    class (reference-realization axis);
  * the round-9 flagged cells (pad=2 at s=1; s=2,p=2) are included explicitly.

PREDICTIONS FROZEN PRE-RUN (written before execution; a prediction failing is a result, not a
reason to edit the script):
  P1 simple-rule: on NCHW fp32-default cells with s=1 AND d=1, class == "W"
     (cells A00,A05,A06,A08,A09,A10,B01,B02,B03). On cells with s!=1 or d!=1, class != "W"
     (A01,A02,A03,A04,C01,C02,C03). Violation => the simple rule is scoped, and the violated
     coordinate is named in the artifact.
  P2 layout: A07 (real channels_last input) class != "W". If A07 is W, layout is not a dispatch
     coordinate on GB10 at this shape and round-9 F2's invariance claim would hold for real.
  P3 determinism: class agrees across the two repeats in every cell. Disagreement => class is not
     a function of the declared tuple and admission must be enumerated measurement.
  P4 magnitude: A06 (pad=2, s=1, d=1) rho_max in (6.8, 20) — the round-9 reviewer's stored value
     was 12.8024, outside the manuscript's published 6.4-6.8 range; this re-measures it.

KILLS PREDECLARED: P3 failure kills "class is dispatchable from declared coordinates". P1
violation kills the unqualified "W iff s=1 and d=1" rule. Neither kill removes the two-class
result; both scope it.

CLASSIFICATION is read from torch.profiler CUDA kernel names (concrete, not inferred):
  T = tensorop *_s1688* (TF32 tensor core), W = winograd*, I = im2col/simt fp32, O = other.
"""

import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
def _kb_res_dir(here):
    """Resolve the artifact directory: lab/results in the repo, ../results when run from the
    supplement package's kernels/ directory (frozen packaging convention, round-11 F6)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "tuple_surface.json")

ATOL = RTOL = 1e-4
dev = "cuda"
SEED = 4321
BASE = {"C": 256, "k": 3, "H": 7, "W": 7, "batch": 2, "stride": 1, "pad": 1, "dilation": 1,
        "layout": "NCHW"}

# Frozen cell list (id, overrides). Frozen pre-run.
CELLS = [
    ("A00", {}),                                  # base
    ("A01", {"stride": 2}),
    ("A02", {"stride": 3}),
    ("A03", {"dilation": 2}),
    ("A04", {"dilation": 3}),
    ("A05", {"pad": 0}),
    ("A06", {"pad": 2}),                          # round-9 flagged: pad-2 winograd row
    ("A07", {"layout": "NHWC"}),                  # real channels_last input
    ("A08", {"batch": 1}),
    ("A09", {"batch": 4}),
    ("A10", {"batch": 8}),
    ("B01", {"C": 16}),
    ("B02", {"C": 64}),
    ("B03", {"H": 14, "W": 14}),
    ("C01", {"stride": 2, "dilation": 2}),
    ("C02", {"stride": 1, "pad": 2, "dilation": 2}),   # round-9 flagged: stride-2-pad-2 TF32 row
    ("C03", {"stride": 2, "pad": 2}),
]

PRED_W = ["A00", "A05", "A06", "A08", "A09", "A10", "B01", "B02", "B03"]
PRED_NOT_W = ["A01", "A02", "A03", "A04", "C01", "C02", "C03"]


def stats(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    return {"rho_max": float(r.max()), "med": float(r.median()),
            "frac_over": float((r > 1).double().mean()), "n_elem": int(r.numel()),
            "abs_max": float(d.max())}


def kernel_names(fn, warm=1, iters=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    names = {}
    for e in prof.key_averages():
        names[e.key] = names.get(e.key, 0) + int(e.count)
    return names


def engine_class(kn):
    for n in kn:
        if "tensorop" in n:
            return "T"
    for n in kn:
        if "winograd" in n:
            return "W"
    for n in kn:
        if "simt" in n or "im2col" in n or "sgemm" in n:
            return "I"
    return "O"


def expected_h(p, s, d, H, k):
    return (H + 2 * p - d * (k - 1) - 1) // s + 1


def make_inputs(cfg):
    torch.manual_seed(SEED)
    x = torch.randn(cfg["batch"], cfg["C"], cfg["H"], cfg["W"], device=dev)
    w = torch.randn(cfg["C"], cfg["C"], cfg["k"], cfg["k"], device=dev)
    if cfg["layout"] == "NHWC":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    return x, w


def run_cell(cid, cfg, rep):
    x, w = make_inputs(cfg)
    args = dict(padding=cfg["pad"], stride=cfg["stride"], dilation=cfg["dilation"])
    out_h = expected_h(cfg["pad"], cfg["stride"], cfg["dilation"], cfg["H"], cfg["k"])
    row = {"cell": cid, "rep": rep, "cfg": dict(cfg), "expected_out_hw": out_h}

    o = F.conv2d(x, w, **args)
    assert list(o.shape[-2:]) == [out_h, out_h], f"{cid} out shape {list(o.shape[-2:])} != {out_h}"

    # strides of the tensors ACTUALLY passed to conv2d (this is round-11 F2's requirement)
    row["input_strides_passed"] = [int(v) for v in x.stride()]
    row["weight_strides_passed"] = [int(v) for v in w.stride()]
    row["input_is_channels_last"] = bool(x.is_contiguous(memory_format=torch.channels_last))
    row["output_strides"] = [int(v) for v in o.stride()]

    kn = kernel_names(lambda: F.conv2d(x, w, **args))
    row["class"] = engine_class(kn)
    row["kernels"] = dict(kn)

    # fp64 anchor, same layout and same parameters
    x64, w64 = x.double(), w.double()
    o64 = F.conv2d(x64, w64, **args)
    row["rho"] = stats(o, o64)
    kn64 = kernel_names(lambda: F.conv2d(x64, w64, **args))
    row["class_fp64_anchor"] = engine_class(kn64)
    row["kernels_fp64_anchor"] = dict(kn64)

    # paired precision-flag arm on the SAME declared input layout
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        o_off = F.conv2d(x, w, **args)
        o64_off = F.conv2d(x64, w64, **args)
        row["rho_flagoff"] = stats(o_off, o64_off)
        row["flagoff_bit_equal"] = bool(torch.equal(o, o_off))
        row["flagoff_rho_ratio"] = row["rho_flagoff"]["rho_max"] / max(row["rho"]["rho_max"], 1e-30)
        kn_off = kernel_names(lambda: F.conv2d(x, w, **args))
        row["class_flagoff"] = engine_class(kn_off)
        row["kernels_flagoff"] = dict(kn_off)
    finally:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    del x, w, o, x64, w64, o64
    torch.cuda.empty_cache()
    return row


def main():
    rows = []
    for cid, ov in CELLS:
        cfg = dict(BASE)
        cfg.update(ov)
        for rep in (1, 2):
            rows.append(run_cell(cid, cfg, rep))
        print(f"{cid}: class={rows[-1]['class']} rep2={rows[-1]['class']} "
              f"rho={rows[-1]['rho']['rho_max']:.4f} frac_over={rows[-1]['rho']['frac_over']:.4f} "
              f"flagoff={rows[-1]['rho_flagoff']['rho_max']:.4f} "
              f"bit_eq={rows[-1]['flagoff_bit_equal']} fp64={rows[-1]['class_fp64_anchor']} "
              f"in_strides={rows[-1]['input_strides_passed']}", flush=True)

    # frozen reads
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    det = {cid: (len({r["class"] for r in rs}) == 1) for cid, rs in by_cell.items()}
    cls = {cid: rs[0]["class"] for cid, rs in by_cell.items()}
    p1_viol = [cid for cid in PRED_W if cls[cid] != "W"] + [cid for cid in PRED_NOT_W if cls[cid] == "W"]
    verdict = {
        "P1_simple_rule_holds": len(p1_viol) == 0,
        "P1_violations": p1_viol,
        "P2_layout_A07_class": cls["A07"],
        "P2_layout_is_dispatch_coord": cls["A07"] != "W",
        "P3_determinism_all_cells": all(det.values()),
        "P3_disagreements": [c for c, ok in det.items() if not ok],
        "P4_A06_rho_max": by_cell["A06"][0]["rho"]["rho_max"],
        "P4_in_6.8_20": 6.8 < by_cell["A06"][0]["rho"]["rho_max"] < 20,
        "class_by_cell": cls,
        "fp64_class_by_cell": {cid: rs[0]["class_fp64_anchor"] for cid, rs in by_cell.items()},
        "flagoff_class_by_cell": {cid: rs[0]["class_flagoff"] for cid, rs in by_cell.items()},
    }
    out = {"prereg": "tuple_surface.py docstring (frozen pre-run)",
           "cells": CELLS, "rows": rows, "verdict": verdict}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("\nVERDICT:", json.dumps(verdict, indent=1))
    print("OUT:", OUT)


if __name__ == "__main__":
    main()
