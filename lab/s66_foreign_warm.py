"""s66 -- foreign-warm probe: is the reference realization stable under invocation context?

PRE-REGISTRATION (frozen BEFORE first run; this docstring is the registration).

Why: the paper's contribution is that a benchmark verdict is reference-conditional, and the reference
axis was measured by swapping precision configuration (r41) and by observing the dispatched engine
(s63). Both hold the invocation context fixed: one call, warm cache, cudnn.benchmark=False,
deterministic=False. The reflection EPryIvzq1SbrGnDU named the honest gap -- the ensemble over which
admission would be a rate is (input draw x reference realization), and if the realization also moves
with cache state or autotune mode then the point predicate fails for a reason that does not depend on
the tolerance argument at all. This probe tests exactly that: same shapes, same inputs, same flags,
only the invocation context varies.

Population (fixed by rule, not by outcome): the 11 rows of results/gate_stability.jsonl with
armA_delivered_change=true, plus 6 controls = the no-change rows taken in ascending (H,N,C) order at
every 7th position. 17 rows total.

Contexts (each a single reference call, profiled):
  A  warm, default flags, cudnn.benchmark=False, deterministic=False   (the paper's baseline context)
  B  cold: 256 MiB zero-fill immediately before the call (L2 flush), otherwise A
  C  cudnn.benchmark=True (autotune allowed), otherwise A
  D  torch.use_deterministic_algorithms(True), otherwise A
  E  benchmark=True + deterministic=True
  F  allow_tf32=False (the R_ieee reference), otherwise A -- replication check against s63

Recorded per (row, context): classified engine class, raw top-5 kernels with device times, the
reference output's absmax, and rho_context = max|R_ctx - R_A| / (1e-4 + 1e-4|R_A|) against context A.
No timing claims: profiler-instrumented calls only.

Declared reads (two-class, written before the run; P5 -- either outcome is a result):
  R1 (stability): if the engine class is identical across A-E on all 17 rows and rho_context == 0
     everywhere, the reference realization is context-stable on this population and the ensemble needs
     only the (input draw x precision configuration) axes the paper already measures. Kill clause:
     any class change or any rho_context > 0 falsifies stability and adds invocation context as a third
     axis; the per-row labels are the result, no aggregate threshold decides wording.
  R2 (replication): context F must reproduce s63's R_ieee engine class on every row where s63 recorded
     one; a mismatch means the observation pipeline itself is context-dependent and no context claim is
     admissible until that is resolved.

Run on spark: python3 s66_foreign_warm.py   (venv ~/kernelbench/.venv/bin/python)
Output: results/s66_foreign_warm.jsonl + results/s66_foreign_warm_summary.json
"""
import json, os, re, time
import torch
import torch.nn.functional as F

from r1_conv_gated import GRIDS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTL = os.path.join(HERE, "results", "s66_foreign_warm.jsonl")
SUML = os.path.join(HERE, "results", "s66_foreign_warm_summary.json")
GATE = os.path.join(HERE, "results", "gate_stability.jsonl")
S63 = os.path.join(HERE, "results", "s63_engine_predicate.jsonl")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# Classifier = s63_engine_predicate.CLASS_PATTERNS verbatim. A narrower set was used on the first run and
# labelled one cudnn direct kernel OTHER where s63 calls it FP32_SIMT; results/s66b_relabel.json recomputes
# every label under this set from the recorded kernel names and prints both.
CLASSES = [("WINOGRAD", re.compile(r"winograd", re.I)),
           ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
           ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
           ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
           ("CUTLASS", re.compile(r"cutlass", re.I))]


def classify(names):
    for label, pat in CLASSES:
        for n in names:
            if pat.search(n):
                return label
    return "OTHER"


def profile_conv(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    ev = []
    for e in prof.key_averages():
        if e.device_time_total > 0:
            ev.append((e.key, e.device_time_total))
    ev.sort(key=lambda t: -t[1])
    return out, [(n, round(us, 2)) for n, us in ev[:5]]


def rho(A, B):
    return float(((A - B).abs().max()) / (1e-4 + 1e-4 * B.abs().max()))


def pick_rows():
    gate = [json.loads(l) for l in open(GATE)]
    change = [r for r in gate if r.get("armA_delivered_change")]
    rest = [r for r in gate if not r.get("armA_delivered_change")]
    rest.sort(key=lambda r: (r["H"], r["N"], r["C"]))
    controls = rest[::7][:6]
    return change + controls


def main():
    rows = pick_rows()
    grid = {}
    for _tag, pts in GRIDS:
        for (h, n, c) in pts:
            grid[(h, n, c)] = (n, h, h, c, c)  # N, H, W, C, CO: the r1 conv grids are square CO=C
    s63 = {}
    if os.path.exists(S63):
        for l in open(S63):
            r = json.loads(l)
            s63[(r["H"], r["N"], r["C"])] = r["refs"]["ieee"]["engine_class"]
    torch.backends.cudnn.allow_tf32 = True
    out = open(OUTL, "w")
    summary = {"date": DATE, "prereg": "docstring of lab/s66_foreign_warm.py",
               "n_rows": len(rows), "contexts": ["A", "B", "C", "D", "E", "F"],
               "rows": [], "engine_by_context": {}, "transitions": {}, "rho_max_by_context": {},
               "replication_F_vs_s63": {"agree": 0, "disagree": 0, "detail": []}}
    flush = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for r in rows:
        H, N, C = r["H"], r["N"], r["C"]
        CO = r.get("CO", C)
        if grid:
            g = grid.get((H, N, C))
            if g is None:
                print("no grid for", (H, N, C), flush=True)
                continue
            Nn, Hh, Ww, Cc, COo = g[0], g[1], g[2], g[3], g[4]
        else:
            Nn, Hh, Ww, Cc, COo = N, H, H, C, CO
        torch.manual_seed(1234 + Hh * 100 + Nn * 10 + Cc)
        x = torch.randn(Nn, Cc, Hh, Ww, device="cuda")
        w = torch.randn(COo, Cc, 3, 3, device="cuda")
        rec = {"H": Hh, "N": Nn, "W": Ww, "C": Cc, "CO": COo, "date": DATE,
               "armA_delivered_change": bool(r.get("armA_delivered_change")),
               "contexts": {}}
        base = None
        for ctx in ("A", "B", "C", "D", "E", "F"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(False)
            if ctx == "B":
                flush.zero_(); torch.cuda.synchronize()
            elif ctx == "C":
                torch.backends.cudnn.benchmark = True
            elif ctx == "D":
                torch.use_deterministic_algorithms(True)
            elif ctx == "E":
                torch.backends.cudnn.benchmark = True
                torch.use_deterministic_algorithms(True)
            elif ctx == "F":
                torch.backends.cudnn.allow_tf32 = False
            try:
                y, kernels = profile_conv(lambda: F.conv2d(x, w, padding=1))
                lab = classify([n for n, _ in kernels])
                rec["contexts"][ctx] = {"engine_class": lab, "kernels": kernels,
                                        "absmax": float(y.abs().max())}
                if ctx == "A":
                    base = y.clone()
                else:
                    rec["contexts"][ctx]["rho_vs_A"] = round(rho(y, base), 6) if base is not None else None
                summary["engine_by_context"].setdefault(ctx, {}).setdefault(lab, 0)
                summary["engine_by_context"][ctx][lab] += 1
            except Exception as e:
                rec["contexts"][ctx] = {"error": str(e)[:200]}
            finally:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(False)
        classes = {c: rec["contexts"][c].get("engine_class") for c in ("A", "B", "C", "D", "E")}
        rec["context_stable"] = len(set(classes.values())) == 1
        rec["transitions"] = {c: classes[c] for c in classes if classes[c] != classes["A"]}
        f = rec["contexts"].get("F", {}).get("engine_class")
        ref = s63.get((Hh, Nn, Cc))
        if ref is not None:
            if f == ref:
                summary["replication_F_vs_s63"]["agree"] += 1
            else:
                summary["replication_F_vs_s63"]["disagree"] += 1
                summary["replication_F_vs_s63"]["detail"].append(
                    {"row": [Hh, Nn, Cc], "s66_F": f, "s63_ieee": ref})
        for c in ("B", "C", "D", "E"):
            v = rec["contexts"].get(c, {}).get("rho_vs_A")
            if v is not None:
                summary["rho_max_by_context"][c] = max(summary["rho_max_by_context"].get(c, 0.0), v)
        summary["rows"].append({"row": [Hh, Nn, Cc], "context_stable": rec["context_stable"],
                                "transitions": rec["transitions"],
                                "classes": classes})
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"row {Hh}x{Ww} N{Nn} C{Cc}: A={classes['A']} stable={rec['context_stable']} "
              f"trans={rec['transitions']}", flush=True)
    out.close()
    summary["n_context_stable"] = sum(1 for r in summary["rows"] if r["context_stable"])
    summary["n_unstable"] = len(summary["rows"]) - summary["n_context_stable"]
    json.dump(summary, open(SUML, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("n_rows", "n_context_stable", "n_unstable",
                                              "engine_by_context", "rho_max_by_context",
                                              "replication_F_vs_s63")}, indent=1))


if __name__ == "__main__":
    main()
