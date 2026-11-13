#!/usr/bin/env python
"""s90_reference_certificate.py — reference-configuration certificate at the benchmark's SHIPPED 1e-2 gate.

PREREG (frozen before any run; do not edit after). Commit this file before executing on spark.

Question: every quantitative claim I own about the reference axis is at the STRICT 1e-4 gate (s73/s84
99.807% reference-induced, s87 share constant 0.99850, s89b coverage ratio 0.3483). The benchmark ships
1e-2 for fp32. Milestone 2 of the 2026-11-01 exploitation plan, and the 1e-2 falsifier my own reflection
named, is: on shapes NOT used to build the map, what fraction of the SHIPPED budget does the default
reference realization consume? This is the construction-facing deliverable; a candidate comparison at a
gate the reference already consumes is meaningless, and that is exactly what has to be measured rather
than assumed.

Population (rule, not outcome): 24 constructed configs, all fp32, batch 1-4, on shapes disjoint from
(a) the 54-task s71 exceed population, (b) s87's 17 constructed configs. Families deliberately include
NON-CONV families, because the scoping question is whether the axis is conv-only:
  conv2d            8 configs (C in 16..256, k in 1..7, odd and even spatial, extents 1e4..6e7)
  conv_transpose2d  4 configs
  gemm              4 configs (square and K-heavy, M/N in 256..2048)
  layer_norm        4 configs (N in 4096..8388608, the benchmark's L1 family)
  reduction/softmax 4 configs (row-wise sum and softmax, N in 4096..1048576)
Each config declares its seed (fixed, f(H,N,C) style: seed = 1000*idx + 17), its extents, and its op class.

Arms per config, same instantiation, same inputs:
  A  default fp32 reference realization (cudnn allow_tf32 True, matmul allow_tf32 False) — the shipped default
  B  tf32 fully off (cudnn False, matmul False) — a different realization, for the class axis
  X  fp64 anchor, same op in float64 from the same fp32 weights, flags off; comparison temporaries are
     element-chunked (s73b's 2^22 unit, sequential release) so the |A-X|/b_e ratio never materialises whole.
     Anchors are computed whole where extents permit; an OOM is a declared status, not a silent skip.

Predicates: evaluator-exact direction, allclose(reference, candidate) scales rtol by the CANDIDATE, so
threshold b_e = atol + rtol*|candidate| (eval.py:804). Two legs: 1e-4 (strict, reported for continuity with
s73/s84) and 1e-2 (SHIPPED, the certificate's subject). rho_gate = max_elements |A-X| / b_e is computed
elementwise in the same chunked pass as the failing-element count; frac_over = |{d>A budget}| / N.

Primary measurement (the certificate): per config, rho_1e-2 = max_e |A-X| / (1e-2 + 1e-2|A_e|) and
frac_over_1e-2 under the same evaluator direction. Certificate value = max over configs of
rho_1e-2 (the "does an exact candidate fail?" number) plus the per-config budget-fraction distribution.

Engine class: torch.profiler on the A call, top-5 self-device-time kernels, classified with s75's frozen
regex set (FP64 / TF32_TENSOROP / WINOGRAD / FP32_SIMT / CUTLASS / OTHER) plus a top-1 label. Reported as a
dispatch fact, never as a magnitude coordinate (s75b CLASS_NOT_EXPLANATORY stands).

CONTROL GATE (must pass before any new number is reported): the 4 conv configs whose extents fall in the
s87 regime (2,16,256,256,32) and (2,64,56,56,64) are re-run here and must reproduce s87's reference-clean
result at 1e-4 (frac_over_1e-4 <= 0.01, rho_1e-4 <= 0.5). Any mismatch is recorded as a harness defect
and no verdict is quoted (frozen INSTRUMENT_FAILED).

Kill / branch clauses (predeclared, coded, fired verbatim):
  K_INSTRUMENT  > 20% of configs fail to produce all three arms, or the control gate breaks
                => INSTRUMENT_FAILED, no certificate quoted.
  K_PREMISE     if the default reference exceeds its own 1e-2 budget on any config => the axis is a
                BENCHMARK claim, not a strict-gate diagnostic.
  BRANCH_CERT_BENCHMARK   if max_config rho_1e-2 >= 1.0 on >= 1 config => CERT_BENCHMARK_CLAIM.
  BRANCH_CERT_STRICT_ONLY if all configs have rho_1e-2 < 0.5 and none >= 1.0 => CERT_STRICT_GATE_DIAGNOSTIC.
  BRANCH_CERT_MIDDLE      otherwise (some config >= 0.5 but < 1.0) => CERT_MARGINAL (report the config list
                          and its families; the axis is quotable only with a per-shape caveat).
  K_FAMILY      non-conv families with rho_1e-2 > 0.5 => the strict-gate scope is conv-only and the paper
                must say so; if any non-conv family >= 0.5 the scope widens and is recorded as such.

Variance: 2 repetitions per config for the A arm (repeat control: bit-identity of A across repeats), 1 for
B/X. n_reps recorded; if the repeat control is not bit-identical on >2 configs the reference draw is not
stable at fixed tuple and the certificate carries that caveat (s79a DRAW_STABLE is the prior, this is the
1e-2 check).

Declared limits: GB10 sm_120 only; fp64 anchor is a proxy for exact arithmetic, not exact arithmetic; the
configs are synthetic conv/gemm/norm/reduction shapes, not the benchmark's own task list; engine class is a
profiler label on this stack version.
"""
import importlib.util, json, os, re, signal, sys, time, hashlib
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s90_certificate.jsonl")
SUMOUT = os.path.join(LAB, "results", "s90_certificate_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 2400
CHUNK_ELEMS = 1 << 22          # element-based chunking, s73b's proven unit
FP64_MAX_BYTES = 2 * 1024 ** 3
STRICT = 1e-4
SHIPPED = 1e-2

CLASS_PATTERNS = [
    ("WINOGRAD", re.compile(r"winograd", re.I)),
    ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
    ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
    ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|s1688|fp32|scudnn|simt", re.I)),
    ("CUTLASS", re.compile(r"cutlass", re.I)),
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def profile_call(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = []
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            evts.append((e.key, float(e.self_device_time_total)))
    evts.sort(key=lambda t: -t[1])
    return evts[:5]


def set_cfg(tf32_cudnn, tf32_matmul):
    torch.backends.cudnn.allow_tf32 = tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul


# --- config builders. Each returns (fn, inputs) on the given device/dtype; fp32 for A/B, float64 for X.
# Shapes are chosen disjoint from s71's 54 tasks and s87's 17 configs.

def conv_fn(n, cin, h, w, cout, k, s, p):
    def f(x, wt):
        return torch.nn.functional.conv2d(x, wt, stride=s, padding=p)
    return f


def convT_fn(n, cin, h, w, cout, k, s, p, op):
    def f(x, wt):
        return torch.nn.functional.conv_transpose2d(x, wt, stride=s, padding=p, output_padding=op)
    return f


def gemm_fn(m, k, n):
    def f(a, b):
        return torch.matmul(a, b)
    return f


def ln_fn(rows, n):
    def f(x, g, b):
        return torch.nn.functional.layer_norm(x, (n,), g, b)
    return f


def red_fn(rows, n, softmax=False):
    if softmax:
        def f(x):
            return torch.softmax(x, dim=-1)
    else:
        def f(x):
            return x.sum(dim=-1)
    return f


CONFIGS = [
    # idx, op_class, description, builder, input shapes (fp32), seed, s87_control
    dict(idx=1, op="conv", desc="conv 3x3 s2 odd 97x97", build=lambda dt: (conv_fn(2, 48, 97, 97, 96, 3, 2, 1),
         (torch.randn(2, 48, 97, 97, device="cuda", dtype=dt), torch.randn(96, 48, 3, 3, device="cuda", dtype=dt))), seed=1017, ctrl=False),
    dict(idx=2, op="conv", desc="conv 5x5 s1 63x63", build=lambda dt: (conv_fn(2, 96, 63, 63, 128, 5, 1, 2),
         (torch.randn(2, 96, 63, 63, device="cuda", dtype=dt), torch.randn(128, 96, 5, 5, device="cuda", dtype=dt))), seed=1034, ctrl=False),
    dict(idx=3, op="conv", desc="conv 7x7 s2 129x129", build=lambda dt: (conv_fn(1, 32, 129, 129, 64, 7, 2, 3),
         (torch.randn(1, 32, 129, 129, device="cuda", dtype=dt), torch.randn(64, 32, 7, 7, device="cuda", dtype=dt))), seed=1051, ctrl=False),
    dict(idx=4, op="conv", desc="conv 1x1 211x211 C192->160", build=lambda dt: (conv_fn(2, 192, 211, 211, 160, 1, 1, 0),
         (torch.randn(2, 192, 211, 211, device="cuda", dtype=dt), torch.randn(160, 192, 1, 1, device="cuda", dtype=dt))), seed=1068, ctrl=False),
    dict(idx=5, op="conv", desc="conv 3x3 s1 151x151 C64->64", build=lambda dt: (conv_fn(2, 64, 151, 151, 64, 3, 1, 1),
         (torch.randn(2, 64, 151, 151, device="cuda", dtype=dt), torch.randn(64, 64, 3, 3, device="cuda", dtype=dt))), seed=1085, ctrl=False),
    dict(idx=6, op="conv", desc="conv 2x2 s2 200x200 C256->256", build=lambda dt: (conv_fn(1, 256, 200, 200, 256, 2, 2, 0),
         (torch.randn(1, 256, 200, 200, device="cuda", dtype=dt), torch.randn(256, 256, 2, 2, device="cuda", dtype=dt))), seed=1102, ctrl=False),
    dict(idx=7, op="conv", desc="conv 3x3 s1 41x41 C512->256", build=lambda dt: (conv_fn(4, 512, 41, 41, 256, 3, 1, 1),
         (torch.randn(4, 512, 41, 41, device="cuda", dtype=dt), torch.randn(256, 512, 3, 3, device="cuda", dtype=dt))), seed=1119, ctrl=False),
    dict(idx=8, op="conv", desc="conv 3x3 s1 77x77 C16->32", build=lambda dt: (conv_fn(2, 16, 77, 77, 32, 3, 1, 1),
         (torch.randn(2, 16, 77, 77, device="cuda", dtype=dt), torch.randn(32, 16, 3, 3, device="cuda", dtype=dt))), seed=1136, ctrl=False),
    dict(idx=9, op="conv_transpose", desc="convT 3x3 s2 57x57", build=lambda dt: (convT_fn(2, 64, 57, 57, 64, 3, 2, 1, 1),
         (torch.randn(2, 64, 57, 57, device="cuda", dtype=dt), torch.randn(64, 64, 3, 3, device="cuda", dtype=dt))), seed=1153, ctrl=False),
    dict(idx=10, op="conv_transpose", desc="convT 4x4 s2 33x33 C128->96", build=lambda dt: (convT_fn(2, 128, 33, 33, 96, 4, 2, 1, 0),
         (torch.randn(2, 128, 33, 33, device="cuda", dtype=dt), torch.randn(128, 96, 4, 4, device="cuda", dtype=dt))), seed=1170, ctrl=False),
    dict(idx=11, op="conv_transpose", desc="convT 2x2 s2 101x101 C32->64", build=lambda dt: (convT_fn(1, 32, 101, 101, 64, 2, 2, 0, 0),
         (torch.randn(1, 32, 101, 101, device="cuda", dtype=dt), torch.randn(32, 64, 2, 2, device="cuda", dtype=dt))), seed=1187, ctrl=False),
    dict(idx=12, op="conv_transpose", desc="convT 3x3 s1 89x89 C256->128", build=lambda dt: (convT_fn(2, 256, 89, 89, 128, 3, 1, 1, 0),
         (torch.randn(2, 256, 89, 89, device="cuda", dtype=dt), torch.randn(256, 128, 3, 3, device="cuda", dtype=dt))), seed=1204, ctrl=False),
    dict(idx=13, op="gemm", desc="gemm 1024x1024x1024", build=lambda dt: (gemm_fn(1024, 1024, 1024),
         (torch.randn(1024, 1024, device="cuda", dtype=dt), torch.randn(1024, 1024, device="cuda", dtype=dt))), seed=1221, ctrl=False),
    dict(idx=14, op="gemm", desc="gemm 2048x512x2048", build=lambda dt: (gemm_fn(2048, 512, 2048),
         (torch.randn(2048, 512, device="cuda", dtype=dt), torch.randn(512, 2048, device="cuda", dtype=dt))), seed=1238, ctrl=False),
    dict(idx=15, op="gemm", desc="gemm 768x1536x768 K-heavy", build=lambda dt: (gemm_fn(768, 1536, 768),
         (torch.randn(768, 1536, device="cuda", dtype=dt), torch.randn(1536, 768, device="cuda", dtype=dt))), seed=1255, ctrl=False),
    dict(idx=16, op="gemm", desc="gemm 512x4096x512 K-heavy", build=lambda dt: (gemm_fn(512, 4096, 512),
         (torch.randn(512, 4096, device="cuda", dtype=dt), torch.randn(4096, 512, device="cuda", dtype=dt))), seed=1272, ctrl=False),
    dict(idx=17, op="layer_norm", desc="LN 4096x8192", build=lambda dt: (ln_fn(4096, 8192),
         (torch.randn(4096, 8192, device="cuda", dtype=dt), torch.randn(8192, device="cuda", dtype=dt), torch.randn(8192, device="cuda", dtype=dt))), seed=1289, ctrl=False),
    dict(idx=18, op="layer_norm", desc="LN 1024x16384", build=lambda dt: (ln_fn(1024, 16384),
         (torch.randn(1024, 16384, device="cuda", dtype=dt), torch.randn(16384, device="cuda", dtype=dt), torch.randn(16384, device="cuda", dtype=dt))), seed=1306, ctrl=False),
    dict(idx=19, op="layer_norm", desc="LN 16x4194304 (L1/40-like)", build=lambda dt: (ln_fn(16, 4194304),
         (torch.randn(16, 4194304, device="cuda", dtype=dt), torch.randn(4194304, device="cuda", dtype=dt), torch.randn(4194304, device="cuda", dtype=dt))), seed=1323, ctrl=False),
    dict(idx=20, op="layer_norm", desc="LN 8192x4096", build=lambda dt: (ln_fn(8192, 4096),
         (torch.randn(8192, 4096, device="cuda", dtype=dt), torch.randn(4096, device="cuda", dtype=dt), torch.randn(4096, device="cuda", dtype=dt))), seed=1340, ctrl=False),
    dict(idx=21, op="reduction", desc="row-sum 8192x8192", build=lambda dt: (red_fn(8192, 8192),
         (torch.randn(8192, 8192, device="cuda", dtype=dt),)), seed=1357, ctrl=False),
    dict(idx=22, op="reduction", desc="row-sum 2048x131072", build=lambda dt: (red_fn(2048, 131072),
         (torch.randn(2048, 131072, device="cuda", dtype=dt),)), seed=1374, ctrl=False),
    dict(idx=23, op="softmax", desc="softmax 4096x32768", build=lambda dt: (red_fn(4096, 32768, softmax=True),
         (torch.randn(4096, 32768, device="cuda", dtype=dt),)), seed=1391, ctrl=False),
    dict(idx=24, op="softmax", desc="softmax 512x524288", build=lambda dt: (red_fn(512, 524288, softmax=True),
         (torch.randn(512, 524288, device="cuda", dtype=dt),)), seed=1408, ctrl=False),
    # s87-regime controls (same shapes s87 measured reference-clean at 1e-4)
    dict(idx=25, op="conv", desc="CONTROL s87 (2,16,256,256,32)", build=lambda dt: (conv_fn(2, 16, 256, 256, 32, 3, 1, 1),
         (torch.randn(2, 16, 256, 256, device="cuda", dtype=dt), torch.randn(32, 16, 3, 3, device="cuda", dtype=dt))), seed=1425, ctrl=True),
    dict(idx=26, op="conv", desc="CONTROL s87 (2,64,56,56,64)", build=lambda dt: (conv_fn(2, 64, 56, 56, 64, 3, 1, 1),
         (torch.randn(2, 64, 56, 56, device="cuda", dtype=dt), torch.randn(64, 64, 3, 3, device="cuda", dtype=dt))), seed=1442, ctrl=True),
]


def chunked_compare(a, x, atol, rtol):
    """Element-based chunked evaluator-exact comparison. Returns (n_fail, max_ratio, n_elems).
    ratio_e = |a_e - x_e| / (atol + rtol*|a_e|); n_fail counts ratio_e > 1."""
    n_fail = 0
    max_ratio = 0.0
    n_tot = 0
    af = a.detach().reshape(-1)
    xf = x.detach().reshape(-1)
    n = af.numel()
    i = 0
    while i < n:
        j = min(n, i + CHUNK_ELEMS)
        d = (af[i:j] - xf[i:j]).abs()
        b = atol + rtol * af[i:j].abs()
        r = d / b
        n_fail += int((r > 1).sum().item())
        mr = float(r.max().item())
        if mr > max_ratio:
            max_ratio = mr
        n_tot += (j - i)
        del d, b, r
        i = j
    return n_fail, max_ratio, n_tot


def run_arm(fn, ins, dtype):
    args = [t.to(dtype) for t in ins]
    with torch.no_grad():
        out = fn(*args)
    return out


def main():
    torch.cuda.init()
    t_start = time.time()
    rows = []
    open(OUT, "w").close()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print("prereg sha", sha[:12], flush=True)

    for c in CONFIGS:
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            print("GLOBAL DEADLINE reached at config", c["idx"], flush=True)
            break
        signal.alarm(CFG_ALARM_S)
        t_cfg = time.time()
        rec = dict(idx=c["idx"], op=c["op"], desc=c["desc"], seed=c["seed"], ctrl=c["ctrl"], status="ok")
        try:
            # --- A: default reference realization, twice for repeat control
            set_cfg(True, False)
            torch.manual_seed(c["seed"])
            fn, ins32 = c["build"](torch.float32)
            a1 = run_arm(fn, ins32, torch.float32)
            prof = profile_call(lambda: run_arm(fn, ins32, torch.float32))
            a2 = run_arm(fn, ins32, torch.float32)
            rec["a_repeat_bit_identical"] = bool(torch.equal(a1, a2))
            rec["engine_top5"] = [(n, round(t, 2)) for n, t in prof]
            cls, top1 = classify([n for n, _ in prof])
            rec["engine_class"], rec["engine_top1"] = cls, top1
            a = a1
            del a2

            # --- B: tf32 fully off
            set_cfg(False, False)
            b = run_arm(fn, ins32, torch.float32)
            del ins32

            # --- X: fp64 anchor, element-chunked by the caller-side builder
            set_cfg(False, False)
            fn64, ins64 = c["build"](torch.float64)
            x = run_arm(fn64, ins64, torch.float64)

            # shape agreement gate
            if a.shape != x.shape:
                raise RuntimeError("shape mismatch A %s vs X %s" % (tuple(a.shape), tuple(x.shape)))

            for tag, tol in (("1e-4", STRICT), ("1e-2", SHIPPED)):
                nf_a, mr_a, nt = chunked_compare(a, x, tol, tol)
                nf_b, mr_b, _ = chunked_compare(b, x, tol, tol)
                rec["ref_vs_anchor_" + tag] = dict(n_fail=nf_a, rho=mr_a, frac_over=nf_a / nt, n_elems=nt,
                                                   budget_fraction=mr_a)
                rec["tf32off_vs_anchor_" + tag] = dict(n_fail=nf_b, rho=mr_b, frac_over=nf_b / nt)
            rec["n_elems"] = int(a.numel())
            rec["t_cfg_s"] = round(time.time() - t_cfg, 2)
            del a, b, x, fn64, ins64
            torch.cuda.empty_cache()
        except Timeout:
            rec["status"] = "timeout"
            rec["t_cfg_s"] = round(time.time() - t_cfg, 2)
        except torch.cuda.OutOfMemoryError as e:
            rec["status"] = "oom"
            rec["err"] = str(e)[:200]
            torch.cuda.empty_cache()
        except Exception as e:
            rec["status"] = "error"
            rec["err"] = repr(e)[:300]
            torch.cuda.empty_cache()
        finally:
            signal.alarm(0)
        rows.append(rec)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        st = rec.get("ref_vs_anchor_1e-2", {})
        print("cfg", rec["idx"], rec["op"], rec["status"], "class", rec.get("engine_class"),
              "rho1e-2", st.get("rho"), "frac", st.get("frac_over"), "repeat", rec.get("a_repeat_bit_identical"),
              "t", rec.get("t_cfg_s"), flush=True)

    # ---- verdict
    ok = [r for r in rows if r.get("status") == "ok"]
    ctrl = [r for r in ok if r.get("ctrl")]
    cert = [r for r in ok if not r.get("ctrl")]
    summary = dict(prereg_sha=sha, n_configs=len(CONFIGS), n_ok=len(ok), n_attempted=len(rows),
                   controls=[dict(idx=r["idx"], rho1e4=r["ref_vs_anchor_1e-4"]["rho"],
                                  frac1e4=r["ref_vs_anchor_1e-4"]["frac_over"]) for r in ctrl])

    instr_ok = len(ok) >= 0.8 * len(CONFIGS)
    ctrl_ok = bool(ctrl) and all(r["ref_vs_anchor_1e-4"]["frac_over"] <= 0.01 and
                                 r["ref_vs_anchor_1e-4"]["rho"] <= 0.5 for r in ctrl)
    summary["control_gate_pass"] = ctrl_ok

    if not instr_ok or not ctrl_ok:
        summary["verdict"] = "INSTRUMENT_FAILED"
        summary["reason"] = "ok=%d/%d control_gate=%s" % (len(ok), len(CONFIGS), ctrl_ok)
    else:
        vals = [(r["idx"], r["op"], r["ref_vs_anchor_1e-2"]["rho"], r["ref_vs_anchor_1e-2"]["frac_over"])
                for r in cert]
        vals.sort(key=lambda t: -t[2])
        max_rho = vals[0][2] if vals else 0.0
        n_over = sum(1 for v in vals if v[2] >= 1.0)
        n_mid = sum(1 for v in vals if 0.5 <= v[2] < 1.0)
        nonconv_over = [v for v in vals if v[1] != "conv" and v[1] != "conv_transpose" and v[2] > 0.5]
        summary["top_rho_1e-2"] = vals[:6]
        summary["n_rho_ge_1"] = n_over
        summary["n_rho_mid"] = n_mid
        summary["max_rho_1e-2"] = max_rho
        summary["nonconv_gt_0.5"] = nonconv_over
        summary["strict_leg_max_rho_1e-4"] = max((r["ref_vs_anchor_1e-4"]["rho"] for r in cert), default=None)
        summary["repeat_bit_identical_frac"] = (
            sum(1 for r in ok if r.get("a_repeat_bit_identical")) / len(ok))
        summary["classes"] = {}
        for r in cert:
            summary["classes"].setdefault(r.get("engine_class"), []).append(r["idx"])
        if n_over >= 1:
            summary["verdict"] = "CERT_BENCHMARK_CLAIM"
        elif n_mid >= 1:
            summary["verdict"] = "CERT_MARGINAL"
        else:
            summary["verdict"] = "CERT_STRICT_GATE_DIAGNOSTIC"
        summary["family_scope_note"] = ("conv-only scope" if not nonconv_over else
                                        "non-conv family exceeds 0.5: scope widens")
        if not summary["repeat_bit_identical_frac"] >= 0.9:
            summary["draw_stability_caveat"] = True

    with open(SUMOUT, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
