#!/usr/bin/env python
"""s87_fresh_shape_share.py — does the reference-induced share constant generalize outside the 54-row population?

PREREG (frozen before the first run; do not edit after). Commit this file BEFORE executing it on spark.

WHY. The deliverable of this lane is a per-shape error-budget column carrying (a) a per-shape tail magnitude and
(b) a class-level reference-induced share, whose measured value on the 54 exceed rows is 99.807% of 2.183B failing
elements (s84; per-shape median 0.9993, one named exception L2/11 at 22.31%). Three magnitude keys have failed
(engine class s84, continuous geometry s85, binned reproducible key s86), so the magnitude question is parked and
the share is the only surviving compressible quantity. A constant measured on one population of benchmark tasks is
not yet a general statement: the previous reflection's decision was to test whether it survives outside that
population, with a predeclared kill, rather than to hunt a fourth magnitude key. This run is that test.

POPULATION (constructed, not sampled from outcomes). 16 synthetic conv2d / conv_transpose2d configurations on
GB10, from a declared lattice, none of whose (op_class, N, Cin, H, W, Cout) tuple matches any of the 54 exceed
rows' parsed geometry (the disjointness check is computed in-run against s75_engine_class_s73pop.jsonl and
reported; a coincident tuple is recorded and left in the population, never silently dropped). k=3, s=1, p=1
throughout, so conv_transpose output extent equals input extent. Weights and inputs are the framework defaults
(torch.nn.Conv2d/ConvTranspose2d default init; inputs torch.randn), which is what the benchmark's task models use.
Seeds: 8701+i per configuration, declared here so a rerun reproduces the draws.

The lattice is stratified by output extent, because extent is the one axis on which the population's own rows
cannot answer anything (all 54 sit at >=1e8 elements):
  small band  < 1e7 output elements   10 configurations
  mid band    1e7 .. 1e8              4 configurations
  large band  >= 1e8                  3 configurations
The band assignment is computed from the constructed shape, never from a result.

ARMS per configuration, one model instance and one input draw (s70 lesson):
  A   default fp32 reference realization: cudnn tf32 on, matmul tf32 off (the benchmark's shipped default; s38
      established that this IS TF32 on sm_120)
  X   fp64 anchor: same weights widened exactly, flags off
  C2  fp32 candidate with tf32 off (s73's C2; the arm that fails at 1e-4)
torch.compile (s73's C1) is deliberately omitted: on the 54 rows it failed 0/52 tasks at 1e-4, so it contributes
nothing to a failing-element split, and dynamo compile time would eat the tick. Declared limitation.

PREDICATES. Evaluator-exact direction, b_e = atol + rtol*|candidate| (eval.py:804, verified in earlier ticks).
Legs: 1e-4 (the axis of this lane) and 1e-2 (the benchmark's shipped gate for fp32).
Reference-induced element := (|C2-A| > b_e) AND (|A-X| > b_e); candidate-induced := (|C2-A| > b_e) AND NOT(|A-X| >
b_e). This is s73/s84's definition unchanged.

MEASUREMENTS per configuration: total elements, fail_default count, ref_induced, cand_induced, share of fails,
reference-only over-budget fraction, max_ratio_ref = max(|A-X|/b_e), max_ratio_cand, bit-identity of C2 vs A,
and the same at 1e-2. Plus per-shape wall times.

VERDICT RULES (frozen; evaluated in this order):
  V1 SHARE_GENERALIZES      iff pooled share over configurations with fail_default >= 1000 is >= 0.95 AND every
                            band with >=3 such configurations has share >= 0.90.
  V2 EXTENT_SCOPED          iff the large band (>=1e8) share is >= 0.95 AND the small band (<1e7) share is < 0.90
                            (V1 not met).
  V3 SHARE_NOT_TRANSFERABLE otherwise (pooled < 0.95 with no band separation, or bands separate in the opposite
                            direction).
KILL CLAUSES:
  K1 VACUOUS: fewer than 8 configurations with fail_default > 0 at 1e-4 => the constructed population is too clean
     to test the share; report the null and make no share claim on fresh shapes.
  K2 LOW_N: configurations with fail_default < 1000 are reported but excluded from band verdicts and from the
     pooled share (a share over a handful of elements is noise, not a measurement). Counted, not hidden.
  K3 INSTRUMENT: if any configuration's control condition fails (C2 bit-identical to A yet fail_default > 0, or a
     shape where the reference is over budget on 0 elements yet the candidate-induced share is < 0.5 with
     fail_default >= 1000), the harness is recorded as instrument-defective and the verdict is reported as such.
WHAT THIS DOES AND DOES NOT ESTABLISH (written before the run): a V1 result licenses the constant as a
population-independent statement about GB10 conv/convT reference realizations at this gate, with the extent band
as its only declared scope. V2 licenses it only for large extents. V3 retires the constant as a general statement
and leaves the per-shape share as the only citable form. In every branch the magnitude column stays per-shape
(s84/s85/s86) and no magnitude law is revived.
"""
import importlib.util, json, os, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s87_fresh_shape_share.jsonl")
SUMOUT = os.path.join(RES, "s87_fresh_shape_share_summary.json")
S75 = os.path.join(RES, "s75_engine_class_s73pop.jsonl")
os.makedirs(RES, exist_ok=True)

ATOL_RTOL_STRICT = 1e-4
ATOL_RTOL_BENCH = 1e-2
CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 1500
ELEM_CHUNK = 1 << 22
RATIO_CAP = 20000
MIN_FAILS_FOR_BAND = 1000
SHARE_BAR = 0.95
BAND_FLOOR = 0.90

# (op, N, Cin, H, W, Cout) — declared lattice, k=3 s=1 p=1
POP = [
    ("conv",            2,  64,  56,  56,  64),   # 401k    small
    ("conv",            2,  16, 256, 256,  32),   # 4.2M    small
    ("conv",            1, 256, 160, 160, 128),   # 3.3M    small
    ("conv",            4, 128,  64,  64, 128),   # 2.1M    small
    ("conv",            2,  32, 224, 224,  64),   # 6.4M    small
    ("conv",            8,  64, 112, 112, 128),   # 12.8M   mid
    ("conv",            8, 128,  96,  96,  96),   # 7.1M    small
    ("conv",            2,  64, 256, 256,  64),   # 8.4M    small
    ("conv",            8, 128, 128, 128, 128),   # 16.8M   mid
    ("conv",            8,  32,  96,  96,  96),   # 7.1M    small
    ("conv",            8,  64, 224, 224, 256),   # 102.8M  large
    ("conv",            4,  64, 320, 320, 256),   # 104.9M  large
    ("conv_transpose",  8,  64, 112, 112,  64),   # 6.4M    small
    ("conv_transpose",  2,  32, 224, 224,  16),   # 1.6M    small
    ("conv_transpose",  8, 256,  96,  96, 256),   # 18.9M   mid
    ("conv_transpose",  8,  64, 224, 224, 128),   # 51.4M   mid
    ("conv_transpose",  4,  64, 320, 320, 256),   # 104.9M  large
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("config exceeded %ds" % CFG_ALARM_S)


signal.signal(signal.SIGALRM, _alarm)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S73B = load_mod("s73b", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s73b_oom_rerun.py"))


def band_of(total):
    if total >= 100_000_000:
        return "large_1e8_plus"
    if total >= 10_000_000:
        return "mid_1e7_1e8"
    return "small_lt_1e7"


def parse_pop_geometry():
    """Parsed 4-D conv/convT tuples of the 54-row population, for the disjointness check."""
    tups = set()
    for line in open(S75):
        r = json.loads(line)
        if r.get("op_class") not in ("conv", "conv_transpose"):
            continue
        ins = r.get("input_shapes") or []
        out = r.get("output_shape") or []
        if len(ins) != 1 or len(ins[0]) != 4 or len(out) != 4:
            continue
        n, cin, h, w = ins[0]
        tups.add((r["op_class"], int(n), int(cin), int(h), int(w), int(out[1])))
    return tups


def build(op, n, cin, h, w, cout, seed):
    torch.manual_seed(seed)
    if op == "conv":
        m = torch.nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=True)
    else:
        m = torch.nn.ConvTranspose2d(cin, cout, 3, stride=1, padding=1, bias=True)
    x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
    return m, [x]


def run(model, inputs, dtype):
    model = model.to(device="cuda", dtype=dtype)
    model.eval()
    with torch.no_grad():
        out = model(*[t.to(dtype=dtype) for t in inputs])
    return out


def main():
    torch.cuda.init()
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    pop_tuples = parse_pop_geometry()
    print("population tuples parsed", len(pop_tuples), flush=True)

    rows = []
    for i, (op, n, cin, h, w, cout) in enumerate(POP):
        rec = {"idx": i, "op_class": op, "N": n, "Cin": cin, "H": h, "W": w, "Cout": cout,
               "out_extent": n * cout * h * w, "seed": 8701 + i}
        rec["band"] = band_of(rec["out_extent"])
        rec["tuple_in_population"] = bool((op, n, cin, h, w, cout) in pop_tuples)
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            rec["status"] = "deadline"
            rows.append(rec)
            continue
        t_cfg = time.time()
        try:
            model, inputs = build(op, n, cin, h, w, cout, rec["seed"])
            signal.alarm(CFG_ALARM_S)
            t0 = time.time()
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            A = run(model, inputs, torch.float32)
            rec["t_ref_s"] = round(time.time() - t0, 3)
            rec["out_shape"] = list(A.shape)

            t0 = time.time()
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            X = run(model, inputs, torch.float64)
            rec["t_anchor_s"] = round(time.time() - t0, 3)
            if X.shape != A.shape:
                raise RuntimeError("anchor shape mismatch")
            model.float()

            t0 = time.time()
            C2 = run(model, inputs, torch.float32)
            rec["t_cand_s"] = round(time.time() - t0, 3)
            signal.alarm(0)

            d = S73B.decomp_stream(A, X, C2, ATOL_RTOL_STRICT, ATOL_RTOL_BENCH)
            rec["decomp_c2_tf32off"] = d
            for tag in ("1e-4", "1e-2"):
                s = d[tag]
                s["ref_over_frac"] = s["ref_over_budget"] / max(1, s["total"])
                s["low_n"] = bool(s["fail_default"] < MIN_FAILS_FOR_BAND)
            rec["status"] = "ok"
            del A, X, C2
        except Timeout as e:
            rec["status"] = "timeout:%s" % e
        except Exception as e:
            rec["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:200])
        finally:
            signal.alarm(0)
        rec["t_cfg_s"] = round(time.time() - t_cfg, 2)
        rows.append(rec)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        dd = rec.get("decomp_c2_tf32off", {}).get("1e-4", {})
        print(json.dumps({"i": i, "op": op, "band": rec["band"], "ext": rec["out_extent"],
                          "st": rec["status"], "fail": dd.get("fail_default"),
                          "share": dd.get("ref_share_of_fails"), "t": rec["t_cfg_s"]}), flush=True)
        torch.cuda.empty_cache()

    ok = [r for r in rows if r.get("status") == "ok"]
    usable = [r for r in ok if not r["decomp_c2_tf32off"]["1e-4"]["low_n"]]

    def agg(sub):
        fail = sum(r["decomp_c2_tf32off"]["1e-4"]["fail_default"] for r in sub)
        ref = sum(r["decomp_c2_tf32off"]["1e-4"]["ref_induced"] for r in sub)
        cand = sum(r["decomp_c2_tf32off"]["1e-4"]["cand_induced"] for r in sub)
        return {"n": len(sub), "failing_elements": fail, "ref_induced": ref, "cand_induced": cand,
                "share": (ref / fail) if fail else None,
                "share_min_of_shapes": min((r["decomp_c2_tf32off"]["1e-4"]["ref_share_of_fails"] for r in sub
                                            if r["decomp_c2_tf32off"]["1e-4"]["fail_default"] > 0), default=None),
                "share_max_cand_of_shapes": max((r["decomp_c2_tf32off"]["1e-4"]["cand_share_of_fails"] for r in sub
                                                 if r["decomp_c2_tf32off"]["1e-4"]["fail_default"] > 0), default=None),
                "tasks_fail_default": sum(1 for r in sub if r["decomp_c2_tf32off"]["1e-4"]["fail_default"] > 0),
                "tasks_ref_over_budget": sum(1 for r in sub if r["decomp_c2_tf32off"]["1e-4"]["ref_over_budget"] > 0)}

    bands = {}
    for b in ("small_lt_1e7", "mid_1e7_1e8", "large_1e8_plus"):
        bands[b] = agg([r for r in usable if r["band"] == b])

    pooled = agg(usable)
    n_failing = sum(1 for r in ok if r["decomp_c2_tf32off"]["1e-4"]["fail_default"] > 0)

    verdict = None
    if n_failing < 8:
        verdict = "VACUOUS_K1"
    else:
        band_ok = [b for b, a in bands.items() if a["n"] >= 3]
        bands_pass = all((bands[b]["share"] or 0) >= BAND_FLOOR for b in band_ok)
        pooled_pass = (pooled["share"] or 0) >= SHARE_BAR
        if pooled_pass and bands_pass:
            verdict = "V1_SHARE_GENERALIZES"
        elif (bands["large_1e8_plus"]["share"] or 0) >= SHARE_BAR and \
                (bands["small_lt_1e7"]["share"] or 0) < BAND_FLOOR:
            verdict = "V2_EXTENT_SCOPED"
        else:
            verdict = "V3_SHARE_NOT_TRANSFERABLE"

    # instrument checks (K3)
    instr = []
    for r in ok:
        s = r["decomp_c2_tf32off"]["1e-4"]
        if s["bit_identical_ca"] and s["fail_default"] > 0:
            instr.append({"idx": r["idx"], "reason": "bit_identical_ca_but_fail_default"})
        if s["ref_over_budget"] == 0 and s["fail_default"] >= MIN_FAILS_FOR_BAND and \
                (s["cand_share_of_fails"] or 1) < 0.5:
            instr.append({"idx": r["idx"], "reason": "clean_reference_but_reference_induced_fails"})
    # controls: clean-reference shapes
    controls = [r for r in ok if r["decomp_c2_tf32off"]["1e-4"]["ref_over_budget"] == 0]

    summary = {
        "prereg": "docstring of this file, frozen pre-run",
        "prereg_sha256": prereg_sha,
        "population": {"constructed": len(POP), "attempted": len(rows), "ok": len(ok),
                       "usable_fail_ge_1000": len(usable), "failing_configs": n_failing,
                       "low_n_excluded": [r["idx"] for r in ok if r["decomp_c2_tf32off"]["1e-4"]["low_n"]],
                       "tuple_collisions_with_54": [r["idx"] for r in rows if r["tuple_in_population"]],
                       "statuses": {r["idx"]: r.get("status") for r in rows}},
        "pooled": pooled, "bands": bands,
        "controls_clean_reference": {"n": len(controls),
                                     "fail_default_total": sum(r["decomp_c2_tf32off"]["1e-4"]["fail_default"]
                                                               for r in controls),
                                     "cand_induced_total": sum(r["decomp_c2_tf32off"]["1e-4"]["cand_induced"]
                                                               for r in controls)},
        "at_1e-2": {"pooled_fail": sum(r["decomp_c2_tf32off"]["1e-2"]["fail_default"] for r in ok),
                    "pooled_share": (lambda f, rf: (rf / f) if f else None)(
                        sum(r["decomp_c2_tf32off"]["1e-2"]["fail_default"] for r in ok),
                        sum(r["decomp_c2_tf32off"]["1e-2"]["ref_induced"] for r in ok))},
        "reference_constant_s84": 0.99807,
        "verdict": verdict,
        "instrument_defects": instr,
        "instrument_ok": len(instr) == 0,
        "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__,
    }
    json.dump(summary, open(SUMOUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:5000])


if __name__ == "__main__":
    main()
