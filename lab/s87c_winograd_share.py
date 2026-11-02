#!/usr/bin/env python
"""s87c_winograd_share.py — does the reference-induced share constant hold in a non-TF32 engine class?

PREREG (frozen before the first run; do not edit after). Commit before executing on spark.

WHY. s87 measured the share on 17 constructed configurations outside the 54-row population: pooled 0.99850,
every extent band >= 0.9968 (V1_SHARE_GENERALIZES). s87b then labelled all 17 and returned CLASS_CONFINED: every
one dispatches to TF32_TENSOROP (xmma/cutlass s1688 fprop/dgrad). So s87's generalization is established across
shape and extent INSIDE one engine class, and the class axis is still untested. This producer runs the missing
cell: constructed shapes small enough in output extent and deep enough in channels that the reference dispatches
to WINOGRAD on this stack (A6/A7 observed winograd_nonfused 4x4 with rho 10.195 at exactly that regime), then
measures the same share decomposition there.

POPULATION (declared lattice, 8 configurations, conv2d only, k=3 s=1 p=1): the s87 lattice's large-extent shapes
all landed on TF32, so this lattice brackets the A7 winograd regime (H=W in {14,28}, C in {64,128,256}) and
includes two controls at H=W=56 that s87b already labelled TF32. No tuple overlaps s87's lattice except the
controls, which are declared as such and excluded from the winograd verdict.
  (N=2, C=64,  H=W=14, Cout=64)
  (N=4, C=64,  H=W=14, Cout=128)
  (N=2, C=128, H=W=14, Cout=64)
  (N=2, C=256, H=W=14, Cout=256)
  (N=2, C=64,  H=W=28, Cout=64)
  (N=4, C=128, H=W=28, Cout=128)
  (N=2, C=64,  H=W=56, Cout=64)    control (s87 #0, TF32)
  (N=2, C=128, H=W=56, Cout=128)   control (expected TF32)
Seeds 9101+i. Weights/inputs: framework defaults, torch.randn — identical construction to s87.

ARMS/PREDICATES: identical to s87 (A default fp32 flags, X fp64 anchor, C2 tf32-off candidate; evaluator-exact
b_e = atol + rtol*|C|; reference-induced := |C2-A| > b_e AND |A-X| > b_e). Engine labels: s75's frozen patterns,
imported verbatim, applied to a torch.profiler top-5 of the reference call (same protocol as s87b).

FROZEN VERDICT (evaluated in this order):
  WINOGRAD_SHARE_HELDS   iff >=3 configurations label to a non-TF32 class (WINOGRAD or FP32_SIMT_OR_OTHER) with
                         fail_default >= 1000, and their pooled reference-induced share is >= 0.95.
  WINOGRAD_SHARE_DIFFERS iff >=3 such configurations exist and the pooled share is < 0.95.
  NO_NON_TF32_DISPATCH   iff fewer than 3 configurations leave the TF32 class => the class axis stays untested on
                         this stack and is reported as such, not as a negative.
KILLS: K1 if fewer than 6 of 8 configurations run, no claim beyond the completed subset. K2 if the two declared
controls do not label TF32, the labelling harness is recorded as instrument-defective and the verdict is void.
WHAT THIS DOES NOT ESTABLISH: a single alternative class on one stack is not a class-independence law; the claim
is "the share holds (or does not hold) in the observed alternative class", with the class named.
"""
import importlib.util, json, os, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s87c_winograd_share.json")
CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 900
MIN_FAILS = 1000
SHARE_BAR = 0.95

POP = [
    ("conv", 2,  64,  14,  14,  64, False),
    ("conv", 4,  64,  14,  14, 128, False),
    ("conv", 2, 128,  14,  14,  64, False),
    ("conv", 2, 256,  14,  14, 256, False),
    ("conv", 2,  64,  28,  28,  64, False),
    ("conv", 4, 128,  28,  28, 128, False),
    ("conv", 2,  64,  56,  56,  64, True),
    ("conv", 2, 128,  56,  56, 128, True),
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


HERE = os.path.dirname(os.path.abspath(__file__))
S73B = load_mod("s73b", os.path.join(HERE, "s73b_oom_rerun.py"))
S75 = load_mod("s75", os.path.join(HERE, "s75_engine_class_s73pop.py"))


def top_kernels(fn, k=5):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages() if e.self_device_time_total > 0]
    evts.sort(key=lambda t: -t[1])
    return evts[:k]


def main():
    torch.cuda.init()
    t_start = time.time()
    prereg_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    rows = []
    for i, (op, n, cin, h, w, cout, control) in enumerate(POP):
        rec = {"idx": i, "op_class": op, "N": n, "Cin": cin, "H": h, "W": w, "Cout": cout,
               "out_extent": n * cout * h * w, "seed": 9101 + i, "declared_control": control}
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            rec["status"] = "deadline"
            rows.append(rec)
            continue
        try:
            torch.manual_seed(rec["seed"])
            model = torch.nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=True)
            x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
            model = model.to(device="cuda", dtype=torch.float32).eval()
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False

            def fn():
                with torch.no_grad():
                    return model(x)

            signal.alarm(CFG_ALARM_S)
            evts = top_kernels(fn)
            names = [e[0] for e in evts]
            prim, pname = S75.classify_primary(names)
            refined, rname = S75.classify_refined(names)
            rec.update({"engine_class_primary": prim, "engine_class_refined": refined,
                        "primary_matched_on": pname, "refined_matched_on": rname,
                        "kernel_top1": names[0] if names else None,
                        "kernels": [{"name": a, "self_us": b} for a, b in evts]})

            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                A = model(x)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                X = model.double()(x.double())
            model.float()
            with torch.no_grad():
                C2 = model(x)
            signal.alarm(0)

            d = S73B.decomp_stream(A, X, C2, 1e-4, 1e-2)
            rec["decomp_c2_tf32off"] = d
            for tag in ("1e-4", "1e-2"):
                s = d[tag]
                s["ref_over_frac"] = s["ref_over_budget"] / max(1, s["total"])
            rec["status"] = "ok"
            del A, X, C2, model, x
        except Timeout as e:
            rec["status"] = "timeout:%s" % e
        except Exception as e:
            rec["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:200])
        finally:
            signal.alarm(0)
        rows.append(rec)
        d = rec.get("decomp_c2_tf32off", {}).get("1e-4", {})
        print(json.dumps({"i": i, "cls": rec.get("engine_class_primary"), "top": (rec.get("kernel_top1") or "")[:60],
                          "ext": rec["out_extent"], "fail": d.get("fail_default"),
                          "share": d.get("ref_share_of_fails"), "refover": d.get("ref_over_frac"),
                          "st": rec.get("status")}), flush=True)
        torch.cuda.empty_cache()

    ok = [r for r in rows if r.get("status") == "ok"]
    controls = [r for r in ok if r["declared_control"]]
    non_tf32 = [r for r in ok if not r["declared_control"] and r.get("engine_class_primary") != "TF32_TENSOROP"
                and r["decomp_c2_tf32off"]["1e-4"]["fail_default"] >= MIN_FAILS]

    def agg(sub):
        fail = sum(r["decomp_c2_tf32off"]["1e-4"]["fail_default"] for r in sub)
        ref = sum(r["decomp_c2_tf32off"]["1e-4"]["ref_induced"] for r in sub)
        cand = sum(r["decomp_c2_tf32off"]["1e-4"]["cand_induced"] for r in sub)
        return {"n": len(sub), "failing_elements": fail, "ref_induced": ref, "cand_induced": cand,
                "share": (ref / fail) if fail else None,
                "share_min": min((r["decomp_c2_tf32off"]["1e-4"]["ref_share_of_fails"] for r in sub
                                  if r["decomp_c2_tf32off"]["1e-4"]["fail_default"] > 0), default=None),
                "max_cand_share": max(((r["decomp_c2_tf32off"]["1e-4"]["cand_induced"] or 0) /
                                       r["decomp_c2_tf32off"]["1e-4"]["fail_default"] for r in sub
                                       if r["decomp_c2_tf32off"]["1e-4"]["fail_default"] > 0), default=None),
                "classes": sorted({r.get("engine_class_primary") for r in sub})}

    verdict = None
    instr_ok = len(controls) == 2 and all(c.get("engine_class_primary") == "TF32_TENSOROP" for c in controls)
    if len(ok) < 6:
        verdict = "NARROWED_K1"
    elif not instr_ok:
        verdict = "INSTRUMENT_DEFECTIVE_K2"
    elif len(non_tf32) < 3:
        verdict = "NO_NON_TF32_DISPATCH"
    elif (agg(non_tf32)["share"] or 0) >= SHARE_BAR:
        verdict = "WINOGRAD_SHARE_HELDS"
    else:
        verdict = "WINOGRAD_SHARE_DIFFERS"

    summary = {
        "prereg": "docstring of this file, frozen pre-run",
        "prereg_sha256": prereg_sha,
        "population": {"constructed": len(POP), "ok": len(ok),
                       "statuses": {r["idx"]: r.get("status") for r in rows}},
        "class_distribution": {c: sum(1 for r in ok if r.get("engine_class_primary") == c)
                               for c in sorted({r.get("engine_class_primary") for r in ok if r.get("engine_class_primary")})},
        "non_tf32_usable": agg(non_tf32),
        "all_ok": agg(ok),
        "controls": {"n": len(controls), "instrument_ok": instr_ok,
                     "classes": [c.get("engine_class_primary") for c in controls]},
        "s87_constant": 0.9984983550131502,
        "verdict": verdict,
        "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__,
    }
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
