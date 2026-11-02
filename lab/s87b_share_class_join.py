#!/usr/bin/env python
"""s87b_share_class_join.py — is s87's generalized share a TF32-tensorop artifact, or class-independent?

PREREG (frozen before the first run; do not edit after). Commit before executing on spark.

WHY. s87 measured the reference-induced share on 17 constructed conv/convT configurations outside the 54-row
population: pooled 0.99850 over 153.4M failing elements, every extent band >= 0.9968, verdict
V1_SHARE_GENERALIZES. That result has one unclosed hole: s87 never profiled the reference call, so its
"generalizes" could mean nothing more than "the same engine was dispatched on every constructed shape". The
s84 constant is annotated TF32_TENSOROP for 47/54 rows, and s75b already showed that class membership does not
set the magnitude; the open question here is the opposite one, whether the SHARE survives a change of class.
This producer supplies the labels s87 lacks and performs the predeclared join.

POPULATION: exactly s87's 17 configurations (same lattice, same seeds 8701+i, rebuilt identically), so the join
is row-for-row by idx. Nothing is added or dropped.

METHOD: rebuild each configuration, set the benchmark's shipped default flags (cudnn allow_tf32 True, matmul
allow_tf32 False — the s87 A arm), run the reference forward once under torch.profiler with CUDA activity, take
the top-5 kernels by self device time, and apply s75's frozen ordered patterns VERBATIM (imported from
s75_engine_class_s73pop.py, not re-typed) for engine_class_primary, plus s75's refined descriptive classifier.

PREDECLARED READS (no threshold invented after the run):
  R1 class distribution over the 17 configurations, and the top-1 kernel names.
  R2 join: pooled and per-class reference-induced share, taken from s87's committed jsonl, keyed by idx.
  VERDICT (frozen, evaluated in this order):
    SHARE_CLASS_INDEPENDENT  iff >=2 primary classes are present AND every class with n>=3 has share >= 0.95.
    CLASS_CONFINED           iff all 17 configurations label to a single primary class (the generalization is
                             then stated as within-class only, and s84's class annotation is not a restriction).
    CLASS_DEPENDENT          iff some class with n>=3 has share < 0.95 (the constant is class-scoped).
KILL: K1 if fewer than 80% of the configurations yield a profiler label, the labelled subset is reported and no
claim is made beyond it.
LIMITS declared: one profiler pass per configuration, no timing claims (profiler-instrumented runs only), labels
are observations of this GB10/cudnn stack at rev 423217d, and the join inherits s87's constructed-shape scope.
"""
import importlib.util, json, os, signal, time, hashlib
import torch

LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s87b_share_class_join.json")
S87 = os.path.join(RES, "s87_fresh_shape_share.jsonl")
CFG_ALARM_S = 300
GLOBAL_DEADLINE_S = 900
SHARE_BAR = 0.95

# s87's lattice, verbatim
POP = [
    ("conv",            2,  64,  56,  56,  64),
    ("conv",            2,  16, 256, 256,  32),
    ("conv",            1, 256, 160, 160, 128),
    ("conv",            4, 128,  64,  64, 128),
    ("conv",            2,  32, 224, 224,  64),
    ("conv",            8,  64, 112, 112, 128),
    ("conv",            8, 128,  96,  96,  96),
    ("conv",            2,  64, 256, 256,  64),
    ("conv",            8, 128, 128, 128, 128),
    ("conv",            8,  32,  96,  96,  96),
    ("conv",            8,  64, 224, 224, 256),
    ("conv",            4,  64, 320, 320, 256),
    ("conv_transpose",  8,  64, 112, 112,  64),
    ("conv_transpose",  2,  32, 224, 224,  16),
    ("conv_transpose",  8, 256,  96,  96, 256),
    ("conv_transpose",  8,  64, 224, 224, 128),
    ("conv_transpose",  4,  64, 320, 320, 256),
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


S75 = load_mod("s75", os.path.join(os.path.dirname(os.path.abspath(__file__)), "s75_engine_class_s73pop.py"))


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
    shares = {json.loads(l)["idx"]: json.loads(l) for l in open(S87)}

    rows = []
    for i, (op, n, cin, h, w, cout) in enumerate(POP):
        rec = {"idx": i, "op_class": op, "N": n, "Cin": cin, "H": h, "W": w, "Cout": cout}
        if time.time() - t_start > GLOBAL_DEADLINE_S:
            rec["status"] = "deadline"
            rows.append(rec)
            continue
        try:
            torch.manual_seed(8701 + i)
            if op == "conv":
                model = torch.nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=True)
            else:
                model = torch.nn.ConvTranspose2d(cin, cout, 3, stride=1, padding=1, bias=True)
            x = torch.randn(n, cin, h, w, device="cuda", dtype=torch.float32)
            model = model.to(device="cuda", dtype=torch.float32).eval()
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False

            def fn():
                with torch.no_grad():
                    return model(x)

            signal.alarm(CFG_ALARM_S)
            evts = top_kernels(fn)
            signal.alarm(0)
            names = [e[0] for e in evts]
            prim, pname = S75.classify_primary(names)
            refined, rname = S75.classify_refined(names)
            rec.update({"status": "ok", "engine_class_primary": prim, "primary_matched_on": pname,
                        "engine_class_refined": refined, "refined_matched_on": rname,
                        "kernel_top1": names[0] if names else None,
                        "kernels": [{"name": a, "self_us": b} for a, b in evts],
                        "n_kernels": len(names)})
        except Timeout as e:
            rec["status"] = "timeout:%s" % e
        except Exception as e:
            rec["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:200])
        finally:
            signal.alarm(0)
        s87 = shares.get(i, {}).get("decomp_c2_tf32off", {}).get("1e-4", {})
        rec["s87_fail_default"] = s87.get("fail_default")
        rec["s87_ref_induced"] = s87.get("ref_induced")
        rec["s87_cand_induced"] = s87.get("cand_induced")
        rec["s87_share"] = s87.get("ref_share_of_fails")
        rec["s87_band"] = shares.get(i, {}).get("band")
        rows.append(rec)
        print(json.dumps({"i": i, "cls": rec.get("engine_class_primary"), "top": rec.get("kernel_top1"),
                          "share": rec.get("s87_share"), "st": rec.get("status")}), flush=True)
        del model, x
        torch.cuda.empty_cache()

    labelled = [r for r in rows if r.get("status") == "ok"]
    by_cls = {}
    for r in labelled:
        c = r["engine_class_primary"]
        by_cls.setdefault(c, []).append(r)

    def agg(sub):
        fail = sum(r["s87_fail_default"] or 0 for r in sub)
        ref = sum(r["s87_ref_induced"] or 0 for r in sub)
        return {"n": len(sub), "failing_elements": fail, "ref_induced": ref,
                "share": (ref / fail) if fail else None,
                "share_min_of_configs": min((r["s87_share"] for r in sub if r["s87_share"] is not None), default=None),
                "max_cand_share": max(((r["s87_cand_induced"] or 0) / r["s87_fail_default"]
                                       for r in sub if (r["s87_fail_default"] or 0) > 0), default=None)}

    cls_agg = {c: agg(v) for c, v in by_cls.items()}
    verdict = None
    if len(labelled) < 0.8 * len(POP):
        verdict = "NARROWED_K1"
    elif len(by_cls) == 1:
        verdict = "CLASS_CONFINED"
    elif any(a["share"] is not None and a["share"] < SHARE_BAR for c, a in cls_agg.items() if a["n"] >= 3):
        verdict = "CLASS_DEPENDENT"
    elif all((a["share"] or 0) >= SHARE_BAR for c, a in cls_agg.items() if a["n"] >= 3) and \
            sum(1 for a in cls_agg.values() if a["n"] >= 3) >= 2:
        verdict = "SHARE_CLASS_INDEPENDENT"
    else:
        verdict = "CLASS_SINGLE_OR_THIN"

    summary = {
        "prereg": "docstring of this file, frozen pre-run",
        "prereg_sha256": prereg_sha,
        "population": {"constructed": len(POP), "labelled": len(labelled),
                       "statuses": {r["idx"]: r.get("status") for r in rows}},
        "class_distribution": {c: len(v) for c, v in by_cls.items()},
        "class_aggregates": cls_agg,
        "pooled": agg(labelled),
        "top1_names": sorted({r.get("kernel_top1") for r in labelled if r.get("kernel_top1")}),
        "verdict": verdict,
        "s87_verdict": "V1_SHARE_GENERALIZES",
        "wall_s": round(time.time() - t_start, 1), "torch": torch.__version__,
    }
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
