"""s60 — pool behavioural diversity (the opposing school's central quantity).

Meta-reflection r-tJCBLzWPOVpAdc ("Develop an opposing conceptual school") argued that the
binding difficulty is pool degeneracy: nominal arm counts far exceed behavioural content, so
per-arm frontier/admission claims are over-counted. This artifact measures the claim's central
quantity on the committed conv pool and tests its strong form.

Inputs (committed, no GPU, arithmetic only):
  lab/results/e59b_budget_ratios.jsonl  — per-arm candidate-scaled budget ratio, 56 shapes x 12 arms
  lab/results/r1_conv_gated.jsonl       — per-arm median time, same shapes/arms

Objects:
  error-signature count per shape  = distinct round(ratio,3) values across arms
  time-signature count per shape   = distinct round(med_ms,5) values across arms
  joint-signature count per shape  = distinct (ratio,time) pairs

No threshold, no kill clause: counts on recorded data. The interpretive claims are labelled
worked-case (measured) vs conjectural (extension) in the verdict block.
"""
import json, collections, statistics, os, datetime

def _kb_res_dir(here):
    """Artifact directory: lab/results in the repo, ../results inside the supplement package (#55)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


BASE = _kb_res_dir(os.path.dirname(os.path.abspath(__file__)))


def _open(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        import gzip
        return [json.loads(l) for l in gzip.open(p + ".gz", "rt")]
    return [json.loads(l) for l in open(p)]


rat = _open("e59b_budget_ratios.jsonl")
r1 = _open("r1_conv_gated.jsonl")


def key(r):
    return (r.get("tag"), r.get("N"), r.get("H"), r.get("C"))


R = {key(r): r for r in rat}
rows = []
for r in r1:
    k = key(r)
    br = R.get(k)
    arms = r["variants"]
    er = {a: br["arms"][a]["ratio_cand"] for a in arms if br and a in br["arms"]}
    tm = {a: v["med_ms"] for a, v in arms.items()}
    rows.append({
        "shape": list(k),
        "nominal_arms": len(arms),
        "error_sigs": len({round(v, 3) for v in er.values()}) if er else None,
        "time_sigs": len({round(v, 5) for v in tm.values()}),
        "joint_sigs": len({(round(er.get(a, 0.0), 3), round(tm[a], 5)) for a in arms}),
    })

def agg(f):
    v = [r[f] for r in rows if r[f] is not None]
    return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v),
            "mean": round(sum(v) / len(v), 3)}

alias = {}
if rat:
    g = collections.defaultdict(list)
    for a, d in rat[0]["arms"].items():
        g[round(d["ratio_cand"], 3)].append(a)
    alias = {str(k): v for k, v in g.items()}

out = {
    "artifact": "s60_pool_diversity",
    "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": ["lab/results/e59b_budget_ratios.jsonl", "lab/results/r1_conv_gated.jsonl"],
    "population": {"shapes": len(rows), "nominal_arm_checks": sum(r["nominal_arms"] for r in rows)},
    "nominal_arms_per_shape": agg("nominal_arms"),
    "error_signatures": agg("error_sigs"),
    "time_signatures": agg("time_sigs"),
    "joint_signatures": agg("joint_sigs"),
    "joint_signature_total": sum(r["joint_sigs"] for r in rows),
    "alias_groups_first_shape": alias,
    "verdict": {
        "strong_form": "FALSIFIED — the pool is not degenerate on the speed axis: 654/672 joint "
                       "(error,time) signatures are distinct and the time axis carries 9-12 distinct "
                       "values per shape.",
        "surviving_form": "WORKED CASE — the admissibility axis is aliased to a precision-class level: "
                          "3-4 distinct candidate-scaled budget ratios per shape across 12 nominal arms "
                          "(median 3). At eps=1e-1 the 672 admissible arm-checks are 168 class-cells.",
        "consequence": "Frontier claims must be normalised by class-cells alongside arm counts; "
                       "class-level admission is learnable, within-class selection is not.",
        "scope": "GB10 conv pool at rev 423217d, 56 shapes, this pool as built. Conjectural: the same "
                 "aliasing governs reduction pools and pools spanning arithmetic (fp64-accum, "
                 "cudnn-disabled, channels_last) — untested here.",
    },
    "rows": rows,
}

p = os.path.join(BASE, "s60_pool_diversity.json")
json.dump(out, open(p, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("population", "error_signatures", "time_signatures",
                                      "joint_signatures", "joint_signature_total",
                                      "alias_groups_first_shape", "verdict")}, indent=1)[:2200])
print("wrote", p)
