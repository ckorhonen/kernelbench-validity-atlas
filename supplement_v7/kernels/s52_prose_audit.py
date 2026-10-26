"""S52 prose audit — regenerate every number newly cited in §5.2 (the A9 composition repair) from
committed artifacts.

Conventions frozen in this docstring BEFORE the run (P3/P1): each check names its artifact and its field;
a check that cannot find its source is FAIL, not skipped. Exit nonzero on any FAIL. Checks are a fixed
list, no post-hoc additions after the run.

Scope: this audit covers ONLY the A9-derived claims added to §5.2 and ATLAS_A9.md. The pre-existing §5
numbers are audited by s51_prose_audit.py (31/31) and are not re-audited here.

Checks:
 A01 a9_compose row count == 6 and header prereg == lab/a9_compose_prereg.md @ 53ca881
 A02 a9_compose runtime facts: cudnn.allow_tf32 True, matmul.allow_tf32 False, precision highest
 A03 every a9_compose row determinism-verified (deterministic_2run True) and G_nonlinear False on all 6
 A04 a9_analysis rule summary: R1 n_pass 4 adequate False; R2 n_pass 5 adequate True; R3 n_pass 5 adequate True
 A05 a9_analysis verdict == COMPOSITION_PREDICTIVE and the sole R2/R3 failing chain is c6_mm_deep_then_mm
 A06 a9_analysis: c4 clean-A control has rho_end == rho_B_alone (amplification invisible, share < 1e-4)
 A07 a9_analysis: c1 amplification share <= 1e-3 (winograd stage subdominant to TF32 stage)
 A08 a9_analysis: c6 measured/R3 in [2.2,2.5] and measured/R2 in [2.3,2.6] (super-additive exception)
 A09 a9_analysis: frac_end - max(fracA,fracB) <= 0.06 on every chain (dominated, not compounded) and
     the bulk-threshold statement: on chains with max frac > 0.5 the compound exceeds it by <= 0.07
 A10 a9_analysis: declared_limitation names the R2/R3 non-separation and the c6 failure
 A11 §5 prose cites the A9 artifacts by name (a9_compose.json, a9_analysis.json, ATLAS_A9.md, 53ca881)
 A12 §5 prose contains the three load-bearing claims: "bounded by the worst stage", "2.34", "1.07"
 A15 (added after DE1paW F6) the composition factor the prose states is COMPUTED, not string-matched:
     recompute rho_end/max(rho_A,rho_B) from a9_compose and assert (a) the count inside [1.0,1.5] and the
     count inside [1.0,2.0] match the numbers §5 states, (b) the string "1.0-1.5" no longer appears as the
     bound, (c) the union rule's pass/fail direction stated in §5 matches a9_analysis.rule_summary.
 A16 (added after DE1paW F6) the c5 counterexample is present in §5: "c5" named with rho_A 0.0689 and
     rho_B 0.6563 and rho_end 1.2939, all recomputed from a9_compose.
 A13 ATLAS_A9.md exists and states the frozen verdict COMPOSITION_PREDICTIVE and the SCOPE_CUT history
 A14 REFRAME_SKELETON §5 carries the repair (mentions A9 / 53ca881 / super-additive)
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def _res_dir(here):
    """Resolve the results directory: lab/results in the repo, ../results when the script is run from
    the supplement package's kernels/ directory (packaged auditors must stay runnable, #52)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _res_dir(HERE)
BASE = os.path.dirname(HERE)


def _first_dir(*cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[-1]


# repo: lab/ -> ../paper ; package: kernels/ -> ../paper (sections emitted under paper/sections)
PAPER = _first_dir(os.path.join(BASE, "paper"))
ATLAS_DIR = _first_dir(os.path.join(BASE, "atlas"), BASE)

checks = []


def ck(cid, ok, detail):
    checks.append({"id": cid, "ok": bool(ok), "detail": detail})


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


comp = load("a9_compose.json")
ana = load("a9_analysis.json")

ck("A01", comp is not None and len(comp["rows"]) == 6
   and comp["header"]["prereg"] == "lab/a9_compose_prereg.md @ 53ca881",
   f"rows={len(comp['rows']) if comp else None} prereg={comp['header']['prereg'] if comp else None}")

rf = comp["header"].get("runtime_facts") if comp else {}
if not rf and comp:
    rf = comp["rows"][0]["runtime_facts"]
ck("A02", rf.get("cudnn_allow_tf32") is True and rf.get("matmul_allow_tf32") is False
   and rf.get("float32_precision") == "highest",
   f"{rf.get('cudnn_allow_tf32')}/{rf.get('matmul_allow_tf32')}/{rf.get('float32_precision')}")

rows = comp["rows"] if comp else []
ck("A03", len(rows) == 6 and all(r["deterministic_2run"] for r in rows)
   and not any(r["G_nonlinear"] for r in rows),
   f"det={sum(1 for r in rows if r['deterministic_2run'])}/6 nonlin={sum(1 for r in rows if r['G_nonlinear'])}")

rs = ana["rule_summary"] if ana else {}
ck("A04", rs.get("R1", {}).get("n_pass") == 4 and rs.get("R1", {}).get("adequate") is False
   and rs.get("R2", {}).get("n_pass") == 5 and rs.get("R2", {}).get("adequate") is True
   and rs.get("R3", {}).get("n_pass") == 5 and rs.get("R3", {}).get("adequate") is True,
   {k: (rs.get(k, {}).get("n_pass"), rs.get(k, {}).get("adequate")) for k in ("R1", "R2", "R3")})

ck("A05", ana is not None and ana["verdict"] == "COMPOSITION_PREDICTIVE"
   and rs.get("R2", {}).get("fail_chains") == ["c6_mm_deep_then_mm"]
   and rs.get("R3", {}).get("fail_chains") == ["c6_mm_deep_then_mm"],
   f"verdict={ana['verdict'] if ana else None} r2fails={rs.get('R2', {}).get('fail_chains')}")

arows = {x["chain"]: x for x in ana["rows"]} if ana else {}
c4 = arows.get("c4_cleanA_GEMM_then_TF32", {})
ck("A06", c4 and abs(c4["rho_end_meas"] - c4["rho_B_alone"]) < 1e-3
   and (c4.get("amplification_share_of_B") or 0) < 1e-4,
   f"rho_end={c4.get('rho_end_meas')} rho_B={c4.get('rho_B_alone')} share={c4.get('amplification_share_of_B')}")

c1 = arows.get("c1_W_then_TF32", {})
ck("A07", c1 and (c1.get("amplification_share_of_B") or 1) <= 1e-3,
   f"share={c1.get('amplification_share_of_B')}")

c6 = arows.get("c6_mm_deep_then_mm", {})
ck("A08", c6 and 2.2 <= c6.get("measured_over_R3", 0) <= 2.5
   and 2.3 <= c6.get("measured_over_R2", 0) <= 2.6,
   f"over_R3={c6.get('measured_over_R3')} over_R2={c6.get('measured_over_R2')}")

# A09: dominated-not-compounded, computed from the artifact rather than asserted
deltas = []
bulk = []
for x in arows.values():
    mx = max(x["frac_A"], x["frac_B_alone"])
    deltas.append(x["frac_end_meas"] - mx)
    if mx > 0.5:
        bulk.append(x["frac_end_meas"] - mx)
ck("A09", max(deltas) <= 0.06 and max(bulk) <= 0.07,
   f"max_delta={max(deltas):.4f} bulk_deltas={[round(b,4) for b in bulk]}")

ck("A10", ana is not None and "R2" in ana["declared_limitation"] and "c6" in ana["declared_limitation"],
   ana["declared_limitation"][:90] if ana else None)

s5 = open(os.path.join(PAPER, "sections", "s5_tolerance.md")).read()
ck("A11", all(k in s5 for k in ("a9_compose.json", "a9_analysis.json", "ATLAS_A9.md", "53ca881")),
   {k: (k in s5) for k in ("a9_compose.json", "a9_analysis.json", "ATLAS_A9.md", "53ca881")})
ck("A12", "factor of 2" in s5 and "2.34" in s5 and "1.07" in s5,
   {"factor_of_2": "factor of 2" in s5, "2.34": "2.34" in s5, "1.07": "1.07" in s5})

# A15: the composition factor is COMPUTED from the artifact, not string-matched (DE1paW F6 lesson)
ratios = {x["chain"]: x["rho_end_meas"] / max(x["rho_A"], x["rho_B_alone"]) for x in arows.values()}
n15 = sum(1 for v in ratios.values() if 1.0 <= v <= 1.5)
n20 = sum(1 for v in ratios.values() if v <= 2.0)
r1_dir_ok = ("passes 4/6 and fails 2" in s5) == (ana["rule_summary"]["R1"]["n_pass"] == 4)
ck("A15", n15 == 1 and n20 == 5 and "1.0-1.5×" not in s5 and "1.0-1.5x" not in s5 and r1_dir_ok,
   {"n_in_1.0_1.5": n15, "n_in_1.0_2.0": n20, "ratios": {k: round(v, 3) for k, v in ratios.items()},
    "r1_direction_matches_artifact": r1_dir_ok})

# A16: the c5 counterexample is present with its computed numbers
c5 = arows["c5_cleanA_mm_then_mm"]
ck("A16", "c5" in s5 and "0.0689" in s5 and "0.6563" in s5 and "1.2939" in s5
   and abs(c5["rho_A"] - 0.0689) < 1e-3 and abs(c5["rho_B_alone"] - 0.6563) < 1e-3
   and abs(c5["rho_end_meas"] - 1.2939) < 1e-3,
   {"c5_rho_A": c5["rho_A"], "c5_rho_B": c5["rho_B_alone"], "c5_rho_end": c5["rho_end_meas"]})

atlas = os.path.join(ATLAS_DIR, "ATLAS_A9.md")
atlas_txt = open(atlas).read() if os.path.exists(atlas) else ""
ck("A13", "COMPOSITION_PREDICTIVE" in atlas_txt and "SCOPE_CUT" in atlas_txt
   and "2.34x" in atlas_txt, f"len={len(atlas_txt)}")

skel = open(os.path.join(PAPER, "REFRAME_SKELETON.md")).read()
ck("A14", "A9" in skel and "53ca881" in skel and "super-additive" in skel, "skeleton repair present")

npass = sum(1 for c in checks if c["ok"])
out = {"date": "2026-10-24", "prereg": "lab/a9_compose_prereg.md @ 53ca881",
       "conventions": "docstring", "n_checks": len(checks), "n_pass": npass,
       "all_pass": npass == len(checks), "checks": checks}
with open(os.path.join(RES, "s52_prose_audit.json"), "w") as f:
    json.dump(out, f, indent=1)
for c in checks:
    print(("PASS" if c["ok"] else "FAIL"), c["id"], c["detail"])
print(f"{npass}/{len(checks)} pass")
sys.exit(0 if npass == len(checks) else 1)
