"""#43 CPU counterfactuals (prereg'd reflect_2026-10-22 session 2; refresh_plan kill criteria).

Prereg'd readings (session 2, turns 1-5): chain-order permutation; anchor-profile
monotonicity {def,iee,64} + iee~64 agreement; cross-family scope on recorded b1/m3 pass flags.
Thresholds declared HERE before first run (the prereg fixed the readings, not numbers; these
are the operationalizations, fixed prior to executing the cell walks below):

  CF1 chain-order sensitivity "high"  := order-dependent delivered gate arm on >20% of
       m3_r42 conv rows, or >20% of rider (row,anchor) cells.
  CF2 profiles := monotone-dominant if >=60% of rider rows have nested pass sets across
       def->iee->64; "non-monotone/P5 branch" if >40% mixed; iee~64 exact-agreement <90%
       => keep 3-anchor lattice, else collapse to 2.
  CF3 cross-family flips "~0" := reduction (row,arm) pass-flip fraction <5% between
       recorded realizations (m3 day-1 errs vs m3_r42 adm, same 1e-4 deployed-reference
       contract); b1 recorded err ~1e-7 vs bar 1e-4 reported as margin, no lattice.

Decision mapping (refresh_plan, prereg'd): chain-sensitive => gate redesign, B paused;
cross-family ~0 AND chain-sensitive => B = conv-family section, per-family tables primary;
no nontrivial tau certifies any selector => stability axis measured-but-empty, P5.
Complement (default): B spine stands with reference-realization axis; interval scenario set
= observed lattice cells.

Run: python3 cf43.py   (CPU only, recorded fields; writes results/cf43.json)
"""
import json, itertools, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))

def _kb_res_dir(here):
    """Resolve the artifact directory: lab/results in the repo, ../results when run from the
    supplement package's kernels/ directory. Frozen packaging convention (#55)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


def _kb_open(path):
    """Open a plain or gzipped artifact; row-streams may ship gzipped in the package."""
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        import gzip as _gz
        return _gz.open(path + ".gz", "rt")
    raise FileNotFoundError(path)
RES = _kb_res_dir(HERE)
R = lambda n: json.load(_kb_open(os.path.join(RES, n)))

out = {"date": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"), "thresholds": "see module docstring (fixed pre-run)"}

# ---------------- CF1: chain-order permutation ----------------
# Gate chain (conv, m3 policy): fitted variant x PRECS in declared order; delivered =
# first passing; else torch(fallback). Order matters iff >=2 chain candidates pass.
# Permutation space here: 2! prec orders (single fitted variant imp_B32x32, meta h0/c0).
def gate_chain_candidates(row):
    v = "imp_B32x32"
    return [k for k in (f"{v}|ieee", f"{v}|tf32") if k in row["adm"]]

m3 = R("m3_r42.json")["rows"]
conv = [r for r in m3 if r["family"] == "conv"]
red = [r for r in m3 if r["family"] == "reduction"]

cf1 = {"m3_conv_rows": len(conv), "order_dependent": 0, "fired_variant": 0, "fallback_torch": 0}
order_dependent_rows = []
for r in conv:
    cands = gate_chain_candidates(r)
    passing = [k for k in cands if r["adm"][k]["passed"]]
    if len(passing) >= 2:
        cf1["order_dependent"] += 1
        order_dependent_rows.append((r["N"], r["H"], r["C"]))
    if r["gate_fires"] and r["arms"]["gate"]["chain"] and r["arms"]["gate"]["chain"][0] not in ("torch(fallback)",):
        cf1["fired_variant"] += 1
    if r["arms"]["gate"]["choice"] == "torch":
        cf1["fallback_torch"] += 1
cf1["order_dependent_frac"] = round(cf1["order_dependent"] / len(conv), 4)
cf1["order_dependent_rows"] = order_dependent_rows

# Rider side: same test per (row, anchor) on the fitted variant's precs.
rider = [json.loads(l) for l in _kb_open(os.path.join(RES, "gate_stability.jsonl"))]
cf1["rider_rows"] = len(rider)
cell = tot = 0
cells_order_dep = []
for row in rider:
    for anchor in ("def", "ieee", "64"):
        p = row.get(f"pass_{anchor}", {})
        tot += 1
        n_pass = sum(1 for k in ("imp_B32x32[ieee]", "imp_B32x32[tf32]") if p.get(k))
        if n_pass >= 2:
            cell += 1
            cells_order_dep.append((row["tag"], row["N"], row["H"], row["C"], anchor))
cf1["rider_cells"] = tot
cf1["rider_order_dependent"] = cell
cf1["rider_order_dependent_frac"] = round(cell / tot, 4) if tot else None
cf1["rider_order_dependent_cells"] = cells_order_dep
out["cf1_chain_order"] = cf1

# ---------------- CF2: anchor-profile monotonicity ----------------
ARM_KEYS = ["torch", "torch_cmp", "imp_B32x32[ieee]", "imp_B32x32[tf32]", "imp_B64x32[ieee]",
            "imp_B64x32[tf32]", "imp_B64x64[ieee]", "imp_B64x64[tf32]", "imp_B128x64[ieee]",
            "imp_B128x64[tf32]", "imp_B32x64K64[ieee]", "imp_B32x64K64[tf32]"]

def keyset(d): return frozenset(k for k, v in d.items() if v)

prof = collections.Counter()
iee64_same = 0
arm_level = collections.Counter()
for row in rider:
    s = {a: keyset(row.get(f"pass_{a}", {})) for a in ("def", "ieee", "64")}
    d, i, s64 = s["def"], s["ieee"], s["64"]
    if d == i == s64:
        prof["flat"] += 1
    elif d <= i <= s64 or d >= i >= s64:
        prof["nested"] += 1
    else:
        prof["mixed"] += 1
    if i == s64:
        iee64_same += 1
    for a in ARM_KEYS:
        pat = tuple(bool(row.get(f"pass_{anchor}", {}).get(a, False)) for anchor in ("def", "ieee", "64"))
        arm_level["".join("1" if x else "0" for x in pat)] += 1

n = len(rider)
cf2 = {
    "n_rows": n,
    "profile_counts": dict(prof),
    "profile_frac": {k: round(v / n, 4) for k, v in prof.items()},
    "iee64_exact_agreement": round(iee64_same / n, 4),
    "arm_level_patterns": dict(arm_level),
}
out["cf2_anchor_profile"] = cf2

# ---------------- CF3: cross-family scope ----------------
day1 = R("m3_curve.json")["rows"]
day1_red = {(r["K"], r["M"]): r for r in day1 if r["family"] == "reduction"}
r42_red = {(r["K"], r["M"]): r for r in red}
BAR = 1e-4
shared = set(day1_red) & set(r42_red)
tot_arm = flip = 0
flips = []
for key in sorted(shared):
    d1, d2 = day1_red[key], r42_red[key]
    for arm, e1 in d1["errs"].items():
        if arm not in d2["adm"]:
            continue
        p1 = e1 <= BAR
        p2 = bool(d2["adm"][arm]["passed"])
        tot += 1
        if p1 != p2:
            flip += 1
            flips.append({"K": key[0], "M": key[1], "arm": arm, "day1_err": e1, "r42_err": d2["adm"][arm]["err"]})
cf3 = {
    "note": "day1 m3 errs vs m3_r42 adm, contract 1e-4 vs deployed reference; same grids",
    "rows_matched": len(shared),
    "arms_compared": tot,
    "pass_flips": flip,
    "flip_frac": round(flip / tot, 4) if tot else None,
    "flip_examples": flips[:8],
    "b1_reduction_margin": "b1_gatezero_v2: 12/12 cells correct, err_factored_vs_ref <=1.2e-7 "
                           "vs 1e-4 bar (>=800x margin, single-anchor; no lattice recorded)",
    "conv_contrast": "rider armA: delivered 21/56, admissible-set 48/56, arm-verdict flips 341",
}
# day1 torch err not recorded separately; torch == reference on reduction (err 0 rows) - noted.
cf3["day1_torch_caveat"] = "day1 errs dict lacks 'torch' key on reduction rows; torch==ref assumed (err 0), excluded from counts"
out["cf3_cross_family"] = cf3

# ---------------- Decision mapping (prereg'd) ----------------
chain_sensitive = (cf1["order_dependent_frac"] > 0.20) or ((cf1["rider_order_dependent_frac"] or 0) > 0.20)
cross_family_quiet = (cf3["flip_frac"] is not None and cf3["flip_frac"] < 0.05)
monotone_dominant = cf2["profile_frac"].get("nested", 0) >= 0.60
mixed_heavy = cf2["profile_frac"].get("mixed", 0) > 0.40
nontrivial_tau = monotone_dominant and not chain_sensitive

decision = []
if chain_sensitive:
    decision.append("chain-sensitive>20%: gate redesign before any stability claim; B paused")
if cross_family_quiet and chain_sensitive:
    decision.append("cross-family flips ~0 AND chain-sensitive: B = conv-family section; per-family tables primary")
if not nontrivial_tau:
    decision.append("no nontrivial tau certifies selector: stability axis measured-but-empty; P5 write")
if not decision:
    decision.append("default: B spine stands with reference-realization axis; scenario set = observed lattice cells")
out["decision"] = {"chain_sensitive": chain_sensitive, "cross_family_quiet": cross_family_quiet,
                   "monotone_dominant": monotone_dominant, "mixed_heavy": mixed_heavy,
                   "nontrivial_tau": nontrivial_tau, "branches": decision}

with open(os.path.join(RES, "cf43.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
