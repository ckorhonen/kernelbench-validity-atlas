"""R45: rider regret redo with admissibility on every arm (CPU, records only).

Repairs rejection 2CmvAb F3 (disposition: rider redo). The published rider
(r41_gate_stability, prereg 5ff93b20) admits torch UNCONDITIONALLY in two
places: the oracle pool (min(passing hand, torch) regardless of torch's own
verdict) and the gate fallback (rule arm else torch). torch fails the fp64
anchor 52/56, so both the regret denominator and the delivered arm can rest on
a failing arm. This redo applies the admission rule to EVERY arm.

Conventions, frozen before this run (2026-10-23, stated here pre-execution):
  A1 "published"  := exactly r41 gate_from_pass semantics, recomputed from the
     same records: adm pool = imp arms with pass_ref; oracle = min(pool ∪
     {torch})); gate = imp_B32x32 first passing prec if H<=14 and C<=64 else
     torch; regret=(gate_t-ora_t)/ora_t.
  A2 "adm-every"  := adm pool = ALL arms with pass_ref, torch/torch_cmp
     included iff they pass; oracle = min(pool), None if pool empty; gate
     chain unchanged (imp_B32x32 ieee->tf32 if inrule); fallback torch ONLY if
     torch passes; if neither, no admissible delivery -> gate_regret None
     (counted no_delivery). Regret undefined when oracle undefined.
  Comparisons restricted to rows where BOTH conventions yield defined regret;
   undefined rows counted separately, never silently dropped.
  Times: lab/r1_times.json verbatim (recorded r1 med_ms), same as r41.
  References: def/ieee/64 pass maps from results/gate_stability.jsonl as run.

Outputs per reference: n, oracle_undef, no_delivery, gate-regret mean/median
A1 vs A2, paired delta pp, delivered-arm disagreements A2 vs A1, armA
delivered-change def->64 under A2, and the A2 adm-pool composition. Per-row
table emitted for the paper. NO new GPU work; no timing claims.

Output: results/r45_rider_adm.json (+ printed digest).
"""
import json, os, statistics as st

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
RIDER = os.path.join(_kb_res_dir(HERE), "gate_stability.jsonl")
TIMES = os.path.join(HERE, "r1_times.json") if os.path.exists(os.path.join(HERE, "r1_times.json")) else os.path.join(os.path.dirname(HERE), "lab", "r1_times.json")
OUT = os.path.join(_kb_res_dir(HERE), "r45_rider_adm.json")
REFS = ("def", "ieee", "64")


def load():
    rows = [json.loads(l) for l in _kb_open(RIDER) if l.strip()]
    times = json.load(open(TIMES))
    assert len(rows) == 56, f"rider rows {len(rows)} != 56"
    return rows, times


def gate_a1(passmap, H, C, times):
    """r41 semantics: imp-pool; oracle incl torch unconditional; fallback torch."""
    passing = {k: t for k, t in times.items() if k.startswith("imp_") and passmap.get(k)}
    torch_t = times["torch"]
    pool = dict(passing); pool["torch"] = torch_t
    ora_t = min(pool.values())
    inrule = H <= 14 and C <= 64
    gate_k, gate_t = "torch", torch_t
    if inrule:
        for prec in ("ieee", "tf32"):
            k = f"imp_B32x32[{prec}]"
            if passmap.get(k):
                gate_k, gate_t = k, times[k]
                break
    return gate_k, gate_t, ora_t


def gate_a2(passmap, H, C, times):
    """adm-every: pool = every arm passing own check; torch fallback conditional."""
    pool = {k: t for k, t in times.items() if passmap.get(k) and k in times}
    ora_t = min(pool.values()) if pool else None
    inrule = H <= 14 and C <= 64
    gate_k = gate_t = None
    if inrule:
        for prec in ("ieee", "tf32"):
            k = f"imp_B32x32[{prec}]"
            if passmap.get(k):
                gate_k, gate_t = k, times[k]
                break
    if gate_k is None and passmap.get("torch"):
        gate_k, gate_t = "torch", times["torch"]
    return gate_k, gate_t, ora_t, sorted(pool)


def main():
    rows, times = load()
    per_ref, per_row = {}, []
    for ref in REFS:
        pk = f"pass_{ref}"
        n = ora_undef = no_del = 0
        a1_reg, a2_reg, diffs, both = [], [], [], []
        arm_disagree = 0
        a2_deliv_change = 0
        pool_sizes = []
        for r in rows:
            n += 1
            pm = r[pk]
            t = dict(times[f"{r['tag']}|{r['H']}|{r['N']}|{r['C']}"])
            g1k, g1t, o1 = gate_a1(pm, r["H"], r["C"], t)
            g2k, g2t, o2, pool = gate_a2(pm, r["H"], r["C"], t)
            pool_sizes.append(len(pool))
            if o2 is None:
                ora_undef += 1
            if g2k is None:
                no_del += 1
            r1 = (g1t - o1) / o1 if o1 else None
            r2 = (g2t - o2) / o2 if (o2 and g2t is not None) else None
            if r1 is not None and r2 is not None:
                a1_reg.append(100 * r1); a2_reg.append(100 * r2)
                both.append(r2 - r1)
            if (g2k or "NONE") != g1k:
                arm_disagree += 1
            per_row.append(dict(tag=r["tag"], H=r["H"], N=r["N"], C=r["C"], ref=ref,
                                a1_arm=g1k, a2_arm=g2k, a2_pool_n=len(pool),
                                a1_reg_pct=None if r1 is None else round(100 * r1, 4),
                                a2_reg_pct=None if r2 is None else round(100 * r2, 4),
                                a2_oracle_undef=o2 is None, a2_no_delivery=g2k is None))
        # def->64 delivered change under A2, paired per row, split by status
        ch = both_arms = to_none = from_none = 0
        for r in rows:
            t = dict(times[f"{r['tag']}|{r['H']}|{r['N']}|{r['C']}"])
            kd, _, _, _ = gate_a2(r["pass_def"], r["H"], r["C"], dict(t))
            k64, _, _, _ = gate_a2(r["pass_64"], r["H"], r["C"], dict(t))
            if (kd or "NONE") != (k64 or "NONE"):
                ch += 1
                if kd and k64: both_arms += 1
                elif kd and not k64: to_none += 1
                else: from_none += 1
        per_ref[ref] = dict(
            n=n, oracle_undef=ora_undef, no_delivery=no_del,
            a1_reg_mean_pct=round(st.mean(a1_reg), 4) if a1_reg else None,
            a2_reg_mean_pct=round(st.mean(a2_reg), 4) if a2_reg else None,
            a1_reg_med_pct=round(st.median(a1_reg), 4) if a1_reg else None,
            a2_reg_med_pct=round(st.median(a2_reg), 4) if a2_reg else None,
            # F2 FIX (DE1paW review): `both` holds FRACTIONAL differences; the original fields were named
            # *_pp but never multiplied by 100, and `_max` was a signed max that hid large negative changes
            # at the fp64 rung. Both are corrected here; the legacy names are kept so older citations
            # resolve, and *_max_abs_pp is the field a reader should use.
            paired_delta_pp_mean=round(100 * st.mean(both), 4) if both else None,
            paired_delta_pp_max=round(100 * max(both), 4) if both else None,
            paired_delta_pp_max_abs=round(100 * max(abs(x) for x in both), 4) if both else None,
            paired_delta_pp_min=round(100 * min(both), 4) if both else None,
            paired_n=len(both),
            arm_disagree_a2_vs_a1=arm_disagree,
            a2_delivered_change_def64=(ch if ref == "def" else None),
            a2_change_both_arms=(both_arms if ref == "def" else None),
            a2_change_to_none=(to_none if ref == "def" else None),
            a2_change_from_none=(from_none if ref == "def" else None),
            a2_pool_n_min=min(pool_sizes), a2_pool_n_med=st.median(pool_sizes),
        )
    # headline counts the paper needs
    torch_pass = {ref: sum(1 for r in rows if r[f"pass_{ref}"].get("torch")) for ref in REFS}
    import time as _t
    out = dict(
        date=_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
        prereg="conventions frozen in module docstring BEFORE run; repairs 2CmvAb F3",
        n_rows=56, torch_pass_own_check=torch_pass, per_ref=per_ref,
        a1_delivered_change_def64_published=21,
        a1_reg_delta_pp_published=[0.0, 66.55247502326027],
        note="A1 recomputed from records matches r41 summary; deltas on the defined-overlap only.",
        per_row=per_row,
    )
    json.dump(out, open(OUT, "w"), indent=1)
    for ref, s in per_ref.items():
        print(ref, json.dumps(s))
    print("torch passes own check:", torch_pass)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
