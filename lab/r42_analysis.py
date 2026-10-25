"""R42 corrected analysis: union dataset, corrected oracle, honest gate rule, paired d_i.

Repairs rejection O4hnCx V1-V3 (dispositions 1, part of 2):
  - Union dataset = r1_conv_gated.jsonl rows EXCEPT the 4 grid-garbled rows
    (w4 N2H3C5, N2H3C12; a2 N2H4C9, N2H4C20) PLUS r42_gridfix.jsonl (the same
    tags/shapes rerun on the frozen (N,H,C) grids, verified against
    w4_heldout.py:76 / a2_day2.py:61).
  - Corrected oracle per row: min med_ms over {torch} + {passing hand arms}
    (V3: the old oracle excluded torch). Recomputed regrets: gate (rule arm if
    passing else torch, NO min-guard -- stated rule, per disposition 1 option B),
    torch, torch_cmp, and a secondary min-guard gate for comparison only.
  - Paired delivery d_i = (t_cmp - t_gate)/t_oracle per row; mean/t/n per tag
    (V2 convention) -- replaces significance claims.
  - Reference-realization scenario check (prereg: reflect_2026-10-22.md session 2
    turn 4; rider b5504bd else-branch): delivery arm re-evaluated under the ieee
    reference passing sets (pass32 columns) -- flip counts def->iee reported next
    to the rider's def->64. Scenario intervals, not bootstrap.

Output: results/r42_analysis.json (+ printed digest). CPU only; no GPU.
"""
import json, os, math

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
R1 = os.path.join(_kb_res_dir(HERE), "r1_conv_gated.jsonl")
R42 = os.path.join(_kb_res_dir(HERE), "r42_gridfix.jsonl")
OUT = os.path.join(_kb_res_dir(HERE), "r42_analysis.json")

# wrong rows: (tag, N, H, C) as stored in r1 jsonl (N,H,C keys)
WRONG = {("w4", 2, 3, 5), ("w4", 2, 3, 12), ("a2", 2, 4, 9), ("a2", 2, 4, 20)}

def load_union():
    rows, dropped = [], []
    for line in _kb_open(R1):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["tag"], r["N"], r["H"], r["C"])
        (dropped if key in WRONG else rows).append((key, r, "r1"))
    for line in _kb_open(R42):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(((r["tag"], r["N"], r["H"], r["C"]), r, "r42gridfix"))
    return rows, dropped

def corrected(r):
    V = r["variants"]
    torch_t = V["torch"]["med_ms"]
    passing = {k: v for k, v in V.items()
               if k.startswith("imp_") and v.get("pass_1e4") and "med_ms" in v}
    ora_t = min([torch_t] + [v["med_ms"] for v in passing.values()])
    ora_src = "torch" if torch_t <= min([v["med_ms"] for v in passing.values()], default=1e9) \
        else min(passing, key=lambda k: passing[k]["med_ms"])
    # gate: rule arm if passing else torch; NO min-guard (declared rule)
    inrule = r["H"] <= 14 and r["C"] <= 64
    gate_k, gate_t = "torch", torch_t
    chain = []
    if inrule:
        for prec in ("ieee", "tf32"):
            k = f"imp_B32x32[{prec}]"
            chain.append((k, bool(k in passing)))
            if k in passing:
                gate_k, gate_t = k, V[k]["med_ms"]
                break
    else:
        chain = [("torch", True)]
    # secondary: min-guard gate (best passing arm incl torch), comparison only
    mg_pool = dict(passing); mg_pool["torch"] = V["torch"]
    mg_k = min(mg_pool, key=lambda k: mg_pool[k]["med_ms"])
    # ieee-reference scenario: passing set under pass32
    passing32 = {k: v for k, v in V.items()
                 if k.startswith("imp_") and v.get("pass32") and "med_ms" in v}
    g32_k, g32_t = "torch", torch_t
    if inrule:
        for prec in ("ieee", "tf32"):
            k = f"imp_B32x32[{prec}]"
            if k in passing32:
                g32_k, g32_t = k, V[k]["med_ms"]
                break
    return dict(
        tag=r["tag"], N=r["N"], H=r["H"], C=r["C"], src=r["_src"],
        torch_t=torch_t, cmp_t=V["torch_cmp"].get("med_ms"),
        gate_arm=gate_k, gate_t=gate_t, gate_chain=chain,
        min_guard_arm=mg_k, min_guard_t=mg_pool[mg_k]["med_ms"],
        oracle_arm=ora_src, oracle_t=ora_t, n_passing=len(passing),
        n_passing_ieeeref=len(passing32), gate_ieeeref_arm=g32_k, gate_ieeeref_t=g32_t,
        gate_regret=(gate_t - ora_t) / ora_t,
        min_guard_regret=(mg_pool[mg_k]["med_ms"] - ora_t) / ora_t,
        torch_regret=(torch_t - ora_t) / ora_t,
        cmp_regret=(V["torch_cmp"]["med_ms"] - ora_t) / ora_t
        if "med_ms" in V["torch_cmp"] else None,
        d_i=(V["torch_cmp"]["med_ms"] - gate_t) / ora_t
        if "med_ms" in V["torch_cmp"] else None,
    )

def paired_t(ds):
    n = len(ds)
    m = sum(ds) / n
    var = sum((x - m) ** 2 for x in ds) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    return dict(n=n, mean=m, sd=sd, t=m / (sd / math.sqrt(n)) if sd > 0 and n > 1 else None)

def main():
    rows, dropped = load_union()
    for _, r, src in rows:
        r["_src"] = src
    assert len(rows) == 56, f"union must be 56 rows, got {len(rows)}"
    assert len(dropped) == 4, f"must drop exactly the 4 wrong rows, got {len(dropped)}"
    cor = [corrected(r) for _, r, _ in rows]

    by_tag = {}
    for c in cor:
        by_tag.setdefault(c["tag"], []).append(c)
    tag_stats = {}
    for tag, cs in by_tag.items():
        ds = [c["d_i"] for c in cs if c["d_i"] is not None]
        g = [100 * c["gate_regret"] for c in cs]
        mg = [100 * c["min_guard_regret"] for c in cs]
        tr = [100 * c["torch_regret"] for c in cs]
        # delivered-arm change under ieee-reference scenario (vs default-ref gate)
        flips_iee = sum(1 for c in cs if c["gate_ieeeref_arm"] != c["gate_arm"])
        tag_stats[tag] = dict(
            n=len(cs),
            d_i=paired_t(ds), d_i_pct_mean=100 * paired_t(ds)["mean"],
            gate_regret_pct_mean=sum(g) / len(g),
            min_guard_regret_pct_mean=sum(mg) / len(mg),
            torch_regret_pct_mean=sum(tr) / len(tr),
            gate_beats_torch=sum(1 for c in cs if c["gate_t"] < c["torch_t"]),
            gate_beats_cmp=sum(1 for c in cs if c["cmp_t"] is not None and c["gate_t"] < c["cmp_t"]),
            max_gate_regret_pct=max(g),
            delivered_flips_ieeeref=flips_iee,
            oracle_is_torch=sum(1 for c in cs if c["oracle_arm"] == "torch"),
        )
    # guard statement + per-row table for the paper
    table = [dict(tag=c["tag"], N=c["N"], H=c["H"], C=c["C"], src=c["src"],
                  gate=c["gate_arm"], gate_t=round(c["gate_t"], 6),
                  oracle=c["oracle_arm"], oracle_t=round(c["oracle_t"], 6),
                  gate_regret_pct=round(100 * c["gate_regret"], 4),
                  d_i_pct=round(100 * c["d_i"], 4) if c["d_i"] is not None else None,
                  n_pass=c["n_passing"], n_pass32=c["n_passing_ieeeref"])
             for c in cor]
    out = dict(
        date=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        n_rows=len(cor), dropped_wrong_rows=[list(k) for k in dropped],
        gate_rule="imp_B32x32 first passing precision (ieee,tf32) if H<=14&C<=64 else torch; NO min-guard; regrets vs corrected oracle min(passing hand, torch)",
        by_tag=tag_stats, per_row=table,
        scenario_note="ieeeref flips are scenario columns (pass32), prereg reflect_2026-10-22 s2t4; rider 8eee9c5 has def->64",
    )
    json.dump(out, open(OUT, "w"), indent=1)
    for t, s in tag_stats.items():
        print(t, json.dumps(s, indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
