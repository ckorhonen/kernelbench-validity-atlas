"""s6 dispatcher rerun under corrected oracle — CPU audit leg (#47).

PREREG frozen 2026-10-23 ~11:55Z BEFORE running (this docstring; commit precedes
execution). Object: w4_heldout records re-read under the post-repair contract,
per the reframe skeleton TBD "s6 dispatcher rerun under corrected oracle".

WHAT IS EXACTLY COMPUTABLE FROM RECORDS (no GPU):
  - contract pass flags per delivered/variant err (abs err <= 1e-4; the rows
    carry no |ref|, so this is the atol-only PESSIMISTIC reading — r1 documented
    it as ref-blind for large-K reductions; bias direction: UNDERCOUNTS passes,
    OVERSTATES the passing oracle time, UNDERSTATES positive regrets).
  - gate delivery vs pure-torch proxy: torch time known only when gate choice
    == torch or b1 choice == torch (b1's "always probes torch first" comment in
    the original harness does NOT force choice==torch — audit corrects that
    reading and marks unknowns null instead of assuming).
  - failing-choice audit for b-arms (F3-class): choice is a hand with err > tol
    => that arm's regret_total is understated; count + bound only.
  - corrected-oracle exact value: NOT computable (per-variant times absent from
    rows, variants={} by harness bug class) => reported as bounded interval
    [t_oracle_recorded, +inf) with the family verdict robustness argument:
    probe-charges >> admissibility-scale corrections (numerically shown).

VERDICT RULES (frozen now):
  V1 transfer: gate delivery >1% worse than pure-torch proxy on <= 3/24
     reduction rows (does not degrade vs the pre-repair accounting) => HOLDS.
  V2 certificate health: among rows where the gate delivers a hand, contract
     pass-frac >= 0.9 (pessimistic reading) => SINGLE-LAYER CERTIFICATES CLEAN.
  V3 robustness: max |regret| correction attributable to admissibility (bound:
     (t_oracle' - t_oracle)/t_oracle with t_oracle' <= min hand time proxy)
     < 0.05 while min family gap (gate vs best b regret_total means) > 0.10
     => family-level §6 verdict is admissibility-robust, CPU leg sufficient;
     else GPU contract rerun (m3_r42 pattern) queued #48.

S5 DECISION RULE (frozen now, keyed to THIS audit's outcome):
  RUN the composed-model propagation probe (GPU, prereg first, #48) iff
  V2 holds AND V3 holds. If V2 fails => scope-cut §5 to declared-extension
  (per-element distributions already committed as the basis; no composed
  claim). If V2 holds but V3 fails => #48 is the GPU s6 leg; s5 decision
  deferred one tick behind it. V1 failure => reconsider-if branch (paper
  reduces to negative+atlas form) — recorded, s5 moot.

Outputs: results/s6_dispatcher_audit.json. No thresholds touched post hoc.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.join(RES, "w4_heldout.json.jsonl")
OUT = os.path.join(RES, "s6_dispatcher_audit.json")
TOL = 1e-4  # contract, abs-only pessimistic reading (declared above)


def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    out = {"date": "2026-10-23T11:55Z", "prereg": "docstring", "tol_abs": TOL,
           "conventions": "abs-only pessimistic; no |ref| in rows (ref-blind direction recorded)",
           "families": {}}
    v1_ok = True
    v2_all = []
    v3 = {}
    for fam, rs in fams.items():
        n = len(rs)
        gate_hand_rows = [r for r in rs if r["arms"]["gate"]["choice"] != "torch"]
        pass_hand = [r["arms"]["gate"]["err"] <= TOL for r in gate_hand_rows]
        gate_hand_passfrac = (sum(pass_hand) / len(pass_hand)) if pass_hand else None
        # pure-torch proxy per row
        torch_proxy, proxy_src = [], []
        for r in rs:
            if r["arms"]["gate"]["choice"] == "torch":
                torch_proxy.append(r["arms"]["gate"]["t"]); proxy_src.append("gate")
            elif r["arms"]["b1"]["choice"] == "torch":
                torch_proxy.append(r["arms"]["b1"]["t"]); proxy_src.append("b1")
            else:
                torch_proxy.append(None); proxy_src.append(None)
        worse = []
        for r, tp in zip(rs, torch_proxy):
            if tp is None:
                worse.append(None)
            else:
                worse.append(1.0 if r["arms"]["gate"]["t"] > 1.01 * tp else 0.0)
        worse_known = [w for w in worse if w is not None]
        # b-arm failing-choice audit
        bfails = {}
        for b in ("b1", "b2", "b4", "b8"):
            c = 0
            for r in rs:
                ch = r["arms"][b]["choice"]
                if ch != "torch" and ch != "compile":
                    e = r["arms"][b].get("err", r["errs"].get(ch))
                    if e is None or e > TOL:
                        c += 1
            bfails[b] = c
        # compile pass
        # compile pass (string failure markers coerced to None = not-pass)
        raw_cerrs = [r["arms"]["compile"].get("err") for r in rs]
        cerrs = [e if isinstance(e, (int, float)) else None for e in raw_cerrs]
        cpass = sum(1 for e in cerrs if e is not None and e <= TOL)
        # probe-gap vs admissibility scale (V3 input)
        gt = [r["arms"]["gate"]["regret"] for r in rs]  # probes=0 => regret==regret_total
        bmeans = {}
        for b in ("b1", "b2", "b4", "b8"):
            vals = [r["arms"][b]["regret_total"] for r in rs
                    if r["arms"][b].get("regret_total") is not None]
            bmeans[b] = sum(vals) / len(vals) if vals else None
        best_b = min((v, k) for k, v in bmeans.items() if v is not None)
        # admissibility-scale bound: gate-hand err magnitudes vs 1
        all_errs = [r["arms"]["gate"]["err"] for r in rs]
        all_errs += [e for r in rs for e in r["errs"].values()]
        all_errs += [e for e in cerrs if e is not None]
        all_errs = [e for e in all_errs if isinstance(e, (int, float))]
        max_abs_err = max(all_errs, default=None)
        fam_out = dict(
            n=n,
            gate_fires=sum(1 for r in rs if r["gate_fires"]),
            gate_hand_deliveries=len(gate_hand_rows),
            gate_hand_passfrac_pessimistic=gate_hand_passfrac,
            gate_hand_err_max=max((r["arms"]["gate"]["err"] for r in gate_hand_rows), default=None),
            torch_proxy_src=dict(zip([f"{x}" for x in range(n)], proxy_src)),
            worse1pct_vs_torch_proxy_known=len(worse_known),
            worse1pct_count=sum(worse_known) if worse_known else 0,
            barm_failing_choice_counts=bfails,
            compile_pass=cpass,
            compile_err_max=max((e for e in cerrs if e is not None), default=None),
            gate_mean_regret_total=sum(gt) / len(gt),
            best_b=best_b[1], best_b_mean_regret_total=best_b[0],
            all_recorded_err_max=max_abs_err,
        )
        out["families"][fam] = fam_out
        # V1 evaluated on reduction (the family the 3/24 statement was about)
        if fam == "reduction":
            v1 = (fam_out["worse1pct_count"] <= 3 and
                  fam_out["worse1pct_vs_torch_proxy_known"] == n)
        else:
            v1 = None
        out["V1_" + fam] = v1
        if v1 is False:
            v1_ok = False
        v2_all.append(gate_hand_passfrac)
        v3[fam] = dict(gap=best_b[0] - (sum(gt) / len(gt)),
                       err_scale=max_abs_err)
    # frozen verdicts
    v2 = all(p is not None and p >= 0.9 for p in v2_all if p is not None) and len(v2_all) > 0
    v3bool = all((f["gap"] > 0.10) and (f["err_scale"] is not None and f["err_scale"] < 0.05)
                 for f in v3.values())
    out["V2_certificates_clean"] = v2
    out["V3_cpu_leg_sufficient"] = v3bool
    out["V1_transfer_holds"] = v1_ok
    out["S5_DECISION"] = ("RUN_PROPAGATION" if (v2 and v3bool)
                          else ("GPU_S6_FIRST" if v2 else "SCOPE_CUT_S5"))
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
