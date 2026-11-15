#!/usr/bin/env python
"""s109b_draw_boundary.py — boundary-adjacent verdict stability + dgrad repeat magnitude.
Prereg: lab/s109b_prereg.md, committed before any row is computed."""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/kb-lab"))
import torch
from s109_draw_band import load_task, run_ref, stats_vs, profile_fn, classify, CHUNK, STRICT, KB

LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s109b_draw_boundary.jsonl")
SUM = os.path.join(RES, "s109b_summary.json")
S107 = os.path.join(RES, "s107_rekey_map.jsonl")
S107B = os.path.join(RES, "s107b_rekey_rest.jsonl")
NDRAW = 16
NDRAW_D1 = 8

TARGETS = [(2, 77, NDRAW), (1, 72, NDRAW_D1), (1, 77, NDRAW_D1)]


def recorded():
    rec = {}
    for path in (S107, S107B):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            rec[(r["level"], r["problem_id"])] = r
    return rec


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    R = recorded()
    rows = []
    t0 = time.time()
    k_anchor, k_tie = [], []
    with open(OUT, "a") as f:
        for lvl, pid, nd in TARGETS:
            r = R[(lvl, pid)]
            path = os.path.join(KB, "KernelBench", "level%d" % lvl, r["name"])
            mod = load_task(path)
            for d in range(nd):
                tw = time.time()
                rec = dict(level=lvl, problem_id=pid, name=r["name"], op_class=r["op_class"],
                           recorded_rho_B_strict=r["B_strict"]["rho_ew"], draw=d, seed=d)
                try:
                    torch.manual_seed(d)
                    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                    inputs = mod.get_inputs()
                    model = mod.Model(*init).cuda().eval()
                    ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                    ev = profile_fn(lambda: run_ref(model, ins, torch.float32, (True, True)))
                    out_b = run_ref(model, ins, torch.float32, (True, True))
                    out_x = run_ref(model, ins, torch.float64, (False, False))
                    if out_x is None or out_b is None:
                        raise RuntimeError("missing output")
                    rec["label"], rec["top1"] = classify([e[0] for e in ev])
                    rec["strict"] = stats_vs(out_b, out_x, STRICT)
                    rec["ref_max_abs"] = float(out_x.abs().max().item())
                    rec["u_strict"] = rec["ref_max_abs"] / STRICT
                    if d == 0:
                        rec["k_anchor_rel"] = 0.0  # anchor self-check done below for first target only
                        rr = rec["recorded_rho_B_strict"]
                        rec["tie_out_rel"] = abs(rec["strict"]["rho_ew"] - rr) / rr if rr else None
                    # repeats for the dgrad leg
                    if nd == NDRAW_D1:
                        out_b2 = run_ref(model, ins, torch.float32, (True, True))
                        rec["d1_bit_identical"] = bool(torch.equal(out_b, out_b2))
                        mx, rf, r2 = 0.0, out_b.reshape(-1), out_b2.reshape(-1)
                        n, i = rf.numel(), 0
                        while i < n:
                            j = min(n, i + CHUNK)
                            mx = max(mx, float((rf[i:j].double() - r2[i:j].double()).abs().max().item()))
                            i = j
                        amax = float(out_x.abs().max().item())
                        rec["d1_delta_max"] = mx
                        rec["d1_delta_rho_repeat"] = mx / (STRICT + STRICT * amax)
                        del out_b2
                    if (lvl, pid) == TARGETS[0][:2] and d == 0:
                        out_x2 = run_ref(model, ins, torch.float64, (False, False))
                        denom = float(out_x.abs().max().item()) or 1.0
                        rec["k_anchor_rel"] = float((out_x - out_x2).abs().max().item()) / denom
                        del out_x2
                    rec["status"] = "ok"
                    rec["wall_s"] = round(time.time() - tw, 2)
                    del model, ins, inputs, out_b, out_x
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec["status"] = "error"; rec["err"] = repr(e)[:200]
                    torch.cuda.empty_cache()
                f.write(json.dumps(rec) + "\n"); f.flush()
                rows.append(rec)
                print(lvl, pid, d, rec["status"], rec.get("label"),
                      (rec.get("strict") or {}).get("rho_ew"), rec.get("d1_delta_rho_repeat"),
                      rec.get("wall_s"), flush=True)

    b = [x for x in rows if (x["level"], x["problem_id"]) == (2, 77) and x.get("status") == "ok"]
    rhos = [x["strict"]["rho_ew"] for x in b]
    r_rate = sum(1 for v in rhos if v > 1) / max(len(rhos), 1)
    d1 = [x for x in rows if x.get("status") == "ok" and "d1_delta_rho_repeat" in x]
    tie = [x.get("tie_out_rel") for x in rows if x.get("draw") == 0 and x.get("tie_out_rel") is not None]
    ka = [x.get("k_anchor_rel") for x in rows if x.get("k_anchor_rel") is not None]
    instr_ok = (not ka or max(ka) <= 1e-9) and tie and max(tie) <= 0.05

    def st(xs):
        m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return dict(n=len(xs), mean=m, sd=sd, cv=sd / m if m else None, min=min(xs), max=max(xs))

    b1 = "INSTRUMENT_FAILED"
    if instr_ok:
        if 0 < r_rate < 1:
            b1 = "BRANCH_BOUNDARY_RATE"
        elif r_rate == 1:
            b1 = "BRANCH_BOUNDARY_STABLE_ABOVE"
        else:
            b1 = "BRANCH_BOUNDARY_STABLE_BELOW"
    b2 = dict(branch="BRANCH_REPEAT_MAGNITUDE",
              n_cells=len(d1),
              n_bit_identical=sum(1 for x in d1 if x.get("d1_bit_identical")),
              max_delta_rho=max([x["d1_delta_rho_repeat"] for x in d1], default=None),
              per_row={})
    for x in d1:
        key = "%d/%d" % (x["level"], x["problem_id"])
        b2["per_row"].setdefault(key, []).append(x["d1_delta_rho_repeat"])

    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   wall_s=round(time.time() - t0, 1), n_cells=len(rows),
                   n_ok=sum(1 for x in rows if x.get("status") == "ok"),
                   instrument_ok=instr_ok, k_anchor_rel=ka, tie_out=tie,
                   boundary_row=dict(id="2/77", rho=st(rhos) if rhos else None,
                                     exceed_rate=r_rate, labels=sorted({x.get("label") for x in b})),
                   branch_b1=b1, branch_b2=b2, rows=rows)
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("branch_b1", "branch_b2", "instrument_ok", "k_anchor_rel",
                                              "tie_out", "n_ok", "wall_s")}, indent=1))
    print(json.dumps(summary["boundary_row"], indent=1))


if __name__ == "__main__":
    main()
