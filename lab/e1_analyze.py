"""E1 analysis: anchored-collapse test + atlas entry v0 (elementwise dispatch).

Anchors from calibration.json (independent of E1 data):
  c0 anchor  = single-shot launch overhead
  BW(B)      = copy ladder interpolation at matching bytes moved
Tests:
  T1 anchor-match: fitted fixed-cost c0_fit ~ c0_anchor (sanity of instrument)
  T2 bw-match: BW_eff from fit ~ copy ladder at large B (triad vs copy parity)
  T3 collapse: t - (c0_anchor + B/BW_copy(B)) residual -> ~0 for well-occupied
     configs at large B; structured residual = occupancy shortfall (derived).
  T4 atlas v0: per size, argmin config; dominance boundaries in bytes AND waves.
"""
import json, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))

def interp_bw(ladder_rows, bytes_needed):
    rows = sorted(ladder_rows, key=lambda r: r["bytes"])
    bs = [r["bytes"] for r in rows]; gs = [r["GBps"] for r in rows]
    if bytes_needed <= bs[0]: return gs[0]
    if bytes_needed >= bs[-1]: return gs[-1]
    for i in range(1, len(bs)):
        if bs[i] >= bytes_needed:
            x0, x1 = math.log2(bs[i-1]), math.log2(bs[i])
            f = (math.log2(bytes_needed) - x0) / (x1 - x0)
            return gs[i-1] * (1 - f) + gs[i] * f
    return gs[-1]

def fit_c0_bw(pts):
    # pts: list (bytes, t_ms); least squares on top-30% sizes: t = c0 + B/BW
    pts = sorted(pts)[-max(4, int(0.3 * len(pts))):]
    n = len(pts)
    sb = sum(b for b, _ in pts); st = sum(t for _, t in pts)
    sbb = sum(b * b for b, _ in pts); sbt = sum(b * t for b, t in pts)
    denom = n * sbb - sb * sb
    if denom == 0: return None
    # t = c0 + (1/BW) * B  -> slope m = 1/BW, intercept c0
    m = (n * sbt - sb * st) / denom
    c0 = (st - m * sb) / n
    if m <= 0: return None
    return {"c0_ms": c0, "BW_Gbps": 1e-6 / m}  # t[ms]=c0+B/BW; slope m[ms/B]; GB/s = 1e-6/m

def main():
    cal = json.load(open(os.path.join(HERE, "results", "calibration.json")))
    e1 = json.load(open(os.path.join(HERE, "results", "e1_triad.json")))
    c0_anchor_ms = cal["launch_overhead"]["single_shot_med_ms"]
    ladder = cal["copy_bandwidth_ladder"]["fp32"]

    by_cfg = {}
    for r in e1["rows"]:
        if "error" in r or "med_ms" not in r: continue
        key = json.dumps(r["cfg"], sort_keys=True) + "|" + r["rule"]
        by_cfg.setdefault(key, []).append(r)

    report = {"meta": e1["meta"], "c0_anchor_ms": c0_anchor_ms, "configs": {}}
    for key, rows in sorted(by_cfg.items()):
        pts = [(r["bytes_moved"], r["med_ms"]) for r in rows]
        fit = fit_c0_bw(pts)
        if not fit: continue
        # collapse residual with ANCHORED c0 and copy-BW curve
        res = []
        for r in rows:
            bw = interp_bw(ladder, r["bytes_moved"]) * 1e9  # B/s
            t_pred_ms = c0_anchor_ms + r["bytes_moved"] / bw * 1e3
            res.append({"n": r["n"], "t_ms": r["med_ms"], "t_pred_ms": round(t_pred_ms, 5),
                        "rel_resid": round((r["med_ms"] - t_pred_ms) / t_pred_ms, 4),
                        "waves": r.get("derived", {}).get("nominal_waves")})
        big = [x for x in res if x["n"] >= 2 ** 24]
        rms_big = math.sqrt(sum((x["rel_resid"]) ** 2 for x in big) / max(1, len(big)))
        # transition size: first n (descending scan) where anchored model within 20%
        trans = None
        for x in sorted(res, key=lambda x: x["n"]):
            if abs(x["rel_resid"]) <= 0.20:
                trans = x["n"]; break
        cfg = json.loads(key.split("|")[0])
        report["configs"][key.split("|")[1] + json.dumps(cfg, sort_keys=True)] = {
            "c0_fit_ms": round(fit["c0_ms"], 5),
            "BW_eff_Gbps": round(fit["BW_GBps"] if "BW_GBps" in fit else fit["BW_Gbps"], 1),
            "transition_n_at_20pct": trans,
            "resid_rms_large_n": round(rms_big, 4),
            "n_points": len(rows)}
    json.dump(report, open(os.path.join(HERE, "results", "e1_analysis.json"), "w"), indent=1)

    # console summary
    print("c0 anchor (calib single-shot): %.4f ms" % c0_anchor_ms)
    for k, v in report["configs"].items():
        print("%-58s c0_fit=%.4f BW=%7.1f trans_n=%s resid_rms=%.3f" %
              (k[:58], v["c0_fit_ms"], v["BW_eff_Gbps"], v["transition_n_at_20pct"], v["resid_rms_large_n"]))
    # atlas v0: dominance per size among triton flat configs + torch
    print("\nAtlas v0 (elementwise): argmin per size")
    by_n = {}
    for r in e1["rows"]:
        if "med_ms" not in r: continue
        by_n.setdefault(r["n"], []).append(r)
    for n in sorted(by_n):
        rows = by_n[n]
        best = min(rows, key=lambda r: r["med_ms"])
        print("n=%9d best=%-28s %8.4f ms (torch %8.4f)" %
              (n, best["rule"] + json.dumps(best.get("cfg", {}), sort_keys=True)[:12],
               best["med_ms"], next((r["med_ms"] for r in rows if r["rule"] == "torch_add"), float("nan"))))

if __name__ == "__main__":
    main()
