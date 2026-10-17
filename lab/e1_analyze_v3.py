"""E1 analysis v3: debt fix 'local-slope fitting' + per-config asymptotic intercept.

v1/v2 flaw: global single-slope LSQ across the copy-BW knee (259->220 GB/s) produced
meaningless c0. v3:
  c0_asym(config)  = median over top-k sizes of t_i - B_i/BW_copy(B_i)
                     (anchored on the measured two-slope copy ladder, not a global fit)
  BW_local         = pairwise slopes (dt/dB) between consecutive tail points
  residual v3      = t - (c0_asym + B/BW_copy(B)); systematic structure test:
                     Spearman sign trend of resid vs log2 n over the tail
Decision rule (meta-reflection 2026-10-17): tail rms <= 3% => intercept question closed
as operational def; > 5% systematic => two-slope story wrong, reopen as first-class object.
"""
import json, os, math, statistics
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

def spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j+1]] == v[s[i]]: j += 1
            rr = (i + j) / 2
            for k in range(i, j + 1): r[s[k]] = rr
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return num/den if den else 0.0

def main():
    cal = json.load(open(os.path.join(HERE, "results", "calibration.json")))
    e1 = json.load(open(os.path.join(HERE, "results", "e1_triad.json")))
    c0_launch = cal["launch_overhead"]["single_shot_med_ms"]
    ladder = cal["copy_bandwidth_ladder"]["fp32"]

    by_cfg = {}
    for r in e1["rows"]:
        if "error" in r or "med_ms" not in r: continue
        key = json.dumps(r["cfg"], sort_keys=True) + "|" + r["rule"]
        by_cfg.setdefault(key, []).append(r)

    report = {"meta": dict(e1["meta"], version=3), "c0_launch_ms": c0_launch, "configs": {}}
    verdicts = []
    for key, rows in sorted(by_cfg.items()):
        rows = sorted(rows, key=lambda r: r["n"])
        pts = [(r["bytes_moved"], r["med_ms"], r["n"]) for r in rows]
        tail = pts[-6:]
        # per-config asymptotic intercept, anchored on copy ladder
        c0s = [t - b / (interp_bw(ladder, b) * 1e9) * 1e3 for b, t, _ in tail]
        c0_asym = statistics.median(c0s)
        # local pairwise slopes in the tail -> BW_local
        bw_loc = []
        for (b0, t0, _), (b1, t1, _) in zip(tail, tail[1:]):
            if b1 > b0 and t1 > 0:
                s = (t1 - t0) / (b1 - b0)          # ms per B
                if s > 0: bw_loc.append(1e-6 / s)  # GB/s
        # residual v3 + structure test
        res = []
        for b, t, n in pts:
            bw = interp_bw(ladder, b) * 1e9
            t_pred = c0_asym + b / bw * 1e3
            res.append({"n": n, "rel_resid": (t - t_pred) / t_pred})
        big = [x for x in res if x["n"] >= 2 ** 24]
        rms_big = math.sqrt(sum(x["rel_resid"]**2 for x in big) / max(1, len(big)))
        rho = spearman([math.log2(x["n"]) for x in big], [x["rel_resid"] for x in big]) if len(big) > 3 else None
        max_abs = max(abs(x["rel_resid"]) for x in big)
        name = key.split("|")[1] + " " + key.split("|")[0]
        report["configs"][name] = {
            "c0_asym_ms": round(c0_asym, 5),
            "c0_spread_tail_pct": round(100 * (max(c0s) - min(c0s)) / abs(c0_asym), 1) if c0_asym else None,
            "BW_local_min_med_max_Gbps": [round(min(bw_loc), 1), round(statistics.median(bw_loc), 1), round(max(bw_loc), 1)] if bw_loc else None,
            "resid_rms_large_n_v3": round(rms_big, 4),
            "resid_max_abs_large_n": round(max_abs, 4),
            "resid_spearman_vs_log2n": round(rho, 3) if rho is not None else None,
            "n_points": len(rows)}
        verdicts.append(rms_big)

    json.dump(report, open(os.path.join(HERE, "results", "e1_analysis_v3.json"), "w"), indent=1)
    print("c0 launch anchor: %.4f ms" % c0_launch)
    for k, v in report["configs"].items():
        print("%-46s c0_asym=%7.4f BW_loc=%s rms=%.3f max|e|=%.3f rho=%s" %
              (k[:46], v["c0_asym_ms"], v["BW_local_min_med_max_Gbps"],
               v["resid_rms_large_n_v3"], v["resid_max_abs_large_n"], v["resid_spearman_vs_log2n"]))
    worst = max(verdicts)
    print("\nVERDICT: worst tail rms = %.3f -> %s" %
          (worst, "intercept closed (<=3c)" if worst <= 0.03 else
           ("borderline 3-5c, watch" if worst <= 0.05 else "REOPEN intercept question (>5c)")))

if __name__ == "__main__":
    main()
