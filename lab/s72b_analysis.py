"""s72b -- post-hoc descriptive analysis of results/s72_fresh_process.jsonl.

Labeled POST-HOC and descriptive: no threshold is changed, no preregistered read is refit.
s72's registered reads (R1 process identity, R2 budget invariance) are decided by the
producer's summary; this script adds the sharper measurement the summary does not carry --
whether the engine IDENTITY (kernel-name set) is identical across fresh processes, not only
the class label, and whether the reference output is bit-identical across processes.

Writes results/s72b_analysis.json.
"""
import collections, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results", "s72_fresh_process.jsonl")
OUT = os.path.join(HERE, "results", "s72b_analysis.json")
DATE = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

cells = [json.loads(l) for l in open(IN)]
by = collections.defaultdict(list)
for c in cells:
    by[(c["H"], c["N"], c["C"])].append(c)

rows = []
name_ident = 0
absmax_ident = 0
class_ident = 0
for k, v in sorted(by.items()):
    names = [tuple(sorted(n for n, _ in c["top_kernels"])) for c in v]
    am = [c["ref_absmax"] for c in v]
    cls = [c["engine_class"] for c in v]
    ni = len(set(names)) == 1
    ai = len(set(am)) == 1
    ci = len(set(cls)) == 1
    name_ident += ni
    absmax_ident += ai
    class_ident += ci
    rows.append({"H": k[0], "N": k[1], "C": k[2], "n_cells": len(v),
                 "engine_class": cls[0], "class_identical": ci,
                 "kernel_names_identical": ni, "ref_absmax_identical": ai,
                 "ref_absmax": am[0], "top_kernel": v[0]["top_kernels"][0][0][:120],
                 "budgets": [c["budget"] for c in v], "pids": [c["pid"] for c in v],
                 "nonces": [c["nonce"] for c in v]})

# device-time dispersion of the top kernel across cells: descriptive only, no timing claim
disp = []
for k, v in sorted(by.items()):
    t = [c["top_kernels"][0][1] for c in v]
    disp.append({"H": k[0], "N": k[1], "C": k[2], "top_us": t,
                 "spread": round(max(t) - min(t), 2)})

summary = {"date": DATE, "source": "results/s72_fresh_process.jsonl", "n_cells": len(cells),
           "n_rows": len(by),
           "class_distribution": dict(collections.Counter(c["engine_class"] for c in cells)),
           "class_identical_rows": class_ident, "kernel_names_identical_rows": name_ident,
           "ref_absmax_identical_rows": absmax_ident,
           "distinct_pids": len(set(c["pid"] for c in cells)),
           "distinct_nonces": len(set(c["nonce"] for c in cells)),
           "rows": rows, "top_kernel_us_dispersion": disp,
           "note": "descriptive; the registered reads are s72's R1/R2, decided by the producer summary"}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("n_cells", "n_rows", "class_distribution",
                                          "class_identical_rows", "kernel_names_identical_rows",
                                          "ref_absmax_identical_rows", "distinct_pids")}))
