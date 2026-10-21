"""E2 v2 verdict: noise-aware comparison of best hand variant vs torch, per point.
Class: beat (hand faster beyond combined-IQR noise and >1%), tie, lose.
Also: where does the best hand variant's derived-coords boundary sit; monotonicity of
m*(K) in programs-per-SM coords. P2 fields only; no claims beyond the data."""
import json, sys

P = "/Users/ckorhonen/rstation/kernelbench/lab/results/e2_splitk_v2.json"
d = json.load(open(P))
sm = d["meta"]["sm_count"]

def iqr(v): return v["iqr_ms"][1] - v["iqr_ms"][0]

n = {"beat": 0, "tie": 0, "lose": 0}
events = []
for r in d["rows"]:
    v = r["variants"]
    if "torch" not in v or "med_ms" not in v["torch"]:
        continue
    t = v["torch"]["med_ms"]
    hand = {k: x for k, x in v.items() if k != "torch" and "med_ms" in x}
    if not hand:
        continue
    bk = min(hand, key=lambda k: hand[k]["med_ms"])
    b = hand[bk]
    margin = (t - b["med_ms"]) / t                       # >0 hand wins
    noise = max((iqr(v["torch"]) + iqr(b)) / 2 / t, 0.01)
    if margin > noise:
        cls = "beat"
    elif margin < -noise:
        cls = "lose"
    else:
        cls = "tie"
    n[cls] += 1
    if cls == "beat":
        ctas = r["derived"]["ctas_tile"]
        events.append((r["K"], r["M"], bk, margin, noise, ctas, ctas / sm))

print("points classified: beat %d tie %d lose %d (of %d with torch)" %
      (n["beat"], n["tie"], n["lose"], n["beat"] + n["tie"] + n["lose"]))
print("\nhand-beats-torch points (K, M, variant, margin, noise, ctas, waves/SM):")
for e in events:
    print("  K=%8d M=%5d %-9s %+*.3f noise %.3f ctas=%4d waves=%.2f" % (e[0], e[1], e[2], 7, e[3], e[4], e[5], e[6]))

# best-hand family per M bucket: which family wins where (tile vs split vs neither)
print("\nbest-hand variant by (K,M): tile KT / split S distribution at each M")
from collections import Counter
byM = {}
for r in d["rows"]:
    v = r["variants"]
    hand = {k: x for k, x in v.items() if k != "torch" and "med_ms" in x}
    if not hand or "torch" not in v:
        continue
    t = v["torch"]["med_ms"]
    bk = min(hand, key=lambda k: hand[k]["med_ms"])
    fam = "tile" if bk.startswith("tile") else "split"
    byM.setdefault(r["M"], []).append((r["K"], bk, fam, (t - hand[bk]["med_ms"]) / t))
for M in sorted(byM):
    c = Counter(f for _, _, f, _ in byM[M])
    mg = sum(m for _, _, _, m in byM[M]) / len(byM[M])
    print("  M=%5d tile %2d split %2d  mean margin vs torch %+.1f%%  best:%s" %
          (M, c["tile"], c["split"], 100 * mg,
           Counter(b for _, b, _, _ in byM[M]).most_common(1)[0][0]))
