"""R42 repair: grid-corrected rerun of the 4 affected r1 rows (rejection O4hnCx V1).

The #38 run garbled the two extra points per family: it ran (H,N,C) (3,2,5),(3,2,12) for w4
and (4,2,9),(4,2,20) for a2, where the frozen held-out grids (w4_heldout.py:76,
a2_day2.py:61, verified this tick) are (N,H,C) (3,5,32),(3,12,96) and (4,9,48),(4,20,128)
-- i.e. (H,N,C) (5,3,32),(12,3,96) and (9,4,48),(20,4,128). Wrong shapes AND wrong seeds.
This rerun executes ONLY those 4 rows, against the frozen (N,H,C), reusing r1_conv_gated
verbatim (same arms, checks, contract, seed formula 1234+H*100+N*10+C applied to the
correct (H,N,C) -- declared fresh-evidence stream, per the r1 header).

Output: results/r42_gridfix.jsonl (+ mechanical per-run summary). The CANONICAL corrected
summary/analysis (corrected oracle = min passing hand AND torch; gate rule stated;
paired d_i) is r42_analysis.py over the union {r1 rows} - {4 wrong rows} + {these 4}.

Run on spark: python3 r42_gridfix.py
"""
import os
import r1_conv_gated as r1

HERE = os.path.dirname(os.path.abspath(__file__))
r1.OUTL = os.path.join(HERE, "results", "r42_gridfix.jsonl")
r1.GRIDS = [
    ("w4", [(5, 3, 32), (12, 3, 96)]),   # (H, N, C); frozen (N,H,C) (3,5,32),(3,12,96)
    ("a2", [(9, 4, 48), (20, 4, 128)]),  # frozen (N,H,C) (4,9,48),(4,20,128)
]

if __name__ == "__main__":
    r1.main()
