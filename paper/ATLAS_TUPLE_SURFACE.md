# ATLAS note — tuple surface (round-11 measurement, 2026-10-28)

Producer `lab/tuple_surface.py`, prereg frozen pre-run (commit c16fb5e), record `lab/results/tuple_surface.json`.
Spark GB10, sm_120, torch 2.14.1+cu130, seed 4321, 17 cells × 2 repeats, one coordinate varied per cell at
base C256/k3/H=W=7/batch2.

## Measured

| read | result |
|---|---|
| class determinism across repeats | 34/34 runs agree, 17/17 cells — class is a function of the declared tuple |
| simple rule "W iff s=d=1" | holds for C∈{64,256} at H=7; **fails** at C=16 (T, rho 60.2) and H=W=14 (T, rho 371.2) |
| real channels_last (A07) | class T, rho_max 429.4, frac_over 0.784, input strides [12544,1,1792,256]; flag-off → fp32 SIMT rho 0.239, bit-inequal |
| pad-2 at s=1,d=1 (A06) | class W, rho_max 12.8024 — reproduces the round-9 reviewer's stored value |
| W span across measured cells | rho_max 1.4–12.8, frac_over 0.0005–0.0077, bit-identical under every precision flag (7/7) |
| T span across measured cells | rho_max 60.2–429.4, frac_over 0.760–0.786, flag-off removes 96.4–99.9% (ratios 5.6e-4–3.6e-2) |
| fp64 anchor class | O (`implicit_convolve_dgemm`) in every cell; no tensorop path at fp64 |

## What it settles

- **Positive result:** admission is dispatchable from declared coordinates — a deterministic categorical
  partition of the tuple, step-shaped per axis. Enumerated-tuple lookup, not a continuous model.
- **Scope:** the step positions depend on depth and extent, so a rule over (stride, dilation) alone is partial.
  This replaces both the "no simple surface" overclaim and the reviewer's proposed general threshold rule.
- **Layout:** a genuine dispatch coordinate; the round-8 "NCHW or NHWC" invariance claim is withdrawn.

## Validity domain

Single device (GB10), single dtype path (fp32 in, fp32 out, k=3, C_out=C), one seed, one shape family
(7–14 spatial, batch ≤ 8). No claim about H100 or about k≠3. Determinism is measured only across two repeats
in one process — cross-session stability of class membership is not established here.
