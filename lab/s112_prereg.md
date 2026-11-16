# s112_prereg.md — conv-only profile on the three mixed OTHER-labelled rows (frozen before any row)

Question s110 left as a hypothesis: on L2/48, L2/93, L2/100 the whole-model profiler top-1 is a layout or
elementwise kernel, and their exceed rate is unstable across draws (0.6875 / 0.4375 / 0.1875). Hypothesis:
the conv engine is invisible in the whole-model top slot and may be selected per draw. Test: profile the same
16 draws but restrict the event list to conv-engine kernels and read the engine there.

Population: L2/48 (conv), L2/93 (conv_transpose), L2/100 (conv_transpose) — the three mixed rows of s110.
16 draws each (seed d, manual_seed(d), get_init_inputs/get_inputs/Model), procedure B (cudnn tf32 on,
matmul tf32 on), fp32, one forward per draw, torch.profiler CUDA activity.

CONV_ENGINE set (regex): winograd | implicit_conv | implicit_gemm | dgrad | fprop | wgrad | cudnn | cutlass |
conv2d | conv_transpose | _conv_. Layout and pointwise kernels (nhwcToNchw, elementwise_kernel, copy, etc.)
are excluded by construction; they are recorded separately as whole-model top-1.

Recorded per cell: whole-model top-1 kernel and label (s110 classifier), conv-engine presence, the top
conv-engine kernel by self device time, its device time and its share of total device time, number of distinct
conv-engine names in the cell, and rho_ew (evaluator-exact vs chunked fp64 anchor, same statistic as s110) so
the deviation is joined to the engine label in one row.

Instrument checks: K1 repeat bit-identical on draw 0 of each row (2 forwards), K2 anchor self-check <=1e-9.

Frozen coded rule (no post-hoc repair):
- BRANCH_ENGINE_PER_DRAW if a conv engine is present on all 48 cells AND at least one row shows >=2 distinct
  conv-engine names across its 16 draws.
- BRANCH_NO_CONV_ENGINE if the conv engine is absent on every draw of at least one row.
- BRANCH_ENGINE_STABLE otherwise (one conv engine per row across all draws).

Consequence declared in advance: BRANCH_ENGINE_PER_DRAW => the map's label rule becomes "conv-only profile
required" and the s110 OTHER rows are re-labelled from that profile. BRANCH_NO_CONV_ENGINE => the conv is not
dispatched through a named engine kernel on those rows and the map carries an elementwise-deviance row.
BRANCH_ENGINE_STABLE => engine choice does not explain the 4.4x draw swing on L2/48; the swing is an
arithmetic-draw property and the label failure is a dominance artifact only.

Scope: three rows, one arch (GB10 sm_120), rev 423217d, procedure B, no timing claim.
