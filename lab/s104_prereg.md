# s104 prereg — E3 merge-tree sweep: is the order term a depth property or a tree property?

Frozen 2026-11-14 ~22:1xZ, committed with the runner BEFORE any row of this run. No threshold below is
changed after data; post-hoc additions are labelled as such in the diary and the artifact.

## Why (debt being closed)

E3 has been deferred five ticks. Every order-term measurement I own (s101b, s102, s103, s103b) uses
contiguous channel-order chunks with a sequential merge, so tree *shape* is untested and two tuples of one
topology are not evidence about it. The 22:03Z reflection committed this tick to E3 as the discriminator for
the conjecture "the order term is a depth ratio" (c ~ sigma_perm/sigma_ref, topology entering only through
effective depth).

The prior measurements bound what is at stake: s101b's zero-padded fixed-class arms sit at
rho_vs_fp64/rho_A = 0.999 (13.9393 vs 13.9525 at g=1), i.e. the perturbation is the SAME distance from the
fp64 anchor as the reference and only 0.22-0.27% away from the reference itself. So the shared TF32
quantization error dominates both arms and the order term is a small reassociation residual on top of it.
Whether that residual is set by depth alone or by the tree's own shape is exactly what E3 measures.

## Design (frozen)

Tuple T1 (s91/s101/s101b/s103): conv 3x3 s1 p1, B=2, C_in=64 -> 64, 151x151, seed 1085,
w scaled by 1/sqrt(fan_in) then by g. Reference A = cudnn conv with allow_tf32 True (torch default),
X = fp64 anchor with allow_tf32 False. u = 2.2e-4 * |A|max / atol per gate, as in s91/s101b.

Scale grid g in {1.0, 4.0, 16.0, 64.0} => u_1e-4 10.6 / 42.5 / 170 / 680 and u_1e-2 0.106 / 0.425 /
1.70 / 6.80; all four points qualify (rho_A >= 0.05) at both gates on s101b's record.

Arms at MATCHED DEPTH (16 chunks, each zero-padded back to C_in=64 so the TF32 tensorop engine is held,
per s101b's class-preservation method):
- C_seq: contiguous channel-order membership, sequential ascending merge (s101b P16z replica).
- C_shuf: identical membership to C_seq, sequential merge in a declared fixed permutation
  (numpy RandomState(20261114).permutation(16), recorded in the artifact).
- C_bal: identical membership to C_seq, fan-in-2 balanced merge (pairwise combine at each of 4 levels,
  declared order recorded).
- I_seq: interleaved membership (chunk i takes channels i, i+16, i+32, i+48), sequential ascending merge.
Depth controls (not part of the topology contrast): P4z, P64z = contiguous sequential at 4 and 64 chunks,
to tie the depth axis to s101b (c 0.00233 / 0.00313 at g=1).

Measured per arm and gate: rho_vs_ref (candidate-scaled allclose ratio vs A), frac_vs_ref, rho_vs_fp64,
c = rho_vs_ref / rho_A_x, plus per-arm single-draw wall time (no timing claim).

## Controls (all must pass or the verdict is INSTRUMENT_FAILED)

- K1: A run twice, bit-identical.
- K2: C_seq at g=1 reproduces s101b's P16z rho_vs_ref_1e-4 = 0.0376 within 2%.
- K3: engine class of every chunked arm at g=1 and g=64 is TF32_TENSOROP (else CLASS_NOT_HELD).
- K4: C_seq invoked twice at g=1 is bit-identical (determinism of the declared tree).
- K5: fp64 anchor self-check: X computed twice, bit-identical.

## Frozen branch rule

Qualifying point = rho_A >= 0.05 at that gate. For each gate with >= 2 qualifying points, compute the
matched-g spread S_g = max_arm c / min_arm c over {C_seq, C_shuf, C_bal, I_seq}.
- TOPOLOGY_INVARIANT iff max_g S_g <= 1.20 at BOTH gates (>= 2 qualifying points each).
- TOPOLOGY_DEPENDENT iff max_g S_g > 1.20 at either gate.
- RANGE_INSUFFICIENT if a gate has < 2 qualifying points (then the other gate alone decides, declared).
- INSTRUMENT_FAILED if any control fails; no topology claim is quoted in that case.

## Predeclared interpretation (written before data)

TOPOLOGY_INVARIANT => the order term is a depth property; the column ships c(depth) and the declared-tree
axis collapses to chunk count. TOPOLOGY_DEPENDENT => depth is necessary but not sufficient; the column
carries (depth, tree) and the 22:03Z conjecture is falsified in its topology clause, with the arm-level
ordering reported as a measured surface rather than a law.

## Non-claims (written before data)

One tuple, one draw per g, one seed, contiguous-vs-interleaved membership only, fan-in-2 balanced only,
fp32 partial accumulation only, no timing claim, no split-K across the output/spatial axis (channel axis
only). Nothing here is a statement about non-conv families.
