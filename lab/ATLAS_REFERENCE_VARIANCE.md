# ATLAS note — the reference margin's variance is shape/dispatch, not input (s80/s80b/s80c)

Status: measured on the 54-row exceed population and a 14-tuple dgrad family, GB10 sm_120, torch 2.14.1+cu130,
rev 423217d. Producers `lab/s80_instance_rule.py`, `lab/s80b_dgrad_family_draw.py`,
`lab/s80c_label_reproducibility.py`; preregs frozen pre-run at commits 0d71886, then s80b/s80c separately.

## 1. The instance label was produced by a rule the producer did not declare

s80 re-profiled all 54 exceed rows under two declared rules in one pass: R0 ranks every profiler entry by self
device time (the rule `profile_call` actually implements), R1 ranks only compute kernels (Memcpy/Memset excluded).

- Recorded `kernel_top1` equals the R0 top-1 on **51/54 rows (94.4%)** — the producer's rule is identified, and
  it is not the "CUDA kernels by self device time" its docstring declares.
- **11/54 recorded top-1 labels are Memcpy or Memset events**; 13/54 differ from the R1 top-1 (24.1%).
- Consequence: the instance table published as rOZ7mxOvxdfVUWIf and corrected as gaJoW3UP01V_ISRO was labelled
  by data movement on a fifth of its rows. That table is retracted to "label-dependent grouping".

## 2. Re-emitted under the declared rule: compression is weak, and the largest span is a layout kernel

Under R1 the 54 rows resolve to 21 instances, 12 of them multi-row. Frozen verdict (P3): **COMPRESSION_WEAK** —
8/12 multi-row instances reproduce within 5x (66.7%, below the 70% clause) and the largest multi-row span is
**19,439.9x**, exceeding the 2,608x decade bar.

Texture, declared post-hoc: the 19,440x span belongs to `nhwcToNchwKernel` (n=11), a layout conversion, and the
next two exceedances are elementwise_add (10.0x) and xmma fprop-nchw (15.2x). Every dgrad compute instance is
<=3.77x (indexed xmma 3.66x n=5, cutlass 256x64 3.77x n=3, cutlass 128x64 1.84x n=3, xmma non-indexed 1.34x
n=3). Reading: "engine instance" is not one coordinate — compute engines, layout conversions and elementwise
kernels must be split before an instance span is cited as a magnitude coordinate.

## 3. The label is a one-draw observation

s80c, fixed benchmark flags, two consecutive profiler passes on the same tensors per tuple (14 dgrad-family
tuples): pass-to-pass top-1 identical **14/14**; agreement with s80's label across a changed input draw
**12/14**. The two flips are rows where `nhwcToNchwKernel` (20.2 / 20.4 ms self) displaces a cutlass dgrad
kernel. Frozen verdict at the predeclared 0.90 bar: **LABEL_UNREPRODUCIBLE**, so instance grouping is demoted to
descriptive; the per-shape budget column remains the citable object.

Harness defect declared: s80b's profiling pass inherited tf32-off from its own draw loop, so its S2 agreement
(1/14 with s80) is a flag-state confound, not a reproducibility result. The draw numbers are unaffected because
each draw set the flags explicitly before arm A.

## 4. Draw invariance holds on the corrected population, and shape dominates by ~8x

s80b, 14 dgrad-family tuples selected by the R1 rule, 6 input draws each (seeds 300-305), default reference vs
fp64 anchor, element-chunked predicates at 1e-4 and 1e-2:

- rho_max draw spread max/min **1.004-1.140x** on every tuple (max 1.140x).
- frac_over at 1e-4 stable to <1% relative on every tuple; **0.0 at the shipped 1e-2 gate on every tuple and
  every draw**.
- Across-shape bias span for the family: **8.963x**; shape-to-draw ratio **7.86** => SHAPE_DOMINATES_AT_
  FAMILY_SCOPE.

Reading: at fixed tuple the reference margin is input-draw invariant to ~14%, so the 8.96x family span is
shape/dispatch-attributable. This is the variance decomposition the 02:24Z reflection named, measured on the
corrected population rather than the contaminated containment group.

## 5. What can be cited, and what cannot

Citable: the enumerated per-shape reference budget column; draw invariance at fixed tuple (~14% bound); the
finding that shape-attributable variation exceeds draw-attributable by ~7.9x at family scope; the R0/R1
distinction as a documented producer defect.

Not citable: "one engine instance spans four decades" (that decade lived in a layout kernel); any instance-level
magnitude law; the instance table as a dispatch constant.

Limits: 54 exceed rows only (the 1e-4-exceeding population), one seed per task for weights, six draws per tuple,
single GB10 stack, profiler-instrumented runs only.
