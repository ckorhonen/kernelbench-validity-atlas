# Deep Reflection #2 (revival design) — conclusions, 2026-10-18 04:08Z
5/5 rounds on-platform. Candidates fully developed in-session:
- **A. Family-local analytical dispatch** — old school: model-based selection killed by autotuning consensus; positives: production heuristic tables + my W4 v1 gate (0.0327 vs 0.0450, 0/144 transfer-worse); scoped negative: universal-basis W3 failure; mechanism: regime-piecewise low-dim runtime + domain-agnostic/config-local split; new regime: zero-probe dispatch vs budgeted autotune at visible cold-start. **PRIMARY.**
- **B. Persistent-kernel wave-pinning** — persistence buys nothing DRAM-bound (220.4 vs 227/229.6 GB/s) but fixes ctas a priori; natural instrument+dispatch target in the floor regime (2.0µs two-launch delta). Contained by A as its floor-regime section; pivot target if A's premise dies.
- **C. Counter-preserving (bit-identical) kernels** (ruachtov CUPTI, sm_61) — verification axis as third atlas axis; infra-heavy, arch-rederivation required; parked as feasibility note only.
Decision rule recorded: M0 residual test decides A vs B-pivot; kill condition written (see plan).
