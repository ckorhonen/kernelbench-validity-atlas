# Section 3 prose — Atlas entries (draft, tick #46 2026-10-23)

Sections 3.1 and 3.2 in full prose. Every number carries its artifact path (P1); the P2 schema
row closes each subsection. assembled into the round-4 paper from REFRAME_SKELETON.md.

---

## 3.1 Elementwise triad: the latency→bandwidth boundary collapses, and the copy ladder is not a reference

**Entry.** op_class `elementwise_triad` (a*x+b*y out-of-place, fp32, contiguous); arms: torch `add`,
Triton flat (BLOCK 256–16384, warps 1–8), Triton persistent (BLOCK 1024/4096, cap 48/192 CTAs).
Instrument: interleaved order-shuffled reps with L2 flush; per-config asymptotic intercept
`c0_asym` (median of t − B/BW_copy over the tail) and pairwise **local** slopes — the global
single-slope fit was rejected in analysis v2 after the two-slope copy ladder (259→220 GB/s) was
shown to pollute it (lab/e1_analyze_v3.py; decision recorded DIARY session 2).

**Result 1 — the boundary collapses where grid geometry is held.** On the persistent arms and the
flat BLOCK=4096 arms, latency→bandwidth behavior collapses to `t(n) = c0_asym + n·bytes/BW_eff`
with residual RMS 0.59–2.95% (persistent: 0.59/1.06/1.12/2.95% at cap192/48, BLOCK 4096/1024;
flat-4096: 2.46–5.33% across warps) and saturated BW_eff 216.7–225.9 GB/s — copy parity on this
device (results/e1_analysis_v3.json, 37 n-points per arm). The collapse is day-replicable: at
saturated n≥2^27, day1→day2 BW drift is −0.09% (torch 229.8→229.6), −0.13% (flat), +0.02%
(persistent) over a 9.6 h interval (results/e1_triad_day2.json, 777 rows). Small-n is another
regime: relative drift p90 +17–19% confined to n=2^20 where t sits at the floor — independent
support for carrying an explicit floor term rather than a pure-slope law.

**Result 2 — the copy ladder is NOT a universal reference (falsifier resolved, mechanism found).**
Flat small-BLOCK arms keep rising where the ladder saturates: flat BLOCK=256 reaches 236–265 GB/s across warps
(RMS 20% against the ladder), flat 1024 w8 touches 261.5; torch itself sits at 226–241 with 8.8%
RMS. The pre-registered falsifier "persistent vs flat, same BLOCK, differ >5% saturated" fired,
and e3 decomposed it by intervention (results/e3_gridgeom.json, pre-registered in module
docstring, 4 passes): (R1) permuting the CTA→data mapping at FIXED geometry costs 4.1–4.4% BW
(226.2→216.1 at B256; 220.6→211.6 at B1024), replicated, ≫ run noise (<0.3%) — an address-phase
effect, always present; (R2) wave starvation is real but BLOCK-conditional: at B256, 1-wave
persistent loses 12.3% and ≥2 waves recover only 67% of the gap; at B1024 persistent≈flat (0.6%);
(R3) clocks are ruled out — 2216 MHz at every pre/post sample, run-to-run <0.6 GB/s. The
boundary therefore needs a grid-geometry coordinate (address phase; wave count at small
bytes/CTA), not a device-level BW constant.

**Scope and residues (P5).** The flat-256 excess (246.5 GB/s in the v3 sweep) did not reproduce at
num_warps=4 in e3 — warp-count sensitivity untested, carried as a residue. Clock lock is
permission-denied on this host and power.draw is absent from GB10 telemetry, so DVFS is bounded
only by sampled stability; order-protocol: ascending runs carry a cold-clock head bias
(208 MHz/5.9 W, first ~10 units; anchor spread 1.08% asc vs 0.01% shuffled, results/e1b_order_v2.json)
— shuffled order is the default henceforth.

**P2 row.** `{op_class: elementwise_triad, rule: torch|flat|persistent, domain_predicate:
saturated n≥2^27 ∧ grid-geometry fixed ∧ (address-phase, waves) matched; collapse RMS ≤3%,
day-drift ≤0.13%, derived_coords: [bytes/thread, waves resident, c0_asym], boundary_ref:
results/e1_analysis_v3.json + e3_gridgeom.json + e1_triad_day2.json, variance: per-arm RMS
0.59–20%, n_reps: 25–37, date: 2026-10-16/17}`

---

## 3.2 Split-K, small output: a two-condition boundary — wave band AND launch floor

**Entry.** op_class `reduction` (A[K,M] → b[M], small M); comparator torch reduction and
torch.compile; hand arms: 2D-tiled split-K `tsplit_S{4,16,64}_KT{16,64}` (grid (⌈M/B⌉,S), KT×B
tile loads, two-phase) on GB10.

**Instrument lineage (negative entry kept per P5).** The v1 "flat" arm — 1D column loop, 512
B/CTA/iter — loses 30× at K=2^20, M=32 (34.0 vs 0.64 ms) purely from load shape; v2's fair 2D tile
ties torch for M≥2048 at 0.33–0.67 waves. The v1 gap was an instrument artifact, not physics
(results/e2_splitk_v2.json); any "flat vs torch" claim from a 1D loop is void on this device.

**Result 1 — the hand win exists and is band-bounded (canonical count).** Under the rule frozen in
lab/s46_e2_recount.py (beat ⟺ min hand-IQR-upper < torch-IQR-lower, 12 reps/arm), the tiled split
beats torch at **30/60** (K,M) cells, ties 2, loses 28; margins +0.93 to +70.5% (max at K=4096,
M=16, S16/KT64). Winning cells span 16–128 total CTAs — but losing cells span 16–64, so the CTAs
band is **necessary, not sufficient**; the earlier "wins live in 32–128" reading was necessary-
condition-only and is corrected here. At M≤8: 2 beats, 22 losses — the small-M stratum is mostly
launch-bound, with the exceptions below.

**Result 2 — the boundary is not a shape surface; it is advantage vs a launch constant.** Direct
two-launch floor: 7.90 µs vs 5.86 µs single (Δ 2.04 µs; results/e2b_floor.json). The crossover M*
is non-monotone in K (Spearman −0.643; the pre-registered "MONOTONE iff ρ>0" rule prints
NON-MONOTONE — the direction was never registered; recorded as a pre-registration slip, one-sided
p=0.049 by the corrected tail, n=7, K=2^12 censored with no sustained win). Sustained-win M* vs
first-win M* diverge exactly where the advantage curve crosses the launch constant
(M*_sust/M*_first: 128/64, 64/64, —/32, 128/16, 128/8, 48/4, 10/10, 12/12 across K=2^8…2^18).
Domain predicate gains a time term: **win requires advantage(M;K) > t_launch,Δ ≈ 2 µs**, unreachable
at launch-bound K regardless of waves. The measurable object is the advantage curve, not a
boundary in shape space.

**Result 3 — the induced selector is probe-free and transfers.** The W4 ablation ladder on this
family selects rule E — hand iff **M≥16 ∧ bytes/thread ≥1024** (bt = 2·bytes/(CTAs·128),
shape-derived; results/w4_v1.json): LOO regret 0.0327 vs always-torch 0.0450, median 0,
77% oracle-equal, 4/60 in-sample worse-than-1% (stated); the launch-bound K=2^12 fold flips to
gate-win (0.100 vs 0.112). On the e2b M* grid with a never-trained hand arm (tsplit_pinned):
0/144 cells worse-than-1% — the "never worse" adoption bar passes out-of-sample; in-sample 4/60
and bt-threshold sensitivity remain parked (CTA-edge sharpness bisection likewise parked).

**Numerics.** Regenerated on the fixed harness (results/e2_v3_err.json, 1260 measurements, 3
seeds): median abs err 2–5e-4, max 4.2e-2; S64/KT64 tightest (max 4.5e-3). The original per-variant
error table is void — v3's `o2=empty_like(o)` captured uninitialized memory (lab/e2_debug2.py);
timing unaffected (same kernels, median of 12). Tolerance latitude is real and uneven across
(S,KT) — carried to §5, not averaged away.

**P2 row.** `{op_class: reduction_splitk_smallout, rule: tsplit 2D-tile + gate E(M≥16 ∧ bt≥1024),
domain_predicate: advantage(M;K) > 2.04 µs ∧ CTAs∈[16,128] necessary-not-sufficient ∧ M≥16 for
the shipped gate, derived_coords: [total CTAs, bytes/thread, r=t_kernel/t_launch], boundary_ref:
results/e2b_floor.json + s46_e2_recount.json + w4_v1.json, variance: IQR of 12 reps, n_reps: 12
(×3 seeds correctness), date: 2026-10-17}`
