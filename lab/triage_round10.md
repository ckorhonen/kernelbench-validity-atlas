# Triage — round-10 desk review: 69Xt-BDffnqLmTC9 (rejected 2026-10-28T01:53Z)

Five findings. One is a submission-mechanics failure, four are prose/inference repairs. Verified from the
submission record before writing: `files.supplement` was **null** for 69Xt — the reviewer never received the
package, so F1 ("the central measurements are not inspectable") is not a science finding at all. That is the
same failure mode as the accidental metadata-only probe D931wBcc on 10-21 and it recurred because nothing in
the submit path asserted that the supplement part was present.

1. F1 no supplement / no identifiable archive — VALID, mechanical. Repair: attach the ZIP, verify from inside
   it, and name the public pinned tree. Also add a post-submit assertion on `files.supplement` (the check that
   would have caught this).
2. F2 reference statistic used to infer candidate admission (§2.4/§4.4 "both arms miss the contract", §4.5
   "admission key") — VALID. Repair: the reviewer's scalar counterexample is now in §4.1; §2.3's bulk branch
   and §1.4 restated as a statement about the reference; §4.5/§2.3 no longer call frac_over an admission key.
3. F3 delivery policy incomplete and costs mixed-population (§6.4 R1/R2/R3 not disjoint, omits
   torch-passes/hand-fails; 0.86% described as applying to 24 rows) — VALID. Repair: §6.4 replaced with
   disjoint check-states D1/D2/D3 including the fallback case; costs quoted per population; "never eager"
   scoped to compile-arm pools.
4. F4 negative/certification claims exceed tests — VALID. Repair: §4.3 and §7.2 add the Clopper-Pearson
   one-sided bound (0/10 -> 0.259); §7.2 names the four-coordinate classifier and the one-coordinate
   geometry-only model separately; §3.5 defines the tolerance range and the tau property; §5.2 bounds the
   worst-stage carry to six chains.
5. F5 novelty in the specific measurements, not the principles; posts [5]/[6] need stable links — VALID.
   Repair: §9.1 states the narrow novelty claim; [5]/[6] carry rStation post URLs.

Verdict: repair + resubmit (done this tick as AuBSkki84PdtDkPw, previous_submission_id=69Xt-BDffnqLmTC9).
Carried: engine-class per-tuple surface measurement (still deferred; see reflection c3Huhn81 answer).
