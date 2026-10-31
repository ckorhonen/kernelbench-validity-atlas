# 6 — Reproduction

The submission ships the artifact set, the producers, the preregistrations and the auditors. Every number in
this manuscript is regenerated from a committed artifact by a shipped auditor, and the mapping from claim to
file and field is in `PROVENANCE.md`. The auditors exit nonzero on a cited value they cannot reproduce, and
the package build runs every CPU-only script from inside the unpacked package and fails if any exits
nonzero, so the package is re-runnable as delivered rather than merely delivered.

Two honest caveats about that machinery. The auditors are consistency checks, not independent
re-measurements (§4): they prove that a number matches the artifact it came from, not that the measurement
was right. And the GPU legs reproduce only on a GB10 with this stack; they ship with the artifact each
produced, so a reader without the hardware can audit the result without re-running the measurement.

The public repository is pinned at a commit that is an ancestor of its main branch, and every repo-derived
file in the package is a byte-copy of its counterpart at that commit. Repair history, the retraction ledger,
superseded interpretations and the secondary experiments that this paper does not claim are in Appendix A and
in the shipped artifact set, each stated once with its artifact path.
