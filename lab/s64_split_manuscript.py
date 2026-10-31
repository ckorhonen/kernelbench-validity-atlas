"""Assemble the round-18 SPLIT manuscript: one contribution, ~40% of the round-17 length.

Conventions frozen HERE, before the run (round-18 repair, one session, hard cap):
1. The split document is assembled from paper/split/p1..p6 only. Each part is byte-identical to what the
   shipped auditors were run against; this script assembles, it does not rewrite prose.
2. Repair history, the retraction ledger, superseded interpretations and the secondary experiments are NOT
   in the body. They ship as Appendix A (paper/APPENDIX_HISTORY.md) and in the artifact set, each stated once
   with its artifact path.
3. Hard fail on: a missing part, a part shorter than 500 characters, a cited results/ artifact absent from
   the artifact tree (plain or .gz), a body over 120 KB, and a body above 55% of the round-17 length
   (155,902 bytes) — the split claim is a size claim and it is checked rather than asserted.
4. Audit counts and provenance totals are READ from artifacts, never typed here.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _kb_res_dir(here):
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


def _kb_open(path):
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        import gzip as _gz
        return _gz.open(path + ".gz", "rt")
    raise FileNotFoundError(path)


ROOT = os.path.dirname(HERE)
SPLIT = os.path.join(ROOT, "paper", "split")
OUT = os.path.join(ROOT, "paper", "MANUSCRIPT.md")
RES = _kb_res_dir(HERE)

ORDER = ["p1_question.md", "p2_protocol.md", "p3_results.md", "p4_limits.md", "p5_related.md",
         "p6_reproduction.md"]

TITLE = "The Reference Is an Independent Variable: A Benchmark Verdict Is Reference-Conditional"

R17_BYTES = 155902

errors = []
parts = []


def _audit_counts():
    def load(n):
        with _kb_open(os.path.join(RES, n)) as f:
            return json.load(f)
    out = {}
    a = load("s50_prose_audit.json")["summary"]
    out["s50"] = f'{a["n_checks"] - a["n_failed"]}/{a["n_checks"]}'
    a = load("s51_prose_audit.json")
    out["s51"] = f'{a["n_pass"]}/{a["n_checks"]}'
    a = load("s52_prose_audit.json")
    out["s52"] = f'{a["n_pass"]}/{a["n_checks"]}'
    a = load("s53_intro_audit.json")
    out["s53"] = f'{a["n_pass"]}/{a["n_checks"]}'
    a = load("s54_metadata_audit.json")
    out["s54"] = f'{a["pass"]}/{a["total"]}'
    p = load("s48_provenance.json")
    out["prov"] = f'{p["n_cited"]} cited / {p["n_found"]} found / {len(p["missing"])} missing'
    return out


AC = _audit_counts()

header = f"""# {TITLE}

*What this paper is.* One contribution, stated in §1.2 and measured in §2-§3: a benchmark row's verdict is a
function of the reference realization as well as of the candidate. The paper was split from a larger atlas
after a review round found that its claim set had grown past what one reviewer could check; the secondary
experiments, the retraction ledger and the repair history now ship as Appendix A and as artifacts rather
than as body prose.

*Artifact status.* Every number in this manuscript is regenerated from a committed artifact by shipped
auditors (`s50_prose_audit.py` {AC['s50']}, `s51_prose_audit.py` {AC['s51']}, `s52_prose_audit.py` {AC['s52']},
`s53_intro_audit.py` {AC['s53']}, `s54_metadata_audit.py` {AC['s54']}) and mapped to its file and field in
`PROVENANCE.md` ({AC['prov']}). The auditors exit nonzero on a cited value they cannot reproduce. They are
consistency checks, not independent re-measurements (§4).

*Scope.* One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed
lattices rather than samples from the benchmark's distribution, unlocked clocks (clock lock denied by
permission). H100 folklore is re-derived here, not cited.

*Comparator.* torch.compile / recorded `torch_cmp` throughout, never eager PyTorch.

---

"""

parts.append(header)

for name in ORDER:
    p = os.path.join(SPLIT, name)
    if not os.path.exists(p):
        errors.append(f"missing part {name}")
        continue
    txt = open(p).read().strip()
    if len(txt) < 500:
        errors.append(f"part {name} suspiciously short ({len(txt)} chars)")
    parts.append(txt + "\n\n---\n\n")

APPX = os.path.join(ROOT, "paper", "APPENDIX_HISTORY.md")
if not os.path.exists(APPX):
    errors.append("missing appendix paper/APPENDIX_HISTORY.md")
else:
    parts.append(open(APPX).read().strip() + "\n")

body = "".join(parts)

cited = set(re.findall(r"results/([A-Za-z0-9_\.\-]+\.jsonl?)", body))
missing = sorted(c for c in cited
                 if not (os.path.exists(os.path.join(RES, c))
                         or os.path.exists(os.path.join(RES, c + ".gz"))
                         or os.path.exists(os.path.join(ROOT, "results", c))))
if missing:
    errors.append(f"cited artifacts absent: {missing}")

n_bytes = len(body.encode())
if n_bytes > 120_000:
    errors.append(f"split body {n_bytes} bytes exceeds 120 KB")
if n_bytes > 0.55 * R17_BYTES:
    errors.append(f"split body {n_bytes} bytes exceeds 55% of round-17 length {R17_BYTES}")

with open(OUT, "w") as f:
    f.write(body)

print(f"parts={len(ORDER)} cited={len(cited)} bytes={n_bytes} words={len(body.split())} "
      f"frac_of_r17={n_bytes / R17_BYTES:.3f}")
if errors:
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    sys.exit(1)
print("wrote", OUT)
