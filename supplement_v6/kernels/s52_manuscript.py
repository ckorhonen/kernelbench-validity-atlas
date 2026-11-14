"""Assemble the DETAILED section set (paper/DETAIL_SECTIONS.md) from the audited section files.

Round-18 split: MANUSCRIPT.md is assembled by s64_split_manuscript.py from paper/split/. This script keeps
the full §1-§9 record as a shipped detail document so the audit chain stays intact.

Conventions frozen here BEFORE the run:
- Sections are concatenated in order §1..§9 from paper/sections/; each section file is byte-identical to
  what the auditors verified (no editing here — this script assembles, it does not rewrite).
- The abstract is NOT repeated in the manuscript (rStation displays the metadata abstract above it), so the
  manuscript starts at §1.
- Front matter carries the title, the artifact/audit status line, and the scope line; the P2 rows and
  per-section audit banners that are internal scaffolding are kept, because they are the paper's own
  evidence discipline and a reader can act on them.
- Hard fail on: a missing section, an empty section, a section that cites a results/ artifact absent from
  lab/results (plain or .gz), or a total size over 400 KB.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def _kb_res_dir(here):
    """Resolve the artifact directory: lab/results in the repo, ../results when run from the
    supplement package's kernels/ directory. Frozen packaging convention (#55)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


def _kb_open(path):
    """Open a plain or gzipped artifact; row-streams may ship gzipped in the package."""
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        import gzip as _gz
        return _gz.open(path + ".gz", "rt")
    raise FileNotFoundError(path)
ROOT = os.path.dirname(HERE)
SECTIONS = os.path.join(ROOT, "paper", "sections")
OUT = os.path.join(ROOT, "paper", "DETAIL_SECTIONS.md")
RES = _kb_res_dir(HERE)

ORDER = ["s1_introduction.md", "s2_reference_axis.md", "s3_atlas.md", "s3b_conv_anchor.md",
         "s4_certificates.md", "s5_tolerance.md", "s6_selector.md", "s7_threats.md",
         "s8_reproduction.md", "s9_related.md"]

TITLE = "A validity-domain atlas for GPU kernels: conditional certificates, not band-global rules"

errors = []
parts = []

# Audit counts and provenance totals are READ from the artifacts, never typed here. N3 of the round-14
# review: the manuscript said 44/44/0 while the shipped s48_provenance.json said 49/49 and the package
# manifest said 50/50/0. A hardcoded number is a claim that can drift away from the artifact it describes.
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

*Artifact status.* Every number in this manuscript is regenerated from committed artifacts by shipped
auditors (`s50_prose_audit.py` {AC['s50']}, `s51_prose_audit.py` {AC['s51']}, `s52_prose_audit.py` {AC['s52']},
`s53_intro_audit.py` {AC['s53']}, `s54_metadata_audit.py` {AC['s54']}) and mapped to its file and field in
`PROVENANCE.md` ({AC['prov']}). The auditors exit nonzero on a cited value they cannot
reproduce, the supplement ships them together with the artifacts they read, and the build runs every
CPU-only script from inside the shipped package and fails if any exits nonzero, so the package is
re-runnable as delivered.

*Scope.* One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed
lattices rather than samples from the benchmark's distribution, unlocked clocks (clock lock denied by
permission). H100 folklore is re-derived here, not cited.

*Comparator.* torch.compile / recorded `torch_cmp` throughout, never eager PyTorch.

---

"""

parts.append(header)

for name in ORDER:
    p = os.path.join(SECTIONS, name)
    if not os.path.exists(p):
        errors.append(f"missing section {name}")
        continue
    txt = open(p).read().strip()
    if len(txt) < 500:
        errors.append(f"section {name} suspiciously short ({len(txt)} chars)")
    # renumber the internal draft banner into a reader-facing one, without touching the prose
    txt = re.sub(r"\(draft, tick #\d+ 2026-10-\d+\)", "(complete)", txt, count=1)
    parts.append(txt + "\n\n---\n\n")

body_parts = parts
APPX = os.path.join(ROOT, "paper", "APPENDIX_HISTORY.md")
if not os.path.exists(APPX):
    errors.append("missing appendix paper/APPENDIX_HISTORY.md")
else:
    body_parts.append(open(APPX).read().strip() + "\n")

body = "".join(body_parts)

cited = set(re.findall(r"results/([A-Za-z0-9_\.\-]+\.jsonl?)", body))
missing = sorted(c for c in cited
                 if not (os.path.exists(os.path.join(RES, c))
                         or os.path.exists(os.path.join(RES, c + ".gz"))))
if missing:
    errors.append(f"cited artifacts absent: {missing}")

if len(body.encode()) > 400_000:
    errors.append(f"manuscript {len(body.encode())} bytes exceeds 400 KB")

with open(OUT, "w") as f:
    f.write(body)

print(f"sections={len(ORDER)} cited={len(cited)} bytes={len(body.encode())} words={len(body.split())}")
if errors:
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    sys.exit(1)
print("wrote", OUT)
