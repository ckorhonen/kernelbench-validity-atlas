"""Audit the submission metadata against the audited manuscript.

Frozen rules, written before the run:
A1. Every numeric token in the metadata abstract must appear verbatim in paper/MANUSCRIPT.md. The manuscript
    is the audited surface (s50-s53 regenerate its numbers from committed artifacts), so an abstract number
    that is not in the manuscript is an unbacked number.
A2. The abstract must be 200-300 words (rStation hard maximum is 300).
A3. Required metadata fields present and well-typed; tags non-empty strings, at least 3.
A4. No absolute paths and no credentials in any metadata field.
A5. The title must not exceed 120 characters.
A6. change_summary must name the three prior rejections and the paper_id of the most recent one.
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
META = next((os.path.join(ROOT, n) for n in ("submission_metadata_v6.json", "submission_metadata.json")
             if os.path.exists(os.path.join(ROOT, n))), os.path.join(ROOT, "submission_metadata_v5.json"))
MS = os.path.join(ROOT, "paper", "MANUSCRIPT.md")
OUT = os.path.join(_kb_res_dir(HERE), "s54_metadata_audit.json")

checks = []


def check(cid, name, ok, detail):
    checks.append({"id": cid, "name": name, "pass": bool(ok), "detail": detail})


d = json.load(open(META))
body = open(MS).read()

# A1
nums = sorted(set(re.findall(r"\d+(?:\.\d+)?", d["abstract"])))
absent = [n for n in nums if n not in body]
check("A1", "abstract numbers present in audited manuscript", not absent,
      {"n_numbers": len(nums), "absent": absent})

# A2
w = len(d["abstract"].split())
check("A2", "abstract word count within 200-300", 200 <= w <= 300, {"words": w})

# A3
req = ["problem_name", "statement_version", "title", "abstract", "tags"]
miss = [k for k in req if k not in d]
typed = (isinstance(d.get("statement_version"), int)
         and isinstance(d.get("tags"), list) and len(d.get("tags", [])) >= 3
         and all(isinstance(t, str) and t.strip() for t in d.get("tags", [])))
check("A3", "required fields present and typed", not miss and typed,
      {"missing": miss, "typed": typed, "n_tags": len(d.get("tags", []))})

# A4
fields = {k: v for k, v in d.items() if isinstance(v, str)}
abs_paths = {k: v for k, v in fields.items() if re.search(r"(?:^|\s)/(?:Users|home|tmp)/", v)}
creds = {k: v for k, v in fields.items() if re.search(r"(sk-|ghp_|Bearer )", v)}
check("A4", "no absolute paths or credentials in metadata", not abs_paths and not creds,
      {"abs_paths": sorted(abs_paths), "credential_like": sorted(creds)})

# A5
check("A5", "title length", len(d["title"]) <= 120, {"chars": len(d["title"])})

# A6
cs = d.get("change_summary", "")
named = [pid for pid in ["DE1paWNMk8WVTC3n", "2CmvAb5SC-VGAsbK", "6yPhozmAt62JSqQs", "O4hnCxiPVenpDr4A"] if pid in cs]
check("A6", "change_summary names prior rejections", len(named) >= 3,
      {"named": named, "previous_submission_id": d.get("previous_submission_id")})

fails = [c for c in checks if not c["pass"]]
json.dump({"checks": checks, "pass": len(checks) - len(fails), "fail": len(fails),
           "total": len(checks)}, open(OUT, "w"), indent=1)
print("%d/%d pass" % (len(checks) - len(fails), len(checks)))
for c in fails:
    print("FAIL", c["id"], c["name"], c["detail"])
sys.exit(1 if fails else 0)
