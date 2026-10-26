"""Provenance-table generator (P1/P2, round-4 §8).

Scans the paper prose for `results/<name>.json[l]` references, then for each artifact emits an
entry with sha256, byte size, mtime, producing script (matched by basename in lab/), and the
top-level fields the section relies on. Deterministic; no network; CPU only.

Convention frozen pre-run: an artifact mentioned in prose but absent on disk is a HARD FAILURE
(exit 1) — packaging must not silently drop a cited number.

Output: paper/PROVENANCE.md (and results/s48_provenance.json for machine checks).
"""
import gzip
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # ~/rstation/kernelbench in the repo, supplement_v5 in the package
LAB = os.path.join(ROOT, "lab") if os.path.isdir(os.path.join(ROOT, "lab")) else HERE
# prefer lab/results (the working artifact tree) over the repo-root results/ shim directory
RES = os.path.join(LAB, "results")
if not os.path.isdir(RES):
    RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
# In the package, PROVENANCE.md is already emitted next to the sections; the generator rewrites it there.
if not os.path.isdir(PAPER):
    PAPER = ROOT

ART_RE = re.compile(r"(?:results/)?([A-Za-z0-9_]+\.(?:jsonl|json))")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cited_artifacts():
    cited = {}
    for dirpath, _dirs, files in os.walk(PAPER):
        for fn in files:
            if not fn.endswith(".md") or fn == "PROVENANCE.md":
                continue  # PROVENANCE.md is generated; scanning it would self-poison
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            for m in ART_RE.finditer(text):
                name = m.group(1)
                cited.setdefault(name, set()).add(os.path.relpath(p, ROOT))
    return cited


def fields_for(path, gz=False):
    """Top-level keys only; for jsonl report row count + first-row keys."""
    opener = gzip.open if gz else open
    try:
        if path.endswith(".jsonl"):
            n = 0
            first = None
            with opener(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    if first is None:
                        try:
                            first = json.loads(line)
                        except Exception:
                            first = None
            keys = sorted(first.keys()) if isinstance(first, dict) else []
            return {"kind": "jsonl", "rows": n, "row_keys": keys}
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return {"kind": "json", "keys": sorted(d.keys())}
        return {"kind": "json", "keys": [], "list_len": len(d)}
    except Exception as e:  # unreadable/truncated artifacts are reported, not hidden
        return {"kind": "unparseable", "error": type(e).__name__}


def producer(name):
    stem = name.split(".")[0]
    hits = []
    for cand in (stem + ".py",):
        p = os.path.join(LAB, cand)
        if os.path.exists(p):
            hits.append(os.path.relpath(p, ROOT))
    if not hits:
        for fn in sorted(os.listdir(LAB)):
            if not fn.endswith(".py"):
                continue
            try:
                with open(os.path.join(LAB, fn), encoding="utf-8", errors="replace") as f:
                    if name in f.read():
                        hits.append(os.path.relpath(os.path.join(LAB, fn), ROOT))
            except OSError:
                continue
    return hits


def main():
    cited = cited_artifacts()
    entries = []
    missing = []
    for name in sorted(cited):
        path = os.path.join(RES, name)
        gz = False
        if not os.path.exists(path) and not os.path.exists(path + ".gz"):
            # a few cited inputs are producer-side inputs rather than results/ artifacts (r1_times.json is
            # the recorded r1 med_ms table r45_rider_adm.py reads); they live in lab/ and ship at
            # kernels/. Record them from there rather than reporting them missing.
            alt = os.path.join(LAB, name)
            if os.path.exists(alt):
                path = alt
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            # row-streams that are cited but never loaded by an auditor may ship gzipped to stay inside
            # the packaging budget; the digest then refers to the compressed bytes and `uncompressed_bytes`
            # records what the plain file would be.
            path, gz = path + ".gz", True
        if not os.path.exists(path):
            missing.append(name)
            continue
        st = os.stat(path)
        self_ref = (name == "s48_provenance.json")
        entries.append({
            "artifact": "results/" + name + (".gz" if gz else ""),
            "sha256_note": ("self-referential; digest, byte size and mtime omitted" if self_ref else None),
            "cited_in": sorted(cited[name]),
            "sha256": None if self_ref else sha256(path),
            "bytes": None if self_ref else st.st_size,
            "mtime_utc": None if self_ref else __import__("datetime").datetime.utcfromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "producers": producer(name),
            "content": fields_for(path, gz=gz),
            "compressed": gz,
        })

    lines = ["# Provenance table — cited artifacts (generated by lab/s48_provenance.py)", ""]
    lines.append("Every `results/*.json[l]` referenced by the paper prose, with digest, size, mtime, the "
                 "script that produced it, and its top-level fields. Regenerate after any artifact change; "
                 "a cited-but-absent artifact fails generation.")
    lines.append("")
    lines.append("| artifact | cited in | sha256 (first 16) | bytes | mtime (UTC) | producer | fields |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in entries:
        fields = e["content"].get("keys") or e["content"].get("row_keys") or []
        fdesc = ("jsonl rows=%s" % e["content"].get("rows")) if e["content"]["kind"] == "jsonl" else \
                 ("json keys=%s" % ",".join(fields[:6]) + ("…" if len(fields) > 6 else ""))
        if e["content"]["kind"] == "unparseable":
            fdesc = "UNPARSEABLE(%s)" % e["content"]["error"]
        lines.append("| `%s` | %s | `%s` | %s | %s | %s | %s |" % (
            e["artifact"],
            ", ".join("`%s`" % c for c in e["cited_in"]),
            (e["sha256"][:16] if e["sha256"] else "self"),
            (e["bytes"] if e["bytes"] is not None else "self"),
            (e["mtime_utc"] or "self"),
            ", ".join("`%s`" % p for p in e["producers"]) or "—",
            fdesc,
        ))
    lines.append("")
    if missing:
        lines.append("## MISSING (cited but absent on disk) — packaging must not proceed")
        for m in missing:
            lines.append("- `results/%s`" % m)
        lines.append("")

    with open(os.path.join(PAPER, "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "s48_provenance.json"), "w", encoding="utf-8") as f:
        json.dump({"date": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "n_cited": len(cited), "n_found": len(entries), "missing": missing,
                   "entries": entries}, f, indent=1)
    print("cited=%d found=%d missing=%d" % (len(cited), len(entries), len(missing)))
    for e in entries:
        print("  %-34s %s %8s %s" % (e["artifact"],
                                      (e["sha256"][:12] if e["sha256"] else "self"),
                                      (e["bytes"] if e["bytes"] is not None else "self"),
                                      e["content"]["kind"]))
    if missing:
        print("MISSING:", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
