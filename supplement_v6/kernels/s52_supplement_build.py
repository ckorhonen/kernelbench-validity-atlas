"""Build supplement_v6 — the round-5 repair package: runnable producers, artifacts, repo pin, dispositions.

Conventions frozen HERE, before the build (#55). These are the F5b/F6 repair conventions from the DE1paW
review, and each one is asserted rather than promised:

1. RUNNABLE-OR-DECLARED. Every shipped .py that is CPU-only must be RUN from inside the package at build
   time and exit 0. A shipped script that cannot run from the package is not a reproduction path. GPU legs
   are shipped too, but each is listed in README.md under "GPU legs" with the artifact it produced, so a
   reader knows what was measured here rather than re-derivable. v5 shipped producers whose paths assumed
   `kernels/results/` and they all died from inside the package; that is the defect this convention exists
   to prevent. The producers now carry a shared resolver (`_kb_res_dir` / `_kb_open`) that finds `results/`
   beside `kernels/`, and every row-stream reader accepts gzip, so a gzipped artifact is still a runnable
   input.
2. SELF-ERASING CHECKS ARE PINNED. s55 audits r45_rider_adm.json, and the F2 fix rewrote that artifact. A
   check that reads the field it repaired would certify its own repair on the next run, so the pre-repair
   values are declared constants in the script and both ratios are reported.
3. REPO PIN. README.md and PACKAGE_MANIFEST.json carry the public repository URL and the exact commit
   that produced this package, checked at build against `git rev-parse HEAD`. The reviewer's objection was
   that omissions were acknowledged but no accessible location was given; a pinned public commit answers it.
4. Every repo-derived file is a byte-copy of its lab/ counterpart; MANIFEST.sha256 is sha256 over the
   emitted bytes and is verified against the repo source at build. A hash mismatch aborts the build.
5. Only README.md, DROPPED.md, DISPOSITIONS.md, env_manifest.json and PACKAGE_MANIFEST.json are authored.
6. No absolute paths in emitted .py files — asserted, not assumed.
7. Budget: expanded total strictly below 2,000,000 bytes (platform cap).
8. The package is CURATED, not a dump. Omissions are declared in DROPPED.md with reasons.
"""
import gzip, hashlib, json, os, shutil, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "supplement_v6")
BUDGET = 2_000_000

# CPU-only scripts: run from inside the package at build, must exit 0. This is the whole F5b repair.
CPU_RUNNABLE = """
s48_provenance.py s50_prose_audit.py s51_prose_audit.py s52_prose_audit.py s53_intro_audit.py
s54_metadata_audit.py s52_manuscript.py f24_verify.py r45_rider_adm.py r42_analysis.py cf43.py
cushion_screen.py s6_dispatcher_audit.py s55_review_repair.py a8_analysis.py a8b_analysis.py a9_analysis.py
""".split()

# GPU producers: shipped so the measurement is inspectable, not re-derivable on a CPU-only reader.
GPU_PRODUCERS = """
a6_refdistance.py a6b_ksweep.py a6c_conv_tf32pair.py a6d_crossing.py a6e_geometry_algo.py
a6f_geom_repl_convmech.py a7_kernelclass.py a7b_precision_policy.py a8_compress.py a9_compose.py
s56_heldout_contract.py s52_supplement_build.py
""".split()

PREREGS = """
a6_refdistance_prereg.md a7_kernelclass_prereg.md a7b_precision_prereg.md a8_compress_prereg.md
a9_compose_prereg.md
""".split()

# artifacts the shipped auditors load by name plus every round-4/5 evidence leg. Heavy row-streams are
# gzipped: no auditor needs them whole, and s48_provenance reads them through gzip and reports the
# uncompressed structure, so they stay verifiable.
GZIP = {"b4_guard_v1.jsonl", "r1_conv_gated.jsonl", "gate_stability.jsonl", "m3_curve.json",
        "e2_splitk_v2.json", "e1b_order_v2.json", "e2_v3_err.json", "e2b_floor.json",
        "b2_direction_audit.json", "e1_triad_day2.json", "e1_triad.json", "m3_r42.jsonl",
        "w4_heldout.json.jsonl", "a8_compress.json", "r42_gridfix.jsonl", "m3_r42.json"}

RESULTS = """
calibration.json e1_triad.json e1_triad_day2.json e2_v3_err.json e2b_floor.json a6d_crossing.json
a7_kernelclass.json a7b_precision_policy.json cf43.json cushion_screen.json f24_verify.json
pop_confirm.json pop_freeze.json r1_conv_gated_summary.json r45_rider_adm.json s6_dispatcher_audit.json
r42_analysis.json r42_gridfix_summary.json b5_stage1.json b2_direction_audit.json b4_direction_audit.json
b2_failure_corpus.jsonl gate_stability_summary.json m3_r42_analysis.json m3_summary.json b5_stage2r.json
r39_text_support.json a8_compress_v2.json a8_analysis.json a8b_analysis.json a9_compose.json
a9_analysis.json s48_provenance.json s50_prose_audit.json s51_prose_audit.json s52_prose_audit.json
s53_intro_audit.json s54_metadata_audit.json s55_review_repair.json s56_heldout_contract.json
s56_analysis.json a6_refdistance.json a6b_ksweep.json a6c_conv_tf32pair.json a6e_geometry_algo.json
a6f_geom_repl_convmech.json e1_analysis_v3.json e3_gridgeom.json m3_curve.json r42_gridfix.jsonl
s46_e2_recount.json w4_v1.json s48_prose_audit.json b1_gatezero_v2.json
r1_conv_gated.jsonl gate_stability.jsonl
b4_guard_v1.jsonl e1b_order_v2.json e2_splitk_v2.json m3_r42.jsonl m3_r42.json
w4_heldout.json.jsonl a8_compress.json
""".split()

# r1_times.json lives at lab/ (not lab/results/) and r45_rider_adm.py requires it; the reviewer flagged
# exactly this ("requires absent r1_times.json"). It ships at kernels/r1_times.json, beside the script.
EXTRA_KERNEL_INPUTS = ["r1_times.json"]
# the paper's submission metadata: s54 audits the abstract against the manuscript, so it must be inside
# the package or s54 cannot run there (v5 shipped only the supplement's own metadata and s54 died on it).
ROOT_FILES = {"submission_metadata_v6.json": "submission_metadata_v6.json"}

PAPER = ["REFRAME_SKELETON.md", "PROVENANCE.md", "MANUSCRIPT.md"]
# Four archived full-stage reviews and two triages: §1, §7 and §8 cite them as evidence and s53 audits
# them, so they ship rather than being referenced from outside the package.
REJECTIONS = ["rejection_review_2CmvAb.md", "rejection_review_6yPhozm.md", "rejection_review_O4hnCx.md",
              "rejection_review_DE1paW.md", "rejection_triage_2CmvAb.md",
              "rejection_triage_DE1paW.md"]
SECTIONS = ["s1_introduction.md", "s2_reference_axis.md", "s3_atlas.md", "s3b_conv_anchor.md",
            "s4_certificates.md", "s5_tolerance.md", "s6_selector.md", "s7_threats.md",
            "s8_reproduction.md", "s9_related.md"]
ATLAS = ["ATLAS_A6.md", "ATLAS_A7.md", "ATLAS_A8.md", "ATLAS_A9.md", "ATLAS_S56.md",
         "SCHOOL_COMPLETION.md"]

# outputs the packaged scripts rewrite when run from inside the package; regenerated rather than
# byte-copied, and their shipped state is what MANIFEST.sha256 covers.
REGENERATED = {"paper/PROVENANCE.md", "paper/MANUSCRIPT.md",
               "results/s48_provenance.json", "results/s50_prose_audit.json",
               "results/s51_prose_audit.json", "results/s52_prose_audit.json",
               "results/s53_intro_audit.json", "results/s54_metadata_audit.json",
               "results/s55_review_repair.json", "results/f24_verify.json", "results/r42_analysis.json",
               "results/cf43.json", "results/cushion_screen.json", "results/r45_rider_adm.json",
               "results/s6_dispatcher_audit.json", "results/a8_analysis.json",
               "results/a8b_analysis.json", "results/a9_analysis.json"}

errors = []


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(rel):
    for base in ("lab/results", "results", "lab"):
        p = os.path.join(ROOT, base, rel)
        if os.path.exists(p):
            return p
    return None


def git_pin():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        errors.append("git rev-parse failed — the repo pin is a required field")
        return None
    return r.stdout.strip()


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    pin = git_pin()

    emitted = {}
    for name in RESULTS:
        sp = find(name)
        if sp is None:
            errors.append(f"missing result artifact {name}")
            continue
        emitted[f"results/{name}"] = sp
    for name in CPU_RUNNABLE + GPU_PRODUCERS:
        sp = find(name)
        if sp is None:
            errors.append(f"missing producer {name}")
            continue
        emitted[f"kernels/{name}"] = sp
    for name in PREREGS:
        sp = find(name)
        if sp is None:
            errors.append(f"missing prereg {name}")
            continue
        emitted[f"kernels/{name}"] = sp
    for name in EXTRA_KERNEL_INPUTS:
        sp = os.path.join(ROOT, "lab", name)
        if not os.path.exists(sp):
            errors.append(f"missing kernel input {name}")
            continue
        emitted[f"kernels/{name}"] = sp
    for dest, name in ROOT_FILES.items():
        sp = os.path.join(ROOT, name)
        if not os.path.exists(sp):
            errors.append(f"missing root file {name}")
            continue
        emitted[dest] = sp
    for name in PAPER:
        sp = os.path.join(ROOT, "paper", name)
        if not os.path.exists(sp):
            errors.append(f"missing paper file {name}")
            continue
        emitted[f"paper/{name}"] = sp
    for name in SECTIONS:
        sp = os.path.join(ROOT, "paper", "sections", name)
        if not os.path.exists(sp):
            errors.append(f"missing section {name}")
            continue
        emitted[f"paper/sections/{name}"] = sp
    for name in REJECTIONS:
        sp = os.path.join(ROOT, name)
        if not os.path.exists(sp):
            errors.append(f"missing rejection record {name}")
            continue
        emitted[f"rejections/{name}"] = sp
    for name in ATLAS:
        sp = os.path.join(ROOT, name)
        if not os.path.exists(sp):
            errors.append(f"missing atlas entry {name}")
            continue
        emitted[f"atlas/{name}"] = sp

    shipped = {}
    for rel, sp in list(emitted.items()):
        dest = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.basename(sp) in GZIP:
            dest = dest + ".gz"
            with open(sp, "rb") as f, gzip.open(dest, "wb", compresslevel=9) as g:
                shutil.copyfileobj(f, g)
        else:
            shutil.copyfile(sp, dest)
        shipped[os.path.relpath(dest, OUT)] = sp

    for rel, text in (("README.md", build_readme(pin)), ("DROPPED.md", build_dropped()),
                      ("DISPOSITIONS.md", build_dispositions()),
                      ("env_manifest.json", build_env()), ("PACKAGE_MANIFEST.json", build_meta(pin))):
        with open(os.path.join(OUT, rel), "w") as f:
            f.write(text)

    for rel in shipped:
        if not rel.endswith(".py") or rel == "kernels/s52_supplement_build.py":
            continue  # the builder's own assertion lines contain the literal paths it is checking for
        for line in open(os.path.join(OUT, rel)).read().splitlines():
            if "/Users/" in line or "/home/" in line:
                errors.append(f"absolute path in {rel}: {line.strip()[:90]}")

    # byte-copy verification against repo sources. REGENERATED files are compared for regenerability
    # rather than byte equality, because running the packaged scripts rewrites them.
    for rel, sp in shipped.items():
        if rel.endswith(".gz"):
            with gzip.open(os.path.join(OUT, rel), "rb") as g:
                got = hashlib.sha256(g.read()).hexdigest()
            if got != sha(sp):
                errors.append(f"gzip round-trip mismatch {rel}")
        elif rel in REGENERATED:
            continue
        elif sha(os.path.join(OUT, rel)) != sha(sp):
            errors.append(f"hash mismatch {rel}")

    print(f"emitted={len(shipped)} bytes(before regeneration)="
          f"{sum(os.path.getsize(os.path.join(OUT, r)) for r in shipped)}")
    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    # CONVENTION 1: run every CPU-only script from inside the package. This is the F5b repair.
    print("running packaged CPU scripts from inside the package:")
    for script in CPU_RUNNABLE:
        p = os.path.join(OUT, "kernels", script)
        if not os.path.exists(p):
            errors.append(f"script {script} not in package")
            continue
        r = subprocess.run([sys.executable, p], cwd=os.path.join(OUT, "kernels"),
                           capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"  {script}: exit={r.returncode} {tail[:100]}")
        if r.returncode != 0:
            errors.append(f"packaged script {script} failed (exit {r.returncode}): "
                          f"{(r.stderr.strip().splitlines() or [''])[-1][:160]}")

    # determinism on the provenance table: it records mtimes, so assert everything except the stamp.
    # Run it twice back-to-back here, after every other script has settled its output, because a table
    # built before another script rewrote an artifact would differ for a legitimate reason.
    prov_json = os.path.join(OUT, "results", "s48_provenance.json")
    prov_py = os.path.join(OUT, "kernels", "s48_provenance.py")
    subprocess.run([sys.executable, prov_py], cwd=os.path.join(OUT, "kernels"), capture_output=True, text=True)
    before = json.load(open(prov_json))
    subprocess.run([sys.executable, prov_py], cwd=os.path.join(OUT, "kernels"), capture_output=True, text=True)
    after = json.load(open(prov_json))
    for d in (before, after):
        d.pop("date", None)
    if before != after:
        errors.append("provenance table not reproducible modulo its date stamp")

    # CONVENTION 2: the s55 F2 pin must still show the defect it was written to record.
    s55 = json.load(open(os.path.join(OUT, "results", "s55_review_repair.json")))
    for ref in ("def", "ieee", "64"):
        pre = s55["F2_regret_admission"]["per_ref"][ref]["pre_repair_over_recomputed_mean_ratio"]
        if not (0.0095 < pre < 0.0105):
            errors.append(f"s55 F2 pin broken at {ref}: pre-repair ratio {pre}")

    # CONVENTION 3: the pin in the metadata must be the commit actually built.
    meta = json.load(open(os.path.join(OUT, "PACKAGE_MANIFEST.json")))
    if pin and meta.get("repository_commit") != pin:
        errors.append("repository_commit in PACKAGE_MANIFEST.json does not match HEAD")

    manifest = []
    total = 0
    for dirpath, _, files in os.walk(OUT):
        for fn in sorted(files):
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, OUT)
            if rel == "MANIFEST.sha256":
                continue
            total += os.path.getsize(p)
            manifest.append((sha(p), rel))
    manifest.sort(key=lambda x: x[1])
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
        for h, rel in manifest:
            f.write(f"{h}  {rel}\n")
    if total > BUDGET:
        errors.append(f"package expanded {total} bytes exceeds budget {BUDGET}")

    print(f"final files={len(manifest)} expanded_bytes={total}")
    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
    print("build OK")


def build_readme(pin):
    pin = pin or "UNPINNED"
    return f"""# Supplement v6 -- validity-domain atlas: runnable producers, artifacts, pinned source

Built 2026-10-24 (tick #55). Supersedes supplement_v5.

## What changed from v5, and why

The DE1paW review found (F5b/F6) that v5's advertised reproduction paths did not run: `r45_rider_adm.py`
required an absent `r1_times.json`, `f24_verify.py` required an absent `m3_r42.jsonl`, several scripts
expected uncompressed inputs under `kernels/results/`, the A6/A7 measurement producers were absent, and no
repository location was given to retrieve what was missing. v5's five auditors did run; nothing else did.
**v6 runs every CPU-only script it ships, from inside the package, at build time** (the build fails
otherwise), ships the A6/A7 producers and the S55/S56 legs, carries `r1_times.json` where `r45` looks for
it, and pins the public source:

    https://github.com/ckorhonen/kernelbench-validity-atlas
    commit {pin}

That commit is the exact tree this package was built from. Anything omitted here is retrievable there at
that revision, so no essential evidence depends on private access.

## Layout

- `kernels/` — producers and auditors, preregs at head, no absolute paths. Scripts resolve artifacts
  through a shared helper that looks for `results/` beside `kernels/`, and every row-stream reader accepts
  a gzipped artifact, so the package's own layout is a runnable layout.
  CPU-only, verified to run from inside this package at build:
  `s48_provenance.py`, `s50_prose_audit.py` (19 checks), `s51_prose_audit.py` (31), `s52_prose_audit.py`
  (16), `s53_intro_audit.py` (15), `s54_metadata_audit.py` (6), `s52_manuscript.py`, `f24_verify.py`,
  `r45_rider_adm.py`, `r42_analysis.py`, `cf43.py`, `cushion_screen.py`, `s6_dispatcher_audit.py`,
  `s55_review_repair.py`, `a8_analysis.py`, `a8b_analysis.py`, `a9_analysis.py`.
  GPU legs (shipped for inspection; they reproduce only on a GB10 with this stack): `a6_refdistance.py`,
  `a6b_ksweep.py`, `a6c_conv_tf32pair.py`, `a6d_crossing.py`, `a6e_geometry_algo.py`,
  `a6f_geom_repl_convmech.py`, `a7_kernelclass.py`, `a7b_precision_policy.py`, `a8_compress.py`,
  `a9_compose.py`, `s56_heldout_contract.py`. Each produced an artifact shipped under `results/`, so a
  reader without a GPU can audit the result without re-running the measurement.
- `results/` — the artifacts the auditors load, plus every artifact the paper cites. Row-level corpora ship
  where a claim is per-row. Fifteen heavy row-streams ship gzipped (`*.gz`) to stay inside the budget; the
  readers handle them transparently and `s48_provenance.py` reports the compressed digest plus the
  uncompressed row/field structure. The list is in `DROPPED.md`.
- `paper/` — the complete section set (§1-§9), the reframe skeleton, `MANUSCRIPT.md` (90,896 bytes,
  reassembled from the repaired sections), and `PROVENANCE.md`.
- `atlas/` — `ATLAS_A6`, `ATLAS_A7`, `ATLAS_A8`, `ATLAS_A9`, `ATLAS_S56`, `SCHOOL_COMPLETION`.
- `rejections/` — four archived full-stage reviews and two triages, including the DE1paW review this
  package answers.
- `DISPOSITIONS.md` — the per-finding disposition map for DE1paW. Every finding, what was found, what
  changed, and where the change is checkable.

## Verifying this package

    cd kernels
    python3 s50_prose_audit.py     # and s51, s52, s53, s54, s48_provenance, s55, f24, r45, r42, cf43, s6,
                                   #     a8_analysis, a8b_analysis, a9_analysis, s52_manuscript
    shasum -a 256 -c MANIFEST.sha256

Every repo-derived file is a byte-copy of its lab/ counterpart, verified against the repository at build
time. Eighteen files are REGENERATED rather than copied, because the shipped scripts rewrite them when run;
the build runs them, then takes the manifest over the final shipped state, so `shasum -a 256 -c` passes
exactly as shipped. Only README.md, DROPPED.md, DISPOSITIONS.md, env_manifest.json and
PACKAGE_MANIFEST.json are authored.

Two caveats, stated rather than hidden. `s48_provenance.py` records artifact mtimes, so re-running it
rewrites `results/s48_provenance.json` and `paper/PROVENANCE.md` and those manifest lines will no longer
match: verify the manifest BEFORE running the auditors, and verify the auditors by running them.
`s55_review_repair.py` audits `r45_rider_adm.json`, which the F2 repair rewrote, so it carries the
pre-repair values as declared constants and reports both ratios (0.0100 before, 1.0 after); without that
pin the check would certify its own repair.

## Headline numbers a reader can regenerate from this package

- Reference budget consumption (a6/a7/a8): fp32 matmul crosses rho = 1 between K 3608 and 3632 at
  M=N=1024; fp32 conv splits into a winograd class (rho 10.195, 0.52% of elements over budget,
  bit-identical under every precision flag) and a TF32 tensor-op class (rho 251.69, 78.2% over budget,
  removed 99.9% by `allow_tf32=False`).
- Compression (a8/a8b): rho_max out-of-fold Spearman 0.833 from geometry alone, 0.986 with the engine
  class; frac_over bulk accuracy 0.7917 / 1.0; class predictability 0.6042 vs 0.5625 majority.
- Composition (a9, s55): rho_end/max(rho_A,rho_B) = 0.981/1.507/1.537/1.000/1.971/2.480. One chain inside
  1.0-1.5x, five inside a factor of 2. The union rule PASSES 4/6 and fails 2 (c5, c6). c5 is two
  individually admissible stages (0.0689, 0.6563) composing to 1.2939, past budget.
- Delivery (s56): reduction gate hand deliveries 11/12 admissible at +4.13% (SE 1.40) time cost against
  the best admissible arm; conv 0/4 admissible at budget ratios 17,431-23,783, so the conv delivery claim
  is withdrawn.
- Regret under admission (s55): the published "|A2-A1| <= 0.025 pp" is withdrawn. Corrected pp means and
  max|.| are +0.21/2.44 (def), -0.66/25.46 (ieee), -4.43/41.20 (fp64).
- d_i denominator (s55): oracle-normalized +11.87/+12.08/+10.30 pp with sd 29.25/23.24/20.32 pp;
  compile-normalized +6.58/+8.17/+7.19 pp with sd 16.28/14.13/13.02 pp.

## Scope statement

One architecture (GB10/sm_120), one fp64 build, one torch version (2.14.1+cu130), designed lattices rather
than benchmark-distribution samples, unlocked clocks. No transfer claim to another architecture is made.
"""


def build_dropped():
    return """# Declared omissions

v6 carries the artifacts the shipped auditors load, plus every round-4 and round-5 evidence leg. Files
present in supplement_v4 or v5 but absent here, with reasons. Nothing was dropped silently and nothing
unique was dropped without a shipped replacement that regenerates or summarizes it. Everything else is at
the pinned commit named in README.md.

- `results/atlas_population.json` (381 KB): upstream sampling record for the population-collapse claim. Its
  frozen threshold ships as `pop_freeze.json` and its confirmatory result as `pop_confirm.json`, so the
  claim stays auditable end to end.
- The `.gz` twins: `e1_triad.json`, `e1_triad_day2.json`, `b4_guard_v1.jsonl`, `r1_conv_gated.jsonl`,
  `gate_stability.jsonl`, `m3_curve.json`, `e2_splitk_v2.json`, `e1b_order_v2.json`, `e2_v3_err.json`,
  `e2b_floor.json`, `b2_direction_audit.json`, `m3_r42.jsonl`, `w4_heldout.json.jsonl`, `a8_compress.json`,
  `r42_gridfix.jsonl` are the same bytes as the plain files at the pinned commit, compressed at level 9.
  They ship compressed because the expanded payload is capped at 2 MB; the packaged readers open them
  through gzip, so every script that needs them still runs from inside this package.
- `results/m3_curve.json`, `results/e2_splitk*.json`, `results/e1b_order*.json`, `results/b4_guard_v*.jsonl`,
  `results/m2_conv_day2.json*`: full row streams for claims whose audited summaries ship here.
- v4 `kernels/` chain runners that are GPU-only and whose outputs' audited summaries ship here
  (`b1_gatezero_v2.py`, `b4_guard_v1.py`, `w4_heldout.py`, `d2_conv.py`, `m1_bt_deployed.py`,
  `m2_analyze.py`, `m3_r42.py`, `r1_conv_gated.py`, `r41_gate_stability.py`, `r42_gridfix.py`,
  `w3_conv_gemm.py`, `w3_ladder.py`, `w4_dispatcher_v0.py`, `w4_dispatcher_v1.py`, `e1_triad.py`,
  `e1_triad_day2.py`, `e1_analyze.py`, `e1_analyze_v3.py`, `e2_splitk.py`, `e2_splitk_v2.py`,
  `e2_splitk_v3.py`, `e2_err_regen.py`, `e2b_floor.py`, `e3_gridgeom.py`, `e1b_order.py`, `b3_leg23.py`,
  `b3_stage1.py`, `d1_cellaffine.py`, `d2_freeze.py`, `d3_freeze.py`, `d3_margin.py`, `nw_sweep.py`,
  `a2_day2.py`, `cd_discrim.py`, `cd_freeze.py`, `m0_residual.py`, `a2_loo.py`, `a1_orthdecile.py`).
  They stay repo-resident at the pinned commit. The §8 auditor set, every round-4 leg, and the round-5
  repair legs do ship.
- `corpus/b2_failure_corpus.jsonl`: ships once, under `results/`, which is the path §5 and the auditors read.
"""


def build_dispositions():
    return """# Per-finding disposition map — DE1paWNMk8WVTC3n

Rejected 2026-10-24T14:52:46Z, stage=full. Review archived at `rejections/rejection_review_DE1paW.md`;
triage at `rejections/rejection_triage_DE1paW.md`. Each row states the finding, what I found when I
recomputed it, and where the change is checkable in this package. Two findings were interpretive and
conceded; five were arithmetic and reproduced exactly from committed records.

| # | Finding | Disposition | Where to check |
|---|---|---|---|
| F1 | The admission rule does not certify the benchmark contract (a row can pass the reference and fail the anchor) | **CONCEDED.** Restructured §4/§8.4 into three separately-evaluated criteria: benchmark agreement (binary contract), anchor accuracy (diagnostic), declared relaxed application contract (must be named) | `paper/sections/s4_certificates.md`, `s8_reproduction.md`; `s52_prose_audit.py` A15/A16 |
| F2 | Regret invariance under admission is numerically false | **VERIFIED, my bug.** `r45_rider_adm.py` stored fractional deltas under `paired_delta_pp_*` (no x100) and used a signed max, so -41.2 pp read as 0.0. Published "<= 0.025 pp" WITHDRAWN. Script fixed (pp fields + max_abs + min) | `kernels/r45_rider_adm.py`, `results/r45_rider_adm.json`, `results/s55_review_repair.json` F2 (reports pre-repair ratio 0.0100 and post-repair 1.0) |
| F3 | Composition conclusions overstate and reverse their evidence | **VERIFIED.** Ratios 0.981/1.507/1.537/1.000/1.971/2.480: ONE chain inside 1.0-1.5x, FIVE inside 2x. Prose also wrote "union rule fails 4/6" where the artifact says it PASSES 4/6. c5 promoted from footnote to named counterexample | `paper/sections/s5_tolerance.md` §5.2; `s52_prose_audit.py` A15/A16 compute the factor and c5 from `a9_compose.json` rather than matching strings |
| F4 | Claimed validity domains and negative findings exceed the tests | **VERIFIED.** Rank is not magnitude: geometry-only median multiplicative error 6.84x, max 156.7x. Crossing is M=N=1024-specific (rho 0.648 at M=256, 2.777 at M=4096) | `paper/sections/s4_certificates.md` §4.5, `s9_related.md`; `results/s55_review_repair.json` F4/F4b |
| F5 | Performance and economic conclusions contain metric errors (d_i denominator) | **VERIFIED.** `r42_analysis.py:92` divides by oracle, not compile; §3.3 said compile. Oracle-denom +11.87/12.08/10.30 pp sd 29.25/23.24/20.32 pp; compile-denom +6.58/8.17/7.19 pp sd 16.28/14.13/13.02 pp. The published 0.29/0.23/0.20 were oracle-denom sd in FRACTION units printed as pp | `paper/sections/s3b_conv_anchor.md`; `results/s55_review_repair.json` F5 |
| F5b | Reproduction paths incomplete; A6/A7 producers absent; no repository location | **VERIFIED.** v5's auditors ran but every producer died from inside the package. v6 runs all 17 CPU scripts from inside the package at build and pins the public commit | `README.md` (repo pin), build log; this table |
| F6 | Auditors are consistency checks, not regenerations; A12 string-matched the wrong number | **VERIFIED.** s50's F1 now RECOMPUTES rather than reading the stored field; s52's A15/A16 compute the factor and c5 from artifacts | `kernels/s50_prose_audit.py`, `s52_prose_audit.py`; `results/s50_prose_audit.json`, `s52_prose_audit.json` |
| F6b | §6's economic claim was argued from numerical error scales compared to regret gaps — different currencies | **CONCEDED and DISCHARGED.** S56 held-out per-arm contract table, prereg frozen pre-run: reduction 11/12 admissible at +4.13% (SE 1.40, t 2.94) time cost vs the best admissible arm; conv 0/4 admissible at budget ratios 17,431-23,783, so the conv delivery claim is WITHDRAWN. Also explains s6's 0.0 pass-frac: pessimism for reduction, true negative for conv | `kernels/s56_heldout_contract.py`, `results/s56_heldout_contract.json`, `results/s56_analysis.json`, `atlas/ATLAS_S56.md`; `paper/sections/s6_selector.md` §6.2 |
| — | Novelty: Barabasz et al. and PyTorch's numerical-accuracy documentation are closer than the cited distribution-shift work | **CONCEDED.** §9 now cites Barabasz [7] and PyTorch [8] and §9.4 lists the non-novel ingredients outright, stating the bounded concrete advance instead | `paper/sections/s9_related.md` |
"""


def build_env():
    return json.dumps({
        "python": "3.12.3", "torch": "2.14.1+cu130", "cuda": "13.0", "cudnn": 92400,
        "device": "NVIDIA GB10", "cc": "12.1", "triton": "3.8.0", "tf32_matmul": False,
        "precision": "fp32 highest (matmul); cudnn.allow_tf32=True in the measured rows, recorded per row",
        "driver": "580.159.03", "packaged": "2026-10-24 tick #55",
        "note": "the CPU-only scripts are dependency-free and run on any Python 3.9+; the GPU legs "
                "reproduce on any GB10 with this stack",
    }, indent=1) + "\n"


def build_meta(pin):
    return json.dumps({
        "supplement": "v6",
        "built": "2026-10-24",
        "supersedes": "supplement_v5",
        "repository": "https://github.com/ckorhonen/kernelbench-validity-atlas",
        "repository_commit": pin,
        "previous_submission_id": "DE1paWNMk8WVTC3n",
        "paper_sections": ["s1", "s2", "s3", "s3b", "s4", "s5", "s6", "s7", "s8", "s9"],
        "audits": {"s50": "19/19", "s51": "31/31", "s52": "16/16", "s53": "15/15", "s54": "6/6",
                   "provenance": "44/44"},
        "packaged_scripts_run_from_inside_package": len(CPU_RUNNABLE),
        "declared_retractions": 4,
        "curation": "results/ carries the artifacts the shipped auditors load plus every round-4/5 leg; "
                    "omissions listed in DROPPED.md, all retrievable at the pinned commit",
        "dispositions": "DISPOSITIONS.md maps every DE1paW finding to its change and its checkable location",
    }, indent=1) + "\n"


if __name__ == "__main__":
    main()
