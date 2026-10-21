"""CD freeze -- conv-discriminator preregistration (fp64-conv2d / forced-algo).

Written BEFORE any GPU number (tick #32, 2026-10-21). Closes the last recorded
untested explanature from d3 (DIARY 1002): conv-side transform (winograd-class)
error. d3 already showed: fp64-accum gemm arm FAIL 0/116, t*64 within ~5% of
t*32 => gap survives exact accumulation, structural. NEVER TESTED: the conv
side's own fp32 consistency.

Universe: the 116 kept d2 rows VERBATIM (recorder-wrapped builders, tie-out vs
committed results/d2_conv.jsonl asserted before any new number -- d3 pattern
verbatim). All distances measured against the exact fp64 conv reference
conv64 = F.conv2d(x.double(), W.double(), b.double()), denominator symmetric
1+|conv64| (t* convention verbatim from d3: max over elements of d/(1+|ref|)).

Arms (frozen):
  tA1 = t*(conv32 vs conv64)   -- does fp32 conv deviate from exact conv?
  tA2 = t*(gemm64raw vs conv64) -- exact-arith agreement sanity (expect <=eps)
  tB1 = t*(gemm32 vs conv64)    -- descriptive complement (d3 t64 was the
       cast-to-fp32 deployable vs conv32; NOT the same quantity)
  paths p in {default, cudnn.benchmark=True(restored), channels_last}:
       tP = t*(path_out vs conv64); spread=(max-min)/median over tP;
       degenerate if all path outputs mutually torch.equal.

Verdicts (mechanical, thresholds frozen HERE):
  V1 CONFIRM conv-side: median(tA1) >= 1e-3  => fp32 conv owns the gap
     (transform-conditioning class) => RECONSIDER-IF branch: reopen scoped
     tolerance-domain lane note per #20/#24 registration.
  V2 DEAD conv-side: median(tA1) <= 2e-4 => conv32 exact-consistent; with d3's
     gemm-side result the gap is attributed joint/other; account closed, atlas
     stays closed, instrument note recorded.
  V3 partial: 2e-4 < med(tA1) < 1e-3 => recorded only, no lane, bounded stop.
  Rider R1 path: median spread <= 0.10 implementation-invariant; > 0.10
     path-sensitive (note). Degenerate paths void R1, recorded as such.
  Flag F1: median(tA2) > 1e-6 => exact-arith disagreement (unexpected; halt
     interpretation, record only).

No new freedom after this file: thresholds, arms, universe, t* convention all
frozen here; runner only executes. Env pin: fp32, TF32 off both, cuda,
torch 2.14.1+cu130 sm_120 (d2 pin verbatim).
"""
import json, hashlib, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

shas = {}
for key, fn in (("d2_freeze", "d2_freeze.json"), ("d2_conv", "d2_conv.jsonl"),
                ("guard_v1", "guard_v1.json")):
    shas[key] = hashlib.sha256(open(os.path.join(RES, fn), "rb").read()).hexdigest()

fz = dict(
    when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    universe="116 kept d2 rows verbatim (d3 recorder pattern), tie-out vs d2_conv.jsonl",
    reference="conv64 = F.conv2d fp64; all t* = max d/(1+|conv64|)",
    arms=dict(tA1="conv32 vs conv64", tA2="gemm64raw vs conv64",
              tB1="gemm32 vs conv64 (descriptive)",
              paths=["default", "cudnn.benchmark=True", "channels_last"]),
    thresholds=dict(confirm_tA1=1e-3, dead_tA1=2e-4, exact_agree=1e-6,
                    path_spread=0.10),
    verdicts=dict(V1="med tA1>=1e-3 => conv-side CONFIRMED, reopen note",
                  V2="med tA1<=2e-4 => conv-side DEAD, atlas closure stands",
                  V3="else partial, recorded only",
                  R1="path spread<=0.10 invariant; degenerate voids",
                  F1="med tA2>1e-6 => halt interpretation"),
    env_pin=dict(dtype="fp32", tf32=False, device="cuda"),
    shas=shas,
)
path = os.path.join(RES, "cd_freeze.json")
if os.path.exists(path):
    raise SystemExit("cd_freeze.json exists -- freeze is immutable, aborting")
json.dump(fz, open(path, "w"), indent=1)
print(json.dumps(fz, indent=1)[:400])
print("frozen", path)
