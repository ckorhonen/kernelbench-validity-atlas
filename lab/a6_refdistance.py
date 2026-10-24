"""A6 reference-distance (rho) census — tick #49, 2026-10-24.

PREREG: lab/a6_refdistance_prereg.md, committed dbfd600 BEFORE this script ran.
Definition (verbatim from prereg): rho(R) = max over output elements of
|R_def - R_64| / (1e-4 + 1e-4*|R_64|), i.e. the largest fraction of the benchmark's
own tolerance budget that the reference itself consumes.

Families (declared by rule in the prereg, not by record availability):
  conv2d  N=2 H=5 C=32 k3 s1 p1 fp32   (CONV_POINTS, lab/w4_heldout.py:76)
  matmul  M=1024 K=8192 N=1024 fp32    (RED_KS/RED_MS scale)
  b1      H=8192 B=2048 K=1024... no: b1 anchor is (H,B,K)=(8192,2048,8192)
          per lab/b1_gatezero.py ANCHOR -- prereg parenthetical transcription slip
          (it said H=2048,B=8192,K=1024); the script is authoritative and the slip is
          recorded in the results header, substance unaffected.

Predictions (prereg): conv rho>=2 ; matmul rho<0.1 ; b1 rho<0.01.
Kills (prereg): (1) b1 rho>0.1 => no table; (2) family with rho<0.1 whose verdicts flip
=> rho not the mechanism; (3) determinism assert.

Also recorded: runtime allow_tf32 state for cudnn/matmul BEFORE any change, rho_ieee with
TF32 explicitly disabled (conv), med/p99 of the ratio, absolute max, |R_64|max, and a
repeat of R_def to assert determinism.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "a6_refdistance.json")
DATE = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
dev = "cuda"
ATOL = RTOL = 1e-4


def rho_stats(a_def, a_64):
    d = (a_def.double() - a_64).abs()
    denom = ATOL + RTOL * a_64.abs()
    ratio = (d / denom).flatten()
    return {
        "rho": float(ratio.max()),
        "med": float(ratio.median()),
        "p99": float(torch.quantile(ratio[: min(ratio.numel(), 4_000_000)].float(), 0.99)),
        "abs_max": float(d.max()),
        "ref64_absmax": float(a_64.abs().max()),
        "n_elems": int(ratio.numel()),
        "frac_over_budget": float((ratio > 1).double().mean()),
    }


rows = []
hdr = {
    "date": DATE,
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(0),
    "cap": list(torch.cuda.get_device_capability(0)),
    "allow_tf32_cudnn_at_start": bool(torch.backends.cudnn.allow_tf32),
    "allow_tf32_matmul_at_start": bool(torch.backends.cuda.matmul.allow_tf32),
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "prereg": "lab/a6_refdistance_prereg.md @ dbfd600",
}

# ---------------- conv2d ----------------
N, H, C = 2, 5, 32
torch.manual_seed(1234 + H * 100 + N * 10 + C)  # harness convention
x = torch.randn(N, C, H, H, device=dev)
w = torch.randn(C, C, 3, 3, device=dev)
r_def = F.conv2d(x, w, padding=1)
r_def2 = F.conv2d(x, w, padding=1)
det = bool(torch.equal(r_def, r_def2))
r_64 = F.conv2d(x.double(), w.double(), padding=1)
row = {"family": "conv2d", "shape": {"N": N, "H": H, "C": C, "k": 3, "s": 1, "p": 1},
       "det_repeat_equal": det, "tf32_state": "default"}
row.update(rho_stats(r_def, r_64))

# ieee arm: TF32 explicitly disabled
prev_cudnn = torch.backends.cudnn.allow_tf32
torch.backends.cudnn.allow_tf32 = False
r_ieee = F.conv2d(x, w, padding=1)
torch.backends.cudnn.allow_tf32 = prev_cudnn
ie = rho_stats(r_ieee, r_64)
row["rho_ieee"] = ie["rho"]
row["ieee_abs_max"] = ie["abs_max"]
row["ieee_med"] = ie["med"]
rows.append(row)

# ---------------- matmul ----------------
M, K, Nn = 1024, 8192, 1024
torch.manual_seed(4321)
A = torch.randn(M, K, device=dev)
B = torch.randn(K, Nn, device=dev)
m_def = A @ B
m_def2 = A @ B
detm = bool(torch.equal(m_def, m_def2))
m_64 = A.double() @ B.double()
rowm = {"family": "matmul", "shape": {"M": M, "K": K, "N": Nn},
        "det_repeat_equal": detm,
        "tf32_state": "default(matmul allow_tf32=%s, f32_prec=%s)" % (
            torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())}
rowm.update(rho_stats(m_def, m_64))
rows.append(rowm)

# ---------------- b1 factored-pair reference (b1 anchor cell) ----------------
Hb, Bb, Kb = 8192, 2048, 8192  # lab/b1_gatezero.py ANCHOR = (H,B,K)
seed = 1237


def ref_fwd(x_, W_, b_, s_, dt=None):
    if dt is not None:
        x_, W_, b_, s_ = x_.to(dt), W_.to(dt), b_.to(dt), s_.to(dt)
    y = F.linear(x_, W_, b_) - s_
    y = torch.mean(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = F.gelu(y)
    return y + x_


torch.manual_seed(seed)
lin = torch.nn.Linear(Kb, Hb)
Wb = lin.weight.detach().float().cuda()
bb = lin.bias.detach().float().cuda()
g = torch.Generator().manual_seed(seed + 1)
sb = torch.randn(Hb, generator=g).float().cuda()
xb = torch.rand(Bb, Kb, generator=torch.Generator().manual_seed(seed + 2)).float().cuda()
b_def = ref_fwd(xb, Wb, bb, sb)
b_def2 = ref_fwd(xb, Wb, bb, sb)
detb = bool(torch.equal(b_def, b_def2))
b_64 = ref_fwd(xb, Wb, bb, sb, dt=torch.float64)
rowb = {"family": "b1_factored_pair", "shape": {"H": Hb, "B": Bb, "K": Kb},
        "det_repeat_equal": detb,
        "note": "prereg parenthetical said (H=2048,B=8192,K=1024); script ANCHOR (H,B,K)=(8192,2048,8192) is authoritative"}
rowb.update(rho_stats(b_def, b_64))
rows.append(rowb)

# ---------------- verdict per frozen prereg ----------------
conv_rho = rows[0]["rho"]
mm_rho = rows[1]["rho"]
b1_rho = rows[2]["rho"]
kill1 = b1_rho > 0.1
if kill1:
    verdict = "KILL1_NEGATIVE_CONTROL_FAILED_no_table"
elif conv_rho >= 2 and mm_rho < 0.1:
    verdict = "PRIMARY_family_dependent_undeclared_default"
elif conv_rho >= 2 and mm_rho >= 1:
    verdict = "BROADER_reference_realizations_exceed_budget_needs_second_family"
elif conv_rho < 1:
    verdict = "RETRACT_stage2_conclusion"
else:
    verdict = "PARTIAL_conv_between_1_and_2"

out = {"header": hdr, "rows": rows, "verdict": verdict,
       "kill1_b1_over_0p1": kill1,
       "predicted": {"conv_rho": ">=2", "matmul_rho": "<0.1", "b1_rho": "<0.01"},
       "determinism_all": all(r["det_repeat_equal"] for r in rows)}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"verdict": verdict, "conv": conv_rho, "conv_ieee": rows[0]["rho_ieee"],
                  "matmul": mm_rho, "b1": b1_rho, "det": out["determinism_all"],
                  "tf32_cudnn": hdr["allow_tf32_cudnn_at_start"],
                  "tf32_mm": hdr["allow_tf32_matmul_at_start"],
                  "f32_prec": hdr["float32_matmul_precision"]}, indent=1))
