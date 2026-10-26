"""A6c — DIAGNOSTIC (post-hoc, labeled): is conv rho's C-dependence a precision-policy effect
(allow_tf32) or an algorithm-selection effect?

Not preregistered. Motivated by two results already in hand:
  a6 (prereg dbfd600): conv C=32 H=5 -> rho 0.7417, IDENTICAL with allow_tf32=False (16 digits).
  a6b (extension): conv rho non-monotone in C: C=16 54.49, C=64 1.415, C=256 10.195.
If rho is unchanged when TF32 is disabled across all C, the C-dependence is algorithm selection
(cudnn heuristic / transform-class arithmetic), not a precision default — which is a sharper statement
than "the reference is TF32-class" and is the one the atlas can carry.

Reported: rho and frac_over for each C under both flag states, plus whether the two outputs are bit-equal.
"""
import json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "a6c_conv_tf32pair.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def rho(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    return {"rho": float(r.max()), "med": float(r.median()), "abs_max": float(d.max()),
            "frac_over": float((r > 1).double().mean())}


rows = []
prev = torch.backends.cudnn.allow_tf32
for C in (16, 32, 64, 128, 256):
    torch.manual_seed(1234 + 5 * 100 + 2 * 10 + C)
    x = torch.randn(2, C, 5, 5, device=dev)
    w = torch.randn(C, C, 3, 3, device=dev)
    torch.backends.cudnn.allow_tf32 = True
    o_tf = F.conv2d(x, w, padding=1)
    torch.backends.cudnn.allow_tf32 = False
    o_ie = F.conv2d(x, w, padding=1)
    torch.backends.cudnn.allow_tf32 = prev
    o_64 = F.conv2d(x.double(), w.double(), padding=1)
    rt, ri = rho(o_tf, o_64), rho(o_ie, o_64)
    rows.append({"C": C, "K_eff": C * 9, "tf32_default": rt, "tf32_off": ri,
                 "bit_equal": bool(torch.equal(o_tf, o_ie)),
                 "rho_ratio_off_over_def": ri["rho"] / rt["rho"] if rt["rho"] else None})
    del x, w, o_tf, o_ie, o_64
    torch.cuda.empty_cache()

out = {"header": {"date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                  "note": "post-hoc diagnostic, not preregistered; labels in docstring",
                  "shape": {"N": 2, "H": 5, "W": 5, "k": 3, "s": 1, "p": 1}},
       "rows": rows,
       "any_bit_equal": any(r["bit_equal"] for r in rows),
       "all_bit_equal": all(r["bit_equal"] for r in rows)}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps([{ "C": r["C"], "rho_def": round(r["tf32_default"]["rho"], 3),
                     "rho_off": round(r["tf32_off"]["rho"], 3), "bit_equal": r["bit_equal"],
                     "frac_over_def": round(r["tf32_default"]["frac_over"], 4)} for r in rows], indent=1))
