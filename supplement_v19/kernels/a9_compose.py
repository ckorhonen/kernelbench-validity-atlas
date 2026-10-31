"""A9 — two-stage reference budget composition probe.

Prereg: lab/a9_compose_prereg.md, commit 53ca881, frozen BEFORE this run.
Rules R1/R2/R3 and the ADEQUATE bar fire as written; no regrade.

Currency: rho = max_e |R_def - R_64| / (atol + rtol*|R_64|), the harness's own budget. Everything is
reported in budget multiples so no |ref| column is needed (the reason V2 was cut was a missing column,
not a missing mechanism).
"""
import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
def _kb_res_dir(here):
    """Resolve the artifact directory: lab/results in the repo, ../results when run from the
    supplement package's kernels/ directory (frozen packaging convention, round-11 F6)."""
    for cand in (os.path.join(here, "results"), os.path.join(os.path.dirname(here), "results")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(here, "results")


RES = _kb_res_dir(HERE)
OUT = os.path.join(RES, "a9_compose.json")
ATOL = RTOL = 1e-4
dev = "cuda"


def stats(a, b):
    d = (a.double() - b).abs()
    den = ATOL + RTOL * b.abs()
    r = (d / den).flatten()
    return {"rho_max": float(r.max()), "med": float(r.median()),
            "frac_over": float((r > 1).double().mean()), "n_elem": int(r.numel()),
            "abs_max": float(d.max())}


def budget(a64):
    return ATOL + RTOL * a64.abs()


def conv_fn(x, w, k, s, d):
    p = (k - 1) // 2 * d
    return lambda t, wt: F.conv2d(t, wt, padding=p, stride=s, dilation=d)


def mm_fn(K):
    return lambda t, wt: t @ wt


def run_chain(idx, spec):
    """spec: dict with familyA/paramsA, familyB/paramsB. Returns per-chain measurements."""
    torch.manual_seed(7500 + idx)
    fam_a, fam_b = spec["A"]["family"], spec["B"]["family"]

    if fam_a == "conv2d":
        pa = spec["A"]
        xa = torch.randn(2, pa["C"], pa["H"], pa["H"], device=dev)
        wa = torch.randn(pa["C"], pa["C"], pa["k"], pa["k"], device=dev)
        fa = conv_fn(xa, wa, pa["k"], pa["s"], pa["d"])
        a32 = fa(xa, wa)
        a64 = fa(xa.double(), wa.double())
    else:
        pa = spec["A"]
        xa = torch.randn(pa["M"], pa["K"], device=dev)
        wa = torch.randn(pa["K"], pa["M"], device=dev)
        fa = mm_fn(pa["K"])
        a32 = fa(xa, wa)
        a64 = fa(xa.double(), wa.double())

    rho_a, frac_a = stats(a32, a64)["rho_max"], stats(a32, a64)["frac_over"]

    # B parameters are derived from A's actual output shape so the chain is real.
    pb = spec["B"]
    if fam_b == "conv2d":
        Cb, Hb = a32.shape[1], a32.shape[2]
        wb = torch.randn(Cb, Cb, pb["k"], pb["k"], device=dev)
        fb = conv_fn(a32, wb, pb["k"], pb["s"], pb["d"])
        b_alone_32 = fb(a64.float(), wb)
        b_alone_64 = fb(a64, wb.double())
    else:
        Kb = a32.shape[1]
        wb = torch.randn(Kb, a32.shape[1], device=dev)
        fb = mm_fn(Kb)
        b_alone_32 = fb(a64.float(), wb)
        b_alone_64 = fb(a64, wb.double())
    s_alone = stats(b_alone_32, b_alone_64)

    # end-to-end fp32 vs end-to-end fp64 on the same weights
    end32 = fb(a32, wb)
    end64 = fb(a64, wb.double())
    s_end = stats(end32, end64)
    det = bool(torch.equal(end32, fb(a32, wb)))

    # G: one budget multiple injected at the argmax element of A_def (in A's budget units)
    den_a = budget(a64)
    ratio_a = (a32.double() - a64).abs() / den_a
    flat_idx = int(ratio_a.flatten().argmax())
    delta = torch.zeros_like(a32)
    delta.flatten()[flat_idx] = den_a.flatten()[flat_idx].float()
    g1_out = fb(a32 + delta, wb)
    g10_out = fb(a32 + 10 * delta, wb)
    base_out = fb(a32, wb)

    def gain(out):
        r = (out.double() - base_out.double()).abs() / budget(base_out.double())
        return float(r.flatten().max())

    G1, G10 = gain(g1_out), gain(g10_out)
    G10_div = G10 / 10.0
    g_nonlinear = abs(torch.log10(torch.tensor(max(G10_div, 1e-12) / max(G1, 1e-12))).item()) > 0.5

    return {
        "chain": spec["name"], "family_A": fam_a, "family_B": fam_b,
        "A": spec["A"], "B_params": pb,
        "rho_A": rho_a, "frac_A": frac_a,
        "rho_B_alone": s_alone["rho_max"], "frac_B_alone": s_alone["frac_over"],
        "rho_end": s_end["rho_max"], "frac_end": s_end["frac_over"],
        "G1": G1, "G10": G10, "G10_div10": G10_div, "G_nonlinear": bool(g_nonlinear),
        "deterministic_2run": det,
        "runtime_facts": {
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        },
    }


CHAINS = [
    {"name": "c1_W_then_TF32", "A": {"family": "conv2d", "C": 256, "H": 7, "k": 3, "s": 1, "d": 1},
     "B": {"family": "conv2d", "k": 3, "s": 2, "d": 1}},
    {"name": "c2_W_then_W", "A": {"family": "conv2d", "C": 64, "H": 13, "k": 3, "s": 1, "d": 1},
     "B": {"family": "conv2d", "k": 3, "s": 1, "d": 1}},
    {"name": "c3_TF32_then_TF32", "A": {"family": "conv2d", "C": 16, "H": 21, "k": 3, "s": 1, "d": 1},
     "B": {"family": "conv2d", "k": 3, "s": 1, "d": 1}},
    {"name": "c4_cleanA_GEMM_then_TF32", "A": {"family": "conv2d", "C": 16, "H": 7, "k": 1, "s": 1, "d": 1},
     "B": {"family": "conv2d", "k": 3, "s": 1, "d": 1}},
    {"name": "c5_cleanA_mm_then_mm", "A": {"family": "matmul", "M": 256, "K": 128},
     "B": {"family": "matmul"}},
    {"name": "c6_mm_deep_then_mm", "A": {"family": "matmul", "M": 1024, "K": 8192},
     "B": {"family": "matmul"}},
]


def main():
    rows = []
    for i, spec in enumerate(CHAINS):
        try:
            r = run_chain(i, spec)
            rows.append(r)
            print(f"[{r['chain']}] rho_A={r['rho_A']:.3f} rho_B={r['rho_B_alone']:.3f} "
                  f"rho_end={r['rho_end']:.3f} G1={r['G1']:.3f} nonlin={r['G_nonlinear']}")
        except Exception as e:
            print(f"[{spec['name']}] FAILED: {e}", file=sys.stderr)
            raise
        finally:
            torch.cuda.empty_cache()

    out = {"header": {"prereg": "lab/a9_compose_prereg.md @ 53ca881", "atol": ATOL, "rtol": RTOL,
                      "n_chains": len(rows)}, "rows": rows}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
