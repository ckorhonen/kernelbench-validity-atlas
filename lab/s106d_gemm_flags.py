#!/usr/bin/env python
"""s106d_gemm_flags.py — does the map's gemm column measure a different arithmetic than its label claims?
Prereg: lab/s106d_prereg.md, committed before any row."""
import importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s106d_gemm_flags.jsonl")
SUM = os.path.join(LAB, "results", "s106d_summary.json")
S105C = "/Users/ckorhonen/rstation/kernelbench/lab/results/s105c_map_join.json"
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
S105C_REMOTE = os.path.join(LAB, "results", "s105c_map_join.json")
ALARM = 300
CHUNK = 1 << 22
STRICT = 1e-4
CLASS_PATTERNS = [("WINOGRAD", re.compile(r"winograd", re.I)),
                  ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
                  ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
                  ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|fp32|scudnn|simt", re.I)),
                  ("CUTLASS", re.compile(r"cutlass", re.I))]


class Timeout(Exception):
    pass


def _alarm(s, f):
    raise Timeout("task exceeded %ds" % ALARM)


signal.signal(signal.SIGALRM, _alarm)


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab, n
    return ("OTHER", names[0] if names else None)


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profile_fn(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evts = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages()
            if e.self_device_time_total > 0]
    evts.sort(key=lambda t: -t[1])
    return evts[:4]


def run_ref(model, inputs, dtype, cfg):
    torch.backends.cudnn.allow_tf32 = cfg[0]
    torch.backends.cuda.matmul.allow_tf32 = cfg[1]
    model = model.to(dtype=dtype)
    ins = [x.to(dtype=dtype) if torch.is_tensor(x) else x for x in inputs]
    with torch.no_grad():
        out = model(*ins)
    if not torch.is_tensor(out):
        return None
    return out


def rho_vs(ref, anchor):
    n_fail, mx, tot = 0, 0.0, 0
    rf, an = ref.detach().reshape(-1), anchor.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK)
        d = (rf[i:j].double() - an[i:j]).abs()
        bb = STRICT + STRICT * an[i:j].abs()
        r = d / bb
        n_fail += int((r > 1).sum().item())
        mx = max(mx, float(torch.max(r, 0)[0].item()))
        tot += (j - i)
        i = j
    return mx, n_fail / tot


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    join = json.load(open(S105C_REMOTE))
    s71 = {(r["level"], r["problem_id"]): r for r in
           [json.loads(l) for l in open(S71) if l.strip()]}
    cand = []
    for r in join["rows"]:
        if r.get("status") != "ok":
            continue
        src = s71.get((r["level"], r["problem_id"]))
        if not src or src.get("anchor") != "admissible":
            continue
        if (r.get("op_class"), r.get("engine_class")) in (("gemm", "TF32_TENSOROP"),
                                                         ("conv", "TF32_TENSOROP")):
            cand.append((src.get("t_fp64_s") or 99, r, src))
    cand.sort(key=lambda t: t[0])
    gemm = [c for c in cand if c[1]["op_class"] == "gemm"][:10]
    conv = [c for c in cand if c[1]["op_class"] == "conv"][:3]
    picked = gemm + conv
    print("picked", [(c[1]["op_class"], c[1]["problem_id"], c[0]) for c in picked], flush=True)

    rows = []
    with open(OUT, "a") as f:
        for t_fp64, r, src in picked:
            rec = dict(level=r["level"], problem_id=r["problem_id"], name=r["name"],
                       op_class=r["op_class"], engine_class_s105c=r["engine_class"],
                       s71_rho_strict=src["A_vs_anchor"]["ew_1e-4"]["rho_global"],
                       s71_t_fp64_s=src.get("t_fp64_s"), status="ok")
            tw = time.time()
            try:
                signal.alarm(ALARM)
                path = os.path.join(KB, "level%d" % r["level"], r["name"])
                mod = load_task(path)
                torch.manual_seed(0)
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                inputs = mod.get_inputs()
                model = mod.Model(*init).cuda().eval()
                ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                # (a) s71 default
                ev_a = profile_fn(lambda: run_ref(model, ins, torch.float32, (True, False)))
                out_a = run_ref(model, ins, torch.float32, (True, False))
                # (b) matmul tf32 on
                ev_b = profile_fn(lambda: run_ref(model, ins, torch.float32, (True, True)))
                out_b = run_ref(model, ins, torch.float32, (True, True))
                # (c) fp64 anchor
                out_x = run_ref(model, ins, torch.float64, (False, False))
                signal.alarm(0)
                if out_x is None or out_a is None or out_b is None:
                    raise RuntimeError("missing output")
                if out_x.shape != out_a.shape:
                    raise RuntimeError("anchor shape mismatch")
                rec["label_a"], rec["top1_a"] = classify([e[0] for e in ev_a])
                rec["label_b"], rec["top1_b"] = classify([e[0] for e in ev_b])
                rec["rho_a"], rec["frac_a"] = rho_vs(out_a, out_x)
                rec["rho_b"], rec["frac_b"] = rho_vs(out_b, out_x)
                rec["ref_max_abs"] = float(out_x.abs().max().item())
                rec["a_b_bit_identical"] = bool(torch.equal(out_a, out_b))
                if rec["s71_rho_strict"] and rec["s71_rho_strict"] > 0:
                    rec["tie_out_rel"] = abs(rec["rho_a"] - rec["s71_rho_strict"]) / rec["s71_rho_strict"]
                rec["ratio_b_over_a"] = (rec["rho_b"] / rec["rho_a"]) if rec["rho_a"] > 0 else None
                rec["wall_s"] = round(time.time() - tw, 2)
                del model, ins, inputs, out_a, out_b, out_x
                torch.cuda.empty_cache()
            except Timeout:
                signal.alarm(0); rec["status"] = "timeout"
            except torch.cuda.OutOfMemoryError:
                signal.alarm(0); torch.cuda.empty_cache(); rec["status"] = "oom"
            except Exception as e:
                signal.alarm(0); rec["status"] = "error"; rec["err"] = repr(e)[:200]
                torch.cuda.empty_cache()
            f.write(json.dumps(rec) + "\n"); f.flush()
            rows.append(rec)
            print(rec["op_class"], rec["problem_id"], rec["status"], rec.get("label_a"), rec.get("label_b"),
                  rec.get("rho_a"), rec.get("rho_b"), rec.get("ratio_b_over_a"), rec.get("wall_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    gm = [r for r in ok if r["op_class"] == "gemm"]
    cv = [r for r in ok if r["op_class"] == "conv"]
    big = [r for r in gm if (r.get("ratio_b_over_a") or 0) >= 10]
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None
    med_b = med([r["rho_b"] for r in gm])
    if len(gm) < 8:
        branch = "INCOMPLETE"
    elif len(big) >= 6 and med_b is not None and med_b <= 3 * 1.947:
        branch = "BRANCH_FLAGS_ARTIFACT"
    elif len(big) < 6:
        branch = "BRANCH_CLASS_RESIDUAL"
    else:
        branch = "BRANCH_FLAGS_PARTIAL"
    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   n_rows=len(rows), n_ok=len(ok), n_gemm=len(gm), n_conv=len(cv),
                   n_gemm_ratio_ge10=len(big), med_rho_b_gemm=med_b,
                   med_rho_a_gemm=med([r["rho_a"] for r in gm]),
                   med_rho_a_conv=med([r["rho_a"] for r in cv]),
                   med_rho_b_conv=med([r["rho_b"] for r in cv]),
                   tie_out=[r.get("tie_out_rel") for r in ok if "tie_out_rel" in r],
                   labels=[[r["problem_id"], r["op_class"], r.get("label_a"), r.get("label_b")] for r in ok],
                   branch=branch, rows=rows)
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("branch", "n_ok", "n_gemm", "n_gemm_ratio_ge10",
                                              "med_rho_a_gemm", "med_rho_b_gemm", "med_rho_a_conv",
                                              "med_rho_b_conv", "tie_out")}, indent=1))


if __name__ == "__main__":
    main()
