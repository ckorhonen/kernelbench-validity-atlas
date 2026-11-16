#!/usr/bin/env python3
"""s116_bellatrix_control.py — same-arch control on the 10 published task ids.
Prereg: lab/s116_prereg.md (frozen pre-run). Self-contained; runs on the lab host."""
import glob, importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s116_bellatrix.jsonl")
SUM = os.path.join(LAB, "results", "s116_summary.json")
REC = os.path.join(LAB, "s116_recorded.json")
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


def stats_vs(ref, anchor, gate):
    n_fail, mx, gmx = 0, 0.0, 0.0
    rf, an = ref.detach().reshape(-1), anchor.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK)
        d = (rf[i:j].double() - an[i:j]).abs()
        bb = gate + gate * an[i:j].abs()
        r = d / bb
        n_fail += int((r > 1).sum().item())
        mx = max(mx, float(torch.max(r, 0)[0].item()))
        gmx = max(gmx, float(torch.max(d, 0)[0].item()))
        i = j
    ref_max = float(anchor.abs().max().item())
    return dict(rho_ew=mx, frac=n_fail / n, maxabs=gmx,
                rho_global=gmx / (gate + gate * ref_max), total=n)


def resolve(level, pid):
    hits = sorted(glob.glob(os.path.join(KB, "level%d" % level, "%d_*.py" % pid)))
    return hits[0] if hits else None


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    rec = json.load(open(REC))
    rows = []
    n_repeat = 0
    anchor_done = False
    with open(OUT, "a") as f:
        for key, rv in rec.items():
            level, pid = rv["level"], rv["problem_id"]
            row = dict(level=level, problem_id=pid, name=rv["name"], op_class=rv.get("op_class"),
                       rho_recorded=rv["rho_recorded"], host="bellatrix", status="ok")
            path = resolve(level, pid)
            if not path:
                row["status"] = "not_found"
                f.write(json.dumps(row) + "\n"); f.flush(); rows.append(row); continue
            tw = time.time()
            try:
                signal.alarm(ALARM)
                mod = load_task(path)
                torch.manual_seed(0)
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                inputs = mod.get_inputs()
                model = mod.Model(*init).cuda().eval()
                ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                ev_b = profile_fn(lambda: run_ref(model, ins, torch.float32, (True, True)))
                out_b = run_ref(model, ins, torch.float32, (True, True))
                out_x = run_ref(model, ins, torch.float64, (False, False))
                if out_x is None or out_b is None or out_x.shape != out_b.shape:
                    raise RuntimeError("missing output")
                if n_repeat < 3:
                    out_b2 = run_ref(model, ins, torch.float32, (True, True))
                    row["k1_repeat_bit_identical"] = bool(torch.equal(out_b, out_b2))
                    n_repeat += 1
                    del out_b2
                if not anchor_done:
                    out_x2 = run_ref(model, ins, torch.float64, (False, False))
                    denom = float(out_x.abs().max().item()) or 1.0
                    row["k2_anchor_rel"] = float((out_x - out_x2).abs().max().item()) / denom
                    anchor_done = True
                    del out_x2
                signal.alarm(0)
                row["label_B"], row["top1_B"] = classify([e[0] for e in ev_b])
                row["B_strict"] = stats_vs(out_b, out_x, STRICT)
                row["ref_max_abs"] = float(out_x.abs().max().item())
                rho = row["B_strict"]["rho_ew"]
                rr = row["rho_recorded"]
                row["reproduced"] = (abs(rho - rr) <= 0.02) if rr < 0.1 else (abs(rho - rr) / rr <= 0.30)
                row["rel_diff"] = (abs(rho - rr) / rr) if rr else None
                row["wall_s"] = round(time.time() - tw, 2)
                del model, ins, inputs, out_b, out_x
                torch.cuda.empty_cache()
            except Timeout:
                signal.alarm(0); row["status"] = "timeout"
            except torch.cuda.OutOfMemoryError:
                signal.alarm(0); torch.cuda.empty_cache(); row["status"] = "oom"
            except Exception as e:
                signal.alarm(0); row["status"] = "error"; row["err"] = repr(e)[:200]
                torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n"); f.flush()
            rows.append(row)
            print(level, pid, row["status"], row.get("label_B"),
                  (row.get("B_strict") or {}).get("rho_ew"), row["rho_recorded"],
                  row.get("reproduced"), row.get("wall_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    n_rep = sum(1 for r in ok if r.get("reproduced"))
    k1 = [r.get("k1_repeat_bit_identical") for r in ok if "k1_repeat_bit_identical" in r]
    k2 = [r.get("k2_anchor_rel") for r in ok if "k2_anchor_rel" in r]
    k_ok = all(k1) and all(v is not None and v <= 1e-6 for v in k2) and len(k1) == 3
    if not k_ok:
        branch = "INVALID"
    elif n_rep >= 8:
        branch = "BRANCH_SAME_ARCH_REPRODUCED"
    elif n_rep >= 5:
        branch = "BRANCH_SAME_ARCH_PARTIAL"
    else:
        branch = "BRANCH_ARCH_KEYED"
    res = {"date": "2026-11-16", "host": "bellatrix", "n_rows": len(rows), "n_ok": len(ok),
           "n_reproduced": n_rep, "K1": k1, "K2": k2, "K_ok": k_ok, "branch": branch,
           "non_claims": ["one draw per task (seed 0)", "procedure B only", "no timing claim",
                          "same arch = same GPU model + same torch/CUDA stack, not identical driver state"],
           "rows": [{k: v for k, v in r.items() if k in
                     ("level", "problem_id", "name", "op_class", "rho_recorded", "status",
                      "label_B", "reproduced", "rel_diff")} for r in rows],
           "rho_new": {("%d/%d" % (r["level"], r["problem_id"])): (r.get("B_strict") or {}).get("rho_ew")
                       for r in rows}}
    json.dump(res, open(SUM, "w"), indent=1)
    print(json.dumps(res["branch"]), res["n_reproduced"], "/", len(ok))


if __name__ == "__main__":
    main()
