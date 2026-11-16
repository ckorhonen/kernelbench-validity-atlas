#!/usr/bin/env python
"""s112_conv_profile.py — conv-only profile on the three mixed OTHER-labelled rows (L2/48, L2/93, L2/100).
Prereg: lab/s112_prereg.md, committed before any row is computed. Frozen rule is coded verbatim there.
"""
import importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s112_conv_profile.jsonl")
SUM = os.path.join(RES, "s112_summary.json")
ALARM = 180
NDRAW = 16
STRICT = 1e-4
CHUNK = 1 << 22
ROWS = [(2, 48), (2, 93), (2, 100)]
CONV = re.compile(r"winograd|implicit_conv|implicit_gemm|dgrad|fprop|wgrad|cudnn|cutlass|conv2d|conv_transpose|_conv_", re.I)
CLASS_PATTERNS = [("WINOGRAD", re.compile(r"winograd", re.I)),
                  ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
                  ("FP64", re.compile(r"dgrad|dgemm|dconv|fp64|double|_d_", re.I)),
                  ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|fp32|scudnn|simt", re.I)),
                  ("CUTLASS", re.compile(r"cutlass", re.I))]


class Timeout(Exception):
    pass


def _alarm(s, f):
    raise Timeout("cell exceeded %ds" % ALARM)


signal.signal(signal.SIGALRM, _alarm)


def classify(names):
    for lab, pat in CLASS_PATTERNS:
        for n in names:
            if pat.search(n):
                return lab
    return "OTHER"


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profile_events(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [(e.key, float(e.self_device_time_total)) for e in prof.key_averages()
            if e.self_device_time_total > 0]


def stats_vs(ref, anchor, gate):
    n_fail, mx = 0, 0.0
    rf, an = ref.detach().reshape(-1), anchor.detach().reshape(-1)
    n, i = rf.numel(), 0
    while i < n:
        j = min(n, i + CHUNK)
        d = (rf[i:j].double() - an[i:j]).abs()
        bb = gate + gate * an[i:j].abs()
        r = d / bb
        n_fail += int((r > 1).sum().item())
        mx = max(mx, float(torch.max(r, 0)[0].item()))
        i = j
    return dict(rho_ew=mx, frac=n_fail / n, total=n)


def run(model, ins, dtype, cfg):
    torch.backends.cudnn.allow_tf32 = cfg[0]
    torch.backends.cuda.matmul.allow_tf32 = cfg[1]
    m = model.to(dtype=dtype)
    xs = [x.to(dtype=dtype) if torch.is_tensor(x) else x for x in ins]
    with torch.no_grad():
        return m(*xs)


def find_task(level, pid):
    base = os.path.join(KB, "level%d" % level)
    for f in sorted(os.listdir(base)):
        if f.startswith("%d_" % pid) and f.endswith(".py"):
            return os.path.join(base, f)
    raise FileNotFoundError("level%d/%d" % (level, pid))


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    t0 = time.time()
    rows = []
    with open(OUT, "a") as f:
        for level, pid in ROWS:
            path = find_task(level, pid)
            name = os.path.basename(path)
            base = dict(level=level, problem_id=pid, name=name)
            try:
                signal.alarm(ALARM)
                mod = load_task(path)
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                rec = dict(base, draw=-1, status="load_error", err=repr(e)[:200])
                f.write(json.dumps(rec) + "\n"); f.flush(); rows.append(rec)
                continue
            k1_done = False
            for d in range(NDRAW):
                rec = dict(base, draw=d, seed=d, status="ok")
                tw = time.time()
                try:
                    signal.alarm(ALARM)
                    torch.manual_seed(d)
                    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                    inputs = mod.get_inputs()
                    model = mod.Model(*init).cuda().eval()
                    ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                    ev = profile_events(lambda: run(model, ins, torch.float32, (True, True)))
                    out_b = run(model, ins, torch.float32, (True, True))
                    out_x = run(model, ins, torch.float64, (False, False))
                    if d == 0 and not k1_done:
                        out_b2 = run(model, ins, torch.float32, (True, True))
                        rec["k1_bit_identical"] = bool(torch.equal(out_b, out_b2))
                        del out_b2
                        out_x2 = run(model, ins, torch.float64, (False, False))
                        denom = float(out_x.abs().max().item()) or 1.0
                        rec["k2_anchor_rel"] = float((out_x - out_x2).abs().max().item()) / denom
                        del out_x2
                        k1_done = True
                    signal.alarm(0)
                    total = sum(t for _, t in ev) or 1.0
                    conv = [(n, t) for n, t in ev if CONV.search(n)]
                    conv.sort(key=lambda x: -x[1])
                    top_whole = ev[0][0] if ev else None
                    rec["top1_whole"] = top_whole
                    rec["label_whole"] = classify([n for n, _ in ev[:4]])
                    rec["conv_engine_present"] = bool(conv)
                    rec["conv_names"] = sorted({n for n, _ in conv})
                    if conv:
                        rec["conv_top"] = conv[0][0]
                        rec["conv_time_s"] = conv[0][1]
                        rec["conv_share"] = conv[0][1] / total
                    rec["n_events"] = len(ev)
                    rec["strict"] = stats_vs(out_b, out_x, STRICT)
                    rec["wall_s"] = round(time.time() - tw, 2)
                    del model, ins, inputs, out_b, out_x
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
                print(level, pid, d, rec["status"], rec.get("label_whole"), rec.get("conv_engine_present"),
                      (rec.get("conv_top") or "")[:60], round(rec.get("conv_share") or 0, 3),
                      (rec.get("strict") or {}).get("rho_ew"), rec.get("wall_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    per_row = {}
    for level, pid in ROWS:
        cells = [r for r in ok if (r["level"], r["problem_id"]) == (level, pid)]
        names = sorted({r.get("conv_top") for r in cells if r.get("conv_top")})
        per_row["%d/%d" % (level, pid)] = dict(
            n_ok=len(cells),
            conv_present=sum(1 for r in cells if r.get("conv_engine_present")),
            conv_names=names,
            n_distinct=len(names),
            labels_whole=sorted({r.get("label_whole") for r in cells}),
            top1_whole=sorted({r.get("top1_whole") for r in cells}),
            rho=[r["strict"]["rho_ew"] for r in cells],
            share=[round(r.get("conv_share") or 0, 4) for r in cells])
    k1 = [r.get("k1_bit_identical") for r in rows if "k1_bit_identical" in r]
    k2 = next((r.get("k2_anchor_rel") for r in rows if "k2_anchor_rel" in r), None)
    instr_ok = all(k1) and (k2 is None or k2 <= 1e-9) and len(ok) >= 40
    all_present = all(v["conv_present"] == v["n_ok"] and v["n_ok"] > 0 for v in per_row.values())
    per_draw = any(v["n_distinct"] >= 2 for v in per_row.values())
    absent = any(v["conv_present"] == 0 for v in per_row.values())
    if not instr_ok:
        branch = "INSTRUMENT_FAILED"
    elif absent:
        branch = "BRANCH_NO_CONV_ENGINE"
    elif all_present and per_draw:
        branch = "BRANCH_ENGINE_PER_DRAW"
    else:
        branch = "BRANCH_ENGINE_STABLE"
    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=round(time.time() - t0, 1),
                   n_cells=len(rows), n_ok=len(ok), k1=k1, k2_anchor_rel=k2, instrument_ok=instr_ok,
                   branch=branch, per_row=per_row)
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("branch", "instrument_ok", "n_ok", "wall_s")}, indent=1))
    for k, v in per_row.items():
        print(k, "present", v["conv_present"], "/", v["n_ok"], "distinct", v["conv_names"],
              "labels", v["labels_whole"])


if __name__ == "__main__":
    main()
