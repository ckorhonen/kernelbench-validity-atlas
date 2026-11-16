#!/usr/bin/env python
"""s114_engine_relabel.py — engine-restricted relabel of the 15 blind compute-class map rows.
Prereg: lab/s114_prereg.md, committed before any row is computed. Frozen rule is coded verbatim there.
"""
import importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s114_engine_relabel.jsonl")
SUM = os.path.join(RES, "s114_summary.json")
ALARM = 240
WALL_CAP = 900
CHUNK = 1 << 22
STRICT = 1e-4
ROWS = [(2, 48), (2, 79), (2, 23), (2, 100), (2, 74), (2, 83), (1, 12), (2, 6), (1, 88),
        (2, 20), (2, 92), (1, 91), (1, 93), (2, 93), (2, 26)]
LAYOUT = re.compile(r"nhwc|nchw|contiguous|catArrayBatchedCopy|cat|slice|transpose|permute|memcpy|Memcpy|Memset|copy|fill|zero|pad|reshape|view|clone|resize", re.I)
POINTWISE = re.compile(r"elementwise|gpu_kernel|vectorized|unary|binary|tanh|sigmoid|gelu|relu|elu|silu|clamp|exp|log|pow|sqrt|rsqrt|abs|neg|mul|add|div|sub|masked|where|dropout|index|gather|scatter|topk|sort|argmax|argmin", re.I)
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


def classify(name):
    if not name:
        return "NO_COMPUTE"
    for lab, pat in CLASS_PATTERNS:
        if pat.search(name):
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
    ev = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages()
          if e.self_device_time_total > 0]
    ev.sort(key=lambda t: -t[1])
    return ev


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


def extract_compute(ev):
    total = sum(t for _, t in ev) or 1.0
    cand = [(n, t) for n, t in ev if not LAYOUT.search(n) and not POINTWISE.search(n)]
    if not cand:
        return None, 0.0, 0
    cand.sort(key=lambda x: -x[1])
    return cand[0][0], cand[0][1] / total, len(cand)


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    t0 = time.time()
    rows = []
    k1_targets = {ROWS[0], ROWS[1], ROWS[2]}
    with open(OUT, "a") as f:
        for level, pid in ROWS:
            path = find_task(level, pid)
            base = dict(level=level, problem_id=pid, name=os.path.basename(path))
            rec = dict(base, status="ok")
            if time.time() - t0 > WALL_CAP:
                rec["status"] = "wall_capped"
                f.write(json.dumps(rec) + "\n"); f.flush(); rows.append(rec); continue
            tw = time.time()
            try:
                signal.alarm(ALARM)
                mod = load_task(path)
                torch.manual_seed(0)
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                inputs = mod.get_inputs()
                model = mod.Model(*init).cuda().eval()
                ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                ev = profile_events(lambda: run(model, ins, torch.float32, (True, True)))
                out_b = run(model, ins, torch.float32, (True, True))
                out_x = run(model, ins, torch.float32, (False, False))
                if (level, pid) in k1_targets:
                    out_b2 = run(model, ins, torch.float32, (True, True))
                    rec["k1_bit_identical"] = bool(torch.equal(out_b, out_b2))
                    del out_b2
                if (level, pid) == ROWS[1]:
                    out_x2 = run(model, ins, torch.float32, (False, False))
                    den = float(out_x.abs().max().item()) or 1.0
                    rec["k2_anchor_rel"] = float((out_x - out_x2).abs().max().item()) / den
                    del out_x2
                signal.alarm(0)
                cname, share, ncand = extract_compute(ev)
                rec["top1_whole"] = ev[0][0] if ev else None
                rec["label_whole"] = classify(ev[0][0] if ev else None)
                rec["compute_name"] = cname
                rec["compute_share"] = round(share, 5)
                rec["n_candidates"] = ncand
                rec["compute_class"] = classify(cname)
                rec["compute_is_top1"] = bool(cname and ev and cname == ev[0][0])
                rec["n_events"] = len(ev)
                rec["events"] = [[n, round(t, 4)] for n, t in ev]
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
            print(level, pid, rec["status"], rec.get("compute_class"), (rec.get("compute_name") or "")[:70],
                  rec.get("compute_share"), (rec.get("strict") or {}).get("rho_ew"), rec.get("wall_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    k1 = [r.get("k1_bit_identical") for r in rows if "k1_bit_identical" in r]
    k2 = next((r.get("k2_anchor_rel") for r in rows if "k2_anchor_rel" in r), None)
    instr_ok = len(k1) == 3 and all(k1) and k2 is not None and k2 <= 1e-9
    found = [r for r in ok if r.get("compute_name") and (r.get("compute_share") or 0) >= 0.01]
    noop = sum(1 for r in ok if r.get("compute_is_top1"))
    if not instr_ok:
        branch = "INSTRUMENT_FAILED"
    elif noop >= 12:
        branch = "BRANCH_RELABEL_NOOP"
    elif len(found) >= 12:
        branch = "BRANCH_RELABEL_COMPLETE"
    elif len(found) >= 8:
        branch = "BRANCH_RELABEL_PARTIAL"
    else:
        branch = "BRANCH_RELABEL_FAILS"
    # tie-out against recorded map rows (informational)
    tie = {}
    for fn in ("s107_rekey_map.jsonl", "s107b_rekey_rest.jsonl"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            bs = r.get("B_strict") or {}
            if "rho_ew" in bs:
                tie[(r.get("level"), r.get("problem_id"))] = bs["rho_ew"]
    for r in ok:
        rec = tie.get((r["level"], r["problem_id"]))
        got = r["strict"]["rho_ew"]
        if rec:
            r["tie_out_rel"] = abs(got - rec) / (abs(rec) or 1.0)
    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=round(time.time() - t0, 1),
                   n_cells=len(rows), n_ok=len(ok), k1=k1, k2_anchor_rel=k2, instrument_ok=instr_ok,
                   branch=branch, n_found_share1pct=len(found), n_compute_is_top1=noop,
                   relabel=[dict(task="%d/%d" % (r["level"], r["problem_id"]), name=r["name"],
                                 label_before=r.get("label_whole"), compute_name=r.get("compute_name"),
                                 compute_class=r.get("compute_class"), compute_share=r.get("compute_share"),
                                 rho_ew=r["strict"]["rho_ew"], frac=r["strict"]["frac"],
                                 tie_out_rel=r.get("tie_out_rel"))
                            for r in ok])
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("branch", "instrument_ok", "n_ok", "n_found_share1pct",
                                              "n_compute_is_top1", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
