#!/usr/bin/env python
"""s107b_rekey_rest.py — re-key the remaining 107 map rows under procedure B (declared extension of s107).
Prereg: lab/s107b_prereg.md, committed before any row is computed."""
import importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s107b_rekey_rest.jsonl")
SUM = os.path.join(LAB, "results", "s107b_summary.json")
JOIN = os.path.join(LAB, "results", "s105c_map_join.json")
S71 = os.path.join(LAB, "results", "s71_full_audit.jsonl")
ALARM = 240
WALL_CAP = 1200
CHUNK = 1 << 22
STRICT = 1e-4
BENCH = 1e-2
FLAGGED = (("gemm", "TF32_TENSOROP"), ("conv_transpose", "TF32_TENSOROP"), ("conv", "OTHER"))
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
    return dict(rho_ew=mx, frac=n_fail / n, maxabs=gmx, total=n)


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    join = json.load(open(JOIN))
    s71 = {(r["level"], r["problem_id"]): r for r in
           [json.loads(l) for l in open(S71) if l.strip()]}
    picked = []
    for r in join["rows"]:
        if r.get("status") != "ok":
            continue
        if (r.get("op_class"), r.get("engine_class")) in FLAGGED:
            continue
        src = s71.get((r["level"], r["problem_id"]))
        if not src or src.get("anchor") != "admissible":
            continue
        picked.append((src.get("t_fp64_s") or 99, r, src))
    picked.sort(key=lambda t: t[0])
    print("picked", len(picked), flush=True)

    t0 = time.time()
    rows = []
    n_repeat = 0
    anchor_done = False
    with open(OUT, "a") as f:
        for t_fp64, r, src in picked:
            rec = dict(level=r["level"], problem_id=r["problem_id"], name=r["name"],
                       op_class=r["op_class"], label_s105c=r["engine_class"],
                       s71_rho_global_strict=src["A_vs_anchor"]["ew_1e-4"]["rho_global"],
                       s71_t_fp64_s=src.get("t_fp64_s"), status="ok")
            if time.time() - t0 > WALL_CAP:
                rec["status"] = "wall_capped"
                f.write(json.dumps(rec) + "\n"); f.flush()
                rows.append(rec)
                continue
            tw = time.time()
            try:
                signal.alarm(ALARM)
                path = os.path.join(KB, "KernelBench", "level%d" % r["level"], r["name"])
                mod = load_task(path)
                torch.manual_seed(0)
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                inputs = mod.get_inputs()
                model = mod.Model(*init).cuda().eval()
                ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                ev_b = profile_fn(lambda: run_ref(model, ins, torch.float32, (True, True)))
                out_b = run_ref(model, ins, torch.float32, (True, True))
                out_x = run_ref(model, ins, torch.float64, (False, False))
                if out_x is None or out_b is None:
                    raise RuntimeError("missing output")
                if out_x.shape != out_b.shape:
                    raise RuntimeError("shape mismatch")
                if n_repeat < 3:
                    out_b2 = run_ref(model, ins, torch.float32, (True, True))
                    rec["k1_repeat_bit_identical"] = bool(torch.equal(out_b, out_b2))
                    n_repeat += 1
                    del out_b2
                if not anchor_done:
                    out_x2 = run_ref(model, ins, torch.float64, (False, False))
                    denom = float(out_x.abs().max().item()) or 1.0
                    rec["k2_anchor_rel"] = float((out_x - out_x2).abs().max().item()) / denom
                    anchor_done = True
                    del out_x2
                signal.alarm(0)
                rec["label_B"], rec["top1_B"] = classify([e[0] for e in ev_b])
                rec["B_strict"] = stats_vs(out_b, out_x, STRICT)
                rec["B_bench"] = stats_vs(out_b, out_x, BENCH)
                rec["ref_max_abs"] = float(out_x.abs().max().item())
                rec["u_strict_B"] = rec["ref_max_abs"] / STRICT
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
            print(rec["op_class"], rec["problem_id"], rec["status"], rec.get("label_B"),
                  (rec.get("B_strict") or {}).get("rho_ew"), (rec.get("B_strict") or {}).get("frac"),
                  rec.get("wall_s"), flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    k1 = [r.get("k1_repeat_bit_identical") for r in ok if "k1_repeat_bit_identical" in r]
    k2 = [r.get("k2_anchor_rel") for r in ok if "k2_anchor_rel" in r]
    instr_ok = len(ok) >= 90 and (not k1 or all(k1)) and (not k2 or max(k2) <= 1e-9)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    cols = {}
    for cls in sorted({r["op_class"] for r in ok}):
        sub = [r for r in ok if r["op_class"] == cls]
        cols[cls] = dict(n=len(sub),
                         n_label_B_TF32=sum(1 for r in sub if r.get("label_B") == "TF32_TENSOROP"),
                         n_relabel_tf32=sum(1 for r in sub if r.get("label_B") == "TF32_TENSOROP"
                                            and r.get("label_s105c") != "TF32_TENSOROP"),
                         med_rho_B_strict=med([r["B_strict"]["rho_ew"] for r in sub]),
                         med_rho_B_bench=med([r["B_bench"]["rho_ew"] for r in sub]),
                         n_exceed_B_strict=sum(1 for r in sub if r["B_strict"]["rho_ew"] > 1),
                         n_exceed_B_bench=sum(1 for r in sub if r["B_bench"]["rho_ew"] > 1),
                         labels_B=[[r["problem_id"], r.get("label_B"), r.get("top1_B")] for r in sub])
    undesc = [[r["op_class"], r["problem_id"], r.get("label_B"), r.get("top1_B")] for r in ok
              if r.get("label_B") == "TF32_TENSOROP" and r.get("label_s105c") != "TF32_TENSOROP"]
    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   wall_s=round(time.time() - t0, 1), n_rows=len(rows), n_ok=len(ok),
                   n_wall_capped=sum(1 for r in rows if r.get("status") == "wall_capped"),
                   n_timeout=sum(1 for r in rows if r.get("status") == "timeout"),
                   n_oom=sum(1 for r in rows if r.get("status") == "oom"),
                   n_error=sum(1 for r in rows if r.get("status") == "error"),
                   k1_repeat=k1, k2_anchor_rel=k2, instrument_ok=instr_ok,
                   cols=cols, n_undescribed_tf32=len(undesc), undescribed=undesc, rows=rows)
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("n_ok", "n_wall_capped", "k1_repeat", "k2_anchor_rel",
                                              "instrument_ok", "n_undescribed_tf32", "wall_s")}, indent=1))
    print(json.dumps({c: {k: v for k, v in cols[c].items() if k != "labels_B"} for c in cols}, indent=1))


if __name__ == "__main__":
    main()
