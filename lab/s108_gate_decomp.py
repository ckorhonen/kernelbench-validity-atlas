#!/usr/bin/env python
"""s108_gate_decomp.py — decompose the two shipped-gate failures (L2/14, L2/18).

Prereg: lab/s108_prereg.md, committed before any row is computed.
Runs on spark: ~/kb-lab. Writes results/s108_gate_decomp.jsonl, s108_summary.json, s108_screen.json.
"""
import importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
OUT = os.path.join(LAB, "results", "s108_gate_decomp.jsonl")
SUM = os.path.join(LAB, "results", "s108_summary.json")
SCREEN = os.path.join(LAB, "results", "s108_screen.json")
TASKS = [(2, "14_Gemm_Divide_Sum_Scaling.py"), (2, "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py")]
STRICT, BENCH = 1e-4, 1e-2
CHUNK = 1 << 22
ALARM = 420
CLASS_PATTERNS = [("WINOGRAD", re.compile(r"winograd", re.I)),
                  ("TF32_TENSOROP", re.compile(r"tf32|tensorop|1688|16816|h1688", re.I)),
                  ("FP64", re.compile(r"dgemm|fp64|double", re.I)),
                  ("FP32_SIMT", re.compile(r"fprop|implicit_gemm|sgemm|fp32|simt|magma", re.I)),
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


def set_cfg(a, b):
    torch.backends.cudnn.allow_tf32 = a
    torch.backends.cuda.matmul.allow_tf32 = b


def run_model(model, ins, dtype, cfg):
    set_cfg(*cfg)
    m = model.to(dtype=dtype)
    xs = [x.to(dtype=dtype) if torch.is_tensor(x) else x for x in ins]
    with torch.no_grad():
        return m(*xs)


def anchor_fp64(model, ins, rows=128):
    """Chunked fp64 anchor: the tasks' post-gemm ops are row-wise, so row chunks are exact."""
    set_cfg(False, False)
    m = model.to(dtype=torch.float64)
    x = ins[0]
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], rows):
            outs.append(m(x[i:i + rows].double()))
    return torch.cat(outs, 0)


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
    return dict(rho_ew=mx, frac=n_fail / n, maxabs=gmx, rho_global=gmx / (gate + gate * ref_max), total=n)


def per_row(ref, anchor, gate):
    rf = ref.detach().double().reshape(ref.shape[0], -1)
    an = anchor.detach().double().reshape(anchor.shape[0], -1)
    d = (rf - an).abs().max(dim=1)[0]
    x = an.abs().max(dim=1)[0]
    rho = d / (gate + gate * x)
    return d, x, rho


def spearman(a, b):
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def text_screen():
    """K4 declared heuristic: matmul-ish token followed by a full-axis reduction with keepdim."""
    pat_mm = re.compile(r"torch\.matmul|torch\.bmm|nn\.Linear|torch\.mm\(|F\.linear")
    pat_red = re.compile(r"(torch\.)?(sum|max|mean|amax)\s*\([^)]*dim\s*=\s*1[^)]*keepdim\s*=\s*True")
    hits, scanned = [], 0
    for lvl in (1, 2, 3):
        d = os.path.join(KB, "KernelBench", "level%d" % lvl)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            src = open(os.path.join(d, fn)).read()
            scanned += 1
            m = pat_mm.search(src)
            if not m:
                continue
            tail = src[m.end():]
            if pat_red.search(tail):
                hits.append([lvl, fn])
    return dict(scanned=scanned, hits=hits, n_hits=len(hits))


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    rows = []
    screen = text_screen()
    json.dump(screen, open(SCREEN, "w"), indent=1)
    print("screen scanned", screen["scanned"], "hits", screen["n_hits"], flush=True)

    anchor_check_done = False
    with open(OUT, "a") as f:
        for lvl, fn in TASKS:
            rec = dict(level=lvl, name=fn, status="ok")
            tw = time.time()
            try:
                signal.alarm(ALARM)
                mod = load_task(os.path.join(KB, "KernelBench", "level%d" % lvl, fn))
                torch.manual_seed(0)
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                inputs = mod.get_inputs()
                model = mod.Model(*init).cuda().eval()
                ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                rec["in_shape"] = list(ins[0].shape)

                X = anchor_fp64(model, ins)
                rec["ref_max_abs"] = float(X.abs().max().item())
                rec["out_numel"] = int(X.numel())
                if not anchor_check_done:
                    X2 = anchor_fp64(model, ins)
                    den = float(X.abs().max().item()) or 1.0
                    rec["k2_anchor_rel"] = float((X - X2).abs().max().item()) / den
                    anchor_check_done = True
                    del X2
                rec["k2_anchor_rel"] = rec.get("k2_anchor_rel", 0.0)

                R_B = run_model(model, ins, torch.float32, (True, True))
                ev_b = profile_fn(lambda: run_model(model, ins, torch.float32, (True, True)))
                R_B2 = run_model(model, ins, torch.float32, (True, True))
                rec["k1_repeat_bit_identical"] = bool(torch.equal(R_B, R_B2))
                del R_B2
                R_A = run_model(model, ins, torch.float32, (True, False))
                ev_a = profile_fn(lambda: run_model(model, ins, torch.float32, (True, False)))

                rec["label_B"], rec["top1_B"] = classify([e[0] for e in ev_b])
                rec["label_A"], rec["top1_A"] = classify([e[0] for e in ev_a])
                rec["R_B"] = {str(g): stats_vs(R_B, X, g) for g in (STRICT, BENCH)}
                rec["R_A"] = {str(g): stats_vs(R_A, X, g) for g in (STRICT, BENCH)}

                # mechanism leg on R_B vs X at the bench gate
                d, xm, rho = per_row(R_B, X, BENCH)
                rec["mech"] = dict(
                    rho_row_med=float(rho.median().item()), rho_row_max=float(rho.max().item()),
                    d_med=float(d.median().item()), d_max=float(d.max().item()),
                    d_q=[float(v) for v in torch.quantile(d, torch.tensor([0.1, 0.5, 0.9]).double()).tolist()],
                    x_med=float(xm.median().item()), x_min=float(xm.min().item()),
                    x_p01=float(torch.quantile(xm, torch.tensor([0.01]).double()).item()),
                    spearman_x_rho=spearman(xm, rho), spearman_x_d=spearman(xm, d),
                    n_rows=int(rho.numel()), n_fail_rows=int((rho > 1).sum().item()))
                thr = rec["mech"]["d_med"] / BENCH - 1.0
                rec["mech"]["cond_threshold"] = thr
                rec["mech"]["frac_pred"] = float((xm < thr).double().mean().item())
                rec["mech"]["frac_meas"] = rec["mech"]["n_fail_rows"] / rec["mech"]["n_rows"]

                # compile arms
                for tag, cfg in (("C_B", (True, True)), ("C_A", (True, False))):
                    try:
                        torch._dynamo.reset()
                        set_cfg(*cfg)
                        cm = torch.compile(model, dynamic=False)
                        with torch.no_grad():
                            out = cm(*ins)
                        torch.cuda.synchronize()
                        ev = profile_fn(lambda: cm(*ins))
                        rec[tag] = {str(g): stats_vs(out, X, g) for g in (STRICT, BENCH)}
                        rec[tag]["label"], rec[tag]["top1"] = classify([e[0] for e in ev])
                        ref_arm = R_B if tag == "C_B" else R_A
                        rec[tag]["shipped_allclose"] = bool(torch.allclose(
                            out, ref_arm, atol=BENCH, rtol=BENCH))
                        rec[tag]["vs_ref"] = {str(g): stats_vs(out, ref_arm, g) for g in (STRICT, BENCH)}
                        rec[tag]["bit_identical_to_ref"] = bool(torch.equal(out, ref_arm))
                        del out, cm
                    except Exception as e:
                        rec[tag] = dict(status="error", err=repr(e)[:300])
                        torch._dynamo.reset()
                signal.alarm(0)
                rec["wall_s"] = round(time.time() - tw, 2)
                del model, ins, inputs, R_B, R_A, X
                torch.cuda.empty_cache()
            except Timeout:
                signal.alarm(0)
                rec["status"] = "timeout"
            except Exception as e:
                signal.alarm(0)
                rec["status"] = "error"
                rec["err"] = repr(e)[:300]
                torch.cuda.empty_cache()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            rows.append(rec)

    # frozen decision rules, printed not chosen
    summary = dict(rows=[], rules={}, screen=screen)
    for rec in rows:
        if rec.get("status") != "ok":
            summary["rows"].append(dict(name=rec["name"], status=rec["status"]))
            continue
        fb = rec["R_B"][str(BENCH)]["frac"]
        fa = rec["R_A"][str(BENCH)]["frac"]
        if fb > 0 and fa == 0:
            branch = "BRANCH_PROCEDURE_CARRIED"
        elif fb > 0 and fa > 0 and (max(fb, fa) / max(min(fb, fa), 1e-12)) <= 3:
            branch = "BRANCH_REFERENCE_CARRIED"
        elif fb > 0 and fa > 0:
            branch = "BRANCH_MIXED"
        else:
            branch = "BRANCH_NO_FAILURE"
        m = rec["mech"]
        ratio = max(m["frac_pred"], m["frac_meas"]) / max(min(m["frac_pred"], m["frac_meas"]), 1e-12)
        mech = ("MECH_CANCELLATION" if (m["spearman_x_rho"] <= -0.5 and ratio <= 2.0)
                else "MECH_NOT_CANCELLATION")
        ship = "SHIPPED_CONSISTENCY"
        cb = rec.get("C_B", {})
        if cb.get("status") == "error" or not cb:
            ship = "COMPILE_ARM_FAILED"
        elif not (cb["vs_ref"][str(BENCH)]["rho_ew"] <= 1.0 and rec["R_B"][str(BENCH)]["rho_ew"] > 1.0):
            ship = "SHIPPED_BINDING"
        summary["rules"][rec["name"]] = dict(
            ownership=branch, mechanism=mech, shipped=ship,
            frac_B=fb, frac_A=fa, frac_pred=m["frac_pred"], frac_meas=m["frac_meas"],
            spearman_x_rho=m["spearman_x_rho"], spearman_x_d=m["spearman_x_d"],
            rho_ew_B=rec["R_B"][str(BENCH)]["rho_ew"], rho_ew_A=rec["R_A"][str(BENCH)]["rho_ew"],
            d_med=m["d_med"], d_max=m["d_max"], x_p01=m["x_p01"], x_min=m["x_min"],
            k1=rec.get("k1_repeat_bit_identical"), k2=rec.get("k2_anchor_rel"))
        summary["rows"].append(dict(name=rec["name"], branch=branch, mech=mech, ship=ship))
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps(summary["rules"], indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
