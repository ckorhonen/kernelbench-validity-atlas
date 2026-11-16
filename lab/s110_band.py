#!/usr/bin/env python
"""s110_band.py — 16-draw band on the 23 ATLAS_MAP rows in rho_ew [0.5, 2.0].
Prereg: lab/s110_prereg.md, committed before any row is computed.
"""
import importlib.util, json, os, re, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s110_band.jsonl")
SUM = os.path.join(RES, "s110_summary.json")
PICK = os.path.join(RES, "s110_picked.json")
S107 = os.path.join(RES, "s107_rekey_map.jsonl")
S107B = os.path.join(RES, "s107b_rekey_rest.jsonl")
ALARM = 240
WALL_CAP = 2700
CHUNK = 1 << 22
STRICT = 1e-4
BENCH = 1e-2
NDRAW = 16
LO, HI = 0.5, 2.0
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


def select():
    picked = []
    for path in (S107, S107B):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            rho = (r.get("B_strict") or {}).get("rho_ew")
            if rho is None or not (LO <= rho <= HI):
                continue
            picked.append(r)
    picked.sort(key=lambda r: (r.get("s71_t_fp64_s") or 99))
    return picked


def main():
    torch.cuda.init()
    open(OUT, "w").close()
    picked = select()
    n_ex = sum(1 for r in picked if r["B_strict"]["rho_ew"] > 1)
    print("picked", len(picked), "recorded_exceed", n_ex, flush=True)
    json.dump({"n_picked": len(picked), "n_recorded_exceed": n_ex,
               "picked": [[r["op_class"], r["level"], r["problem_id"], r["name"],
                           r.get("s71_t_fp64_s"), r["B_strict"]["rho_ew"]] for r in picked]},
              open(PICK, "w"), indent=1)

    t0 = time.time()
    rows = []
    k2_done = False
    with open(OUT, "a") as f:
        for r in picked:
            base = dict(level=r["level"], problem_id=r["problem_id"], name=r["name"],
                        op_class=r["op_class"], recorded_rho_B_strict=r["B_strict"]["rho_ew"],
                        recorded_frac_B_strict=r["B_strict"]["frac"],
                        recorded_u_strict_B=r.get("u_strict_B"),
                        s71_t_fp64_s=r.get("s71_t_fp64_s"))
            path = os.path.join(KB, "level%d" % r["level"], r["name"])
            try:
                signal.alarm(ALARM)
                mod = load_task(path)
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                rec = dict(base, draw=-1, status="load_error", err=repr(e)[:200])
                f.write(json.dumps(rec) + "\n"); f.flush(); rows.append(rec)
                continue

            d1_done = False
            for d in range(NDRAW):
                rec = dict(base, draw=d, seed=d, status="ok")
                if time.time() - t0 > WALL_CAP:
                    rec["status"] = "wall_capped"
                    f.write(json.dumps(rec) + "\n"); f.flush(); rows.append(rec)
                    continue
                tw = time.time()
                try:
                    signal.alarm(ALARM)
                    torch.manual_seed(d)
                    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                    inputs = mod.get_inputs()
                    model = mod.Model(*init).cuda().eval()
                    ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
                    ev = profile_fn(lambda: run_ref(model, ins, torch.float32, (True, True)))
                    out_b = run_ref(model, ins, torch.float32, (True, True))
                    out_x = run_ref(model, ins, torch.float64, (False, False))
                    if out_x is None or out_b is None:
                        raise RuntimeError("missing output")
                    if out_x.shape != out_b.shape:
                        raise RuntimeError("shape mismatch")
                    if not k2_done:
                        out_x2 = run_ref(model, ins, torch.float64, (False, False))
                        denom = float(out_x.abs().max().item()) or 1.0
                        rec["k2_anchor_rel"] = float((out_x - out_x2).abs().max().item()) / denom
                        k2_done = True
                        del out_x2
                    if d == 0 and not d1_done:
                        out_b2 = run_ref(model, ins, torch.float32, (True, True))
                        rec["d1_bit_identical"] = bool(torch.equal(out_b, out_b2))
                        mx = 0.0
                        rf, r2 = out_b.reshape(-1), out_b2.reshape(-1)
                        n = rf.numel(); i = 0
                        while i < n:
                            j = min(n, i + CHUNK)
                            mx = max(mx, float((rf[i:j].double() - r2[i:j].double()).abs().max().item()))
                            i = j
                        amax = float(out_x.abs().max().item())
                        rec["d1_delta_max"] = mx
                        rec["d1_delta_rho_repeat"] = mx / (STRICT + STRICT * amax)
                        del out_b2
                        d1_done = True
                    signal.alarm(0)
                    rec["label"], rec["top1"] = classify([e[0] for e in ev])
                    rec["strict"] = stats_vs(out_b, out_x, STRICT)
                    rec["bench"] = stats_vs(out_b, out_x, BENCH)
                    rec["ref_max_abs"] = float(out_x.abs().max().item())
                    rec["u_strict"] = rec["ref_max_abs"] / STRICT
                    if d == 0:
                        rr = rec["recorded_rho_B_strict"]
                        rec["tie_out_rel"] = abs(rec["strict"]["rho_ew"] - rr) / rr if rr else None
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
                print(r["op_class"], r["problem_id"], d, rec["status"], rec.get("label"),
                      (rec.get("strict") or {}).get("rho_ew"), (rec.get("bench") or {}).get("frac"),
                      rec.get("wall_s"), flush=True)

    def stats(xs):
        if not xs:
            return None
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return dict(n=len(xs), mean=m, sd=sd, cv=sd / m if m else None, min=min(xs), max=max(xs))

    per_row = {}
    for r in picked:
        key = (r["level"], r["problem_id"])
        cells = [x for x in rows if (x["level"], x["problem_id"]) == key
                 and x.get("status") == "ok" and x.get("strict")]
        rhos = [x["strict"]["rho_ew"] for x in cells]
        n_ok = len(cells)
        r_i = (sum(1 for v in rhos if v > 1) / NDRAW) if n_ok else None
        bench_fracs = [x["bench"]["frac"] for x in cells]
        bench_rho = [x["bench"]["rho_ew"] for x in cells]
        per_row["%s/%s" % key] = dict(op_class=r["op_class"], name=r["name"], n_ok=n_ok,
                                     recorded=r["B_strict"]["rho_ew"],
                                     rho=stats(rhos), exceed_rate=r_i,
                                     bench_frac=stats(bench_fracs), bench_rho=stats(bench_rho),
                                     bench_frac_any_positive=any(v > 0 for v in bench_fracs),
                                     labels=sorted({x.get("label") for x in cells if x.get("label")}),
                                     statuses=sorted({x.get("status") for x in cells}),
                                     tie_out_rel=next((x.get("tie_out_rel") for x in cells if x.get("draw") == 0), None),
                                     d1=[{k: x.get(k) for k in ("d1_bit_identical", "d1_delta_max", "d1_delta_rho_repeat")}
                                         for x in cells if x.get("draw") == 0])

    measured = {k: v for k, v in per_row.items() if v["n_ok"] >= 8 and v["exceed_rate"] is not None}
    k1_cells = [x for x in rows if x.get("draw") == 0 and "d1_bit_identical" in x]
    k1_ok = sum(1 for x in k1_cells if x["d1_bit_identical"])
    k2 = next((x.get("k2_anchor_rel") for x in rows if "k2_anchor_rel" in x), None)
    tie = [v["tie_out_rel"] for v in measured.values() if v.get("tie_out_rel") is not None]
    tie_ok = sum(1 for t in tie if t <= 0.05)
    n_load_fail = len(per_row) - len(measured)
    instr_ok = (k1_ok >= max(1, int(0.75 * len(k1_cells))) and (k2 is None or k2 <= 1e-9) and tie_ok >= 6)

    R = sum(v["exceed_rate"] for v in measured.values()) if instr_ok else None
    mixed = [k for k, v in measured.items() if instr_ok and 0 < v["exceed_rate"] < 1]
    bench_row = [k for k, v in measured.items() if v["bench_frac_any_positive"]]

    if not instr_ok:
        branch = "INSTRUMENT_FAILED"
    elif mixed:
        branch = "BRANCH_COUNT_BAND"
    else:
        branch = "BRANCH_COUNT_STABLE"
    bench_branch = "BRANCH_BENCH_GATE_ROW" if bench_row else "BRANCH_BENCH_GATE_CLEAN"
    partial = (len(measured) < len(picked)) and (len(picked) - len(measured) > 6)

    summary = dict(date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   wall_s=round(time.time() - t0, 1),
                   n_picked=len(picked), n_measured=len(measured), n_rows=len(rows),
                   n_ok_cells=sum(1 for x in rows if x.get("status") == "ok"),
                   n_recorded_exceed=sum(1 for r in picked if r["B_strict"]["rho_ew"] > 1),
                   k1_bit_identical=k1_ok, k1_n=len(k1_cells), k2_anchor_rel=k2,
                   tie_out=tie, tie_out_ok=tie_ok, instrument_ok=instr_ok,
                   branch=branch, bench_branch=bench_branch, partial=partial,
                   mixed_rows=mixed, bench_gate_rows=bench_row,
                   R_sum_exceed_rate=R,
                   count_point=69 + R if R is not None else None,
                   per_row=per_row, rows=rows)
    json.dump(summary, open(SUM, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("branch", "bench_branch", "instrument_ok", "k1_bit_identical",
                                              "k1_n", "k2_anchor_rel", "tie_out", "n_ok_cells", "n_measured",
                                              "R_sum_exceed_rate", "count_point", "mixed_rows", "bench_gate_rows",
                                              "partial", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
