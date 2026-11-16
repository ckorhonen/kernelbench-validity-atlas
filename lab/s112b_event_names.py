#!/usr/bin/env python
"""s112b_event_names.py — declared post-hoc follow-up to s112 (no threshold, no branch).

s112's CONV regex matched cudnn::engines_precompiled::nhwcToNchwKernel through the bare token "cudnn", so the
layout kernel leaked into the "conv engine" set even though the prereg excluded it by construction. s112b
dumps the full profiler event list for one draw of each of the three rows so the actual conv engine can be
named from data instead of from a regex.
"""
import importlib.util, json, os, signal, time
import torch

KB = os.path.expanduser("~/kernelbench/KernelBench/KernelBench")
LAB = os.path.expanduser("~/kb-lab")
RES = os.path.join(LAB, "results")
OUT = os.path.join(RES, "s112b_event_names.json")
ROWS = [(2, 48), (2, 93), (2, 100)]
ALARM = 180


class Timeout(Exception):
    pass


def _alarm(s, f):
    raise Timeout("cell exceeded %ds" % ALARM)


signal.signal(signal.SIGALRM, _alarm)


def load_task(path):
    name = "kb_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_task(level, pid):
    base = os.path.join(KB, "level%d" % level)
    for f in sorted(os.listdir(base)):
        if f.startswith("%d_" % pid) and f.endswith(".py"):
            return os.path.join(base, f)
    raise FileNotFoundError("level%d/%d" % (level, pid))


def main():
    torch.cuda.init()
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from torch.profiler import profile, ProfilerActivity
    out = {}
    for level, pid in ROWS:
        path = find_task(level, pid)
        rec = {"level": level, "problem_id": pid, "name": os.path.basename(path)}
        try:
            signal.alarm(ALARM)
            mod = load_task(path)
            torch.manual_seed(0)
            init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            inputs = mod.get_inputs()
            model = mod.Model(*init).cuda().eval()
            ins = [x.cuda() if torch.is_tensor(x) else x for x in inputs]
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                with torch.no_grad():
                    model(*ins)
                torch.cuda.synchronize()
            signal.alarm(0)
            ev = [(e.key, float(e.self_device_time_total)) for e in prof.key_averages()
                  if e.self_device_time_total > 0]
            ev.sort(key=lambda t: -t[1])
            rec["status"] = "ok"
            rec["n_events"] = len(ev)
            rec["total_s"] = sum(t for _, t in ev)
            rec["events"] = [{"name": n, "s": round(t, 6), "share": round(t / rec["total_s"], 4)}
                             for n, t in ev[:12]]
            del model, ins, inputs
            torch.cuda.empty_cache()
        except Exception as e:
            signal.alarm(0)
            rec["status"] = "error"
            rec["err"] = repr(e)[:200]
        out["%d/%d" % (level, pid)] = rec
        print("==", level, pid, rec["status"])
        for e in rec.get("events", []):
            print("   ", e["share"], e["s"], e["name"][:110])
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
