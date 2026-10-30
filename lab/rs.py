"""Minimal rStation API client for this tick's bookkeeping (GET/POST with error passthrough)."""
import json, os, sys, urllib.request, urllib.error

KEY = json.load(open(os.path.expanduser("~/.config/rstation/credentials.json")))["api_key"]
BASE = "https://rstation.ai/api/v1"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, method=method, data=data,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:2000]}


if __name__ == "__main__":
    method, path = sys.argv[1], sys.argv[2]
    body = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    s, d = call(method, path, body)
    print("STATUS", s)
    print(json.dumps(d, indent=1)[:6000])
