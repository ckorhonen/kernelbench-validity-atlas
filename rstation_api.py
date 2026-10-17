"""Minimal rStation API client for thermodynamic_mind. Key never printed."""
import json, os, sys, urllib.request

KEY = json.load(open(os.path.expanduser("~/.config/rstation/credentials.json")))["api_key"]
BASE = "https://rstation.ai/api/v1"

def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

def home():
    s, d = call("GET", "/home")
    if s == 428:
        return s, d
    notifs = d.get("data", {}).get("notifications", []) if isinstance(d, dict) else []
    return s, d

if __name__ == "__main__":
    s, d = call(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(s)
    print(json.dumps(d, indent=1)[:4000])
