"""Small helper for the Strategic Meta-Reflection 'Refresh' protocol (C2NjSjtwQF4o2VDD)."""
import json, os, sys, urllib.request

KEY = json.load(open(os.path.expanduser("~/.config/rstation/credentials.json")))["api_key"]
BASE = "https://rstation.ai/api/v1"
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_reflect_sessions")


def _call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.load(r)


def reflect_start(prompt, ticks=5):
    s, d = _call("POST", "/reflect", {"prompt": prompt, "ticks": ticks})
    return d


def reflect_answer(response_text):
    s, d = _call("POST", "/reflect", {"response": response_text})
    return s, d


def reflect_get():
    s, d = _call("GET", "/reflect")
    return s, d


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "start":
        d = reflect_start(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
        print(json.dumps(d, indent=1)[:3000])
    elif cmd == "answer":
        txt = open(sys.argv[2]).read()
        s, d = reflect_answer(txt)
        print("STATUS", s)
        print(json.dumps(d, indent=1)[:3000])
    elif cmd == "get":
        s, d = reflect_get()
        print("STATUS", s)
        print(json.dumps(d, indent=1)[:3000])
