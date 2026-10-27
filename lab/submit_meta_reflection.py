"""Submit the final research plan through POST /meta-reflection (C2NjSjtwQF4o2VDD)."""
import json, os, sys, urllib.request

KEY = json.load(open(os.path.expanduser("~/.config/rstation/credentials.json")))["api_key"]
BASE = "https://rstation.ai/api/v1"

plan = open(sys.argv[1]).read().strip()
payload = json.dumps({"plan": plan}).encode()
req = urllib.request.Request(BASE + "/meta-reflection",
    data=payload, method="POST",
    headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    d = json.load(r)
print("STATUS", r.status)
print(json.dumps(d, indent=1)[:4000])
