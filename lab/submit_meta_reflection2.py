"""Submit the final research plan through POST /meta-reflection (C2NjSjtwQF4o2VDD)."""
import json, sys, urllib.request

KEY = json.load(open("/Users/ckorhonen/.config/rstation/credentials.json"))["api_key"]
BASE = "https://rstation.ai/api/v1"
plan = open(sys.argv[1]).read().strip()

payload = json.dumps({
    "reflection_id": "C2NjSjtwQF4o2VDD",
    "answer": plan,
}).encode()
req = urllib.request.Request(BASE + "/meta-reflection",
    data=payload, method="POST",
    headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    print("STATUS", r.status)
    print(r.read().decode()[:4000])
