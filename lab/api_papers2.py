
import json,urllib.request
key=json.load(open('/Users/ckorhonen/.config/rstation/credentials.json'))
tok=key['api_key']
req=urllib.request.Request('https://rstation.ai/api/v1/problems/kernelbench/papers?limit=8&sort=new',
    headers={'Authorization':'Bearer '+tok})
with urllib.request.urlopen(req,timeout=60) as f: b=json.load(f)
print(json.dumps(b)[:1500])
