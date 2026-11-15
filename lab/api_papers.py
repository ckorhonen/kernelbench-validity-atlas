
import json,urllib.request
key=json.load(open('/Users/ckorhonen/.config/rstation/credentials.json'))
tok=key['api_key']
def call(method,path,body=None):
    req=urllib.request.Request('https://rstation.ai/api/v1'+path, method=method,
        headers={'Authorization':'Bearer '+tok,'Content-Type':'application/json'},
        data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req,timeout=60) as f: return f.status, json.load(f)
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read().decode())
s,b=call('GET','/problems/kernelbench/papers?limit=10&sort=new')
print('papers',s)
if s==200:
    for p in b['data'].get('papers',b['data'].get('items',[]))[:8]:
        print(p.get('created_at'), p.get('title'), p.get('author',{}).get('handle') if isinstance(p.get('author'),dict) else p.get('author_handle'))
