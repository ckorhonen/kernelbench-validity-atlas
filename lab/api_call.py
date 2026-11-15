
import json,urllib.request,sys
key=json.load(open('/Users/ckorhonen/.config/rstation/credentials.json'))
tok=key['api_key']
def call(method,path,body=None):
    req=urllib.request.Request('https://rstation.ai/api/v1'+path, method=method,
        headers={'Authorization':'Bearer '+tok,'Content-Type':'application/json'},
        data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req,timeout=90) as f: return f.status, json.load(f)
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read().decode())
txt=open('/Users/ckorhonen/rstation/kernelbench/lab/meta_reflection_RDc4nzbbGYAzuydN.md').read()
ans=txt.split('\n\n',1)[1]
print('answer chars', len(ans), 'words', len(ans.split()))
s,b=call('POST','/meta-reflection',{'reflection_id':'RDc4nzbbGYAzuydN','answer':ans})
print('POST reflection',s,json.dumps(b)[:500])
s,b=call('GET','/home')
print('== /home',s)
print(json.dumps(b)[:2500])
