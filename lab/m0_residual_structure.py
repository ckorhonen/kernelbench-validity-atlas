#!/usr/bin/env python3
"""M0 residual-structure test -- revival plan M0. PRE-REGISTERED 2026-10-18T04:20Z, before eval.

Revival premise under test (deep reflection #1/#2): once the regime (gate-selected hand domain) is
fixed, the advantage surface is low-dimensional -- a linear model on <=2 declared coordinates predicts
the advantage with LOO Spearman rho >= 0.4, and a 3rd coordinate adds < 0.05.

Domains (declared BEFORE eval):
  reduction: e2b transfer grid (144 pts, hand arm tsplit_pinned), gate E: hand iff M>=16 AND
             bt>=1024, bt = 2*bytes/(ctas*128), ctas from the deployed variant row.
  conv:      w3_conv_gemm grid (36 pts, hand arm imp_B32x32), domain (H<=14 & C<=64) | H<=7
             (labeled post-hoc rule from session 8; its stability is M2's job, not this test's).
Coords per family (declared):
  reduction: lgM, lgK, bt, waves=ctas/48, r=t_torch/t_launch2(7.904us; circularity declared, v1 precedent)
  conv:      lgM, lgH, lgC, bt, waves (ctas from deployed variant row if present else fallback declared)
Test: for subset size k=1,2,3 over the declared coords, LOO linear fit -> Spearman(pred, actual).
PASS (revival premise survives): best-2-coord LOO rho >= 0.4 AND best3-best2 < 0.05, in EITHER family...
NO -- in BOTH families? Premise is per-family; program-level revival needs >=1 family passing AND the
other not contradicting (rho best2 >= 0.2). FALSIFIER (no-collapse, P5): best-2 rho < 0.4 in BOTH
families => gates absorbed regime mixing => record and pivot to candidate B.
Control: same computation on the complement domain, no bar, reported for context.
Permutation p for the best pair: 2000 shuffles, one-sided.
"""
import json, math, itertools, random
from itertools import combinations

def rank(x):
    idx=sorted(range(len(x)), key=lambda i:x[i]); r=[0]*len(x)
    for pos,i in enumerate(idx): r[i]=pos
    return r

def spearman(a,b):
    ra,rb=rank(a),rank(b); n=len(a)
    ma=sum(ra)/n; mb=sum(rb)/n
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    da=math.sqrt(sum((ra[i]-ma)**2 for i in range(n))); db=math.sqrt(sum((rb[i]-mb)**2 for i in range(n)))
    return num/(da*db) if da>0 and db>0 else 0.0

def ols_fit(X,y):
    n=len(y); Xl=[xi+[1.0] for xi in X]; k=len(Xl[0])
    # normal equations, tiny k
    A=[[sum(Xl[r][i]*Xl[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    b=[sum(Xl[r][i]*y[r] for r in range(n)) for i in range(k)]
    # gaussian elim
    for c in range(k):
        p=max(range(c,k),key=lambda r:abs(A[r][c])); A[c],A[p]=A[p],A[c]; b[c],b[p]=b[p],b[c]
        if abs(A[c][c])<1e-12: continue
        for r2 in range(k):
            if r2!=c and A[r2][c]:
                f=A[r2][c]/A[c][c]
                for cc in range(c,k): A[r2][cc]-=f*A[c][cc]
                b[r2]-=f*b[c]
    w=[b[i]/A[i][i] if abs(A[i][i])>1e-12 else 0.0 for i in range(k)]
    return w

def ols_pred(w,x): return sum(wi*xi for wi,xi in zip(w,x+[1.0]))

def loo_spearman(X,y):
    n=len(y); preds=[0.0]*n
    for i in range(n):
        Xtr=[X[j] for j in range(n) if j!=i]; ytr=[y[j] for j in range(n) if j!=i]
        w=ols_fit(Xtr,ytr); preds[i]=ols_pred(w,X[i])
    return spearman(preds,y)

def perm_p(X,y,rho_obs,n_perm=2000,seed=7):
    rnd=random.Random(seed); n=len(y); cnt=0
    for _ in range(n_perm):
        y2=y[:]; rnd.shuffle(y2)
        if loo_spearman(X,y2)>=rho_obs-1e-12: cnt+=1
    return cnt/float(n_perm)

def load_reduction():
    e=json.load(open('results/e2b_floor.json'))
    t_launch=7.903999648988247 # us, two-launch floor (P1: from e2b meta)
    D,Comp=[],[]
    for row in e['rows']:
        v=row['variants']; t_torch=v['torch']['med_ms']*1000; t_hand=v['tsplit_pinned']['med_ms']*1000
        ctas=v['tsplit_pinned'].get('ctas',64); bt=2*row['bytes']/(ctas*128)
        feats={'lgM':math.log2(row['M']),'lgK':math.log2(row['K']),'bt':bt,'waves':ctas/48,
               'r':t_torch/t_launch}
        rec={'a':(t_torch-t_hand)/t_torch,**{k:[v2] for k,v2 in feats.items()},'M':row['M']}
        (D if (row['M']>=16 and bt>=1024) else Comp).append(rec)
    return D,Comp,['lgM','lgK','bt','waves','r']

def load_conv():
    w=json.load(open('results/w3_conv_gemm.json'))
    D,Comp=[],[]
    for row in w['rows']:
        v=row['variants']; t_torch=v['torch']['med_ms']*1000
        hand='imp_B32x32'
        if hand not in v: continue
        t_hand=v[hand]['med_ms']*1000
        ctas=v[hand].get('ctas')  # deployed-variant ctas if recorded
        ctas=ctas if ctas else max(1,(row['M']+31)//32)*((row['CO']+31)//32)
        bt=2*row['bytes']/(ctas*128)
        feats={'lgM':math.log2(max(row['M'],2)),'lgH':math.log2(row['H']),'lgC':math.log2(row['C']),
               'bt':bt,'waves':ctas/48}
        rec={'a':(t_torch-t_hand)/t_torch,**{k:[x] for k,x in feats.items()},'H':row['H'],'C':row['C']}
        (D if (row['H']<=14 and row['C']<=64) or row['H']<=7 else Comp).append(rec)
    return D,Comp,['lgM','lgH','lgC','bt','waves']

def run(name,D,Comp,coords):
    out={'family':name,'n_domain':len(D),'n_comp':len(Comp),'subsets':{}}
    if len(D)<8:
        out['verdict']='insufficient n'; return out
    Xall={c:[r[c][0] for r in D] for c in coords}; y=[r['a'] for r in D]
    best={}
    for k in (1,2,3):
        for sub in combinations(coords,k):
            X=[Xall[c] for c in sub]; X=[[xi[j] for xi in X] for j in range(len(y))]
            best[sub]=loo_spearman(X,y)
    for k in (1,2,3):
        cand={s:v for s,v in best.items() if len(s)==k}
        sb=max(cand,key=cand.get); out['subsets']['best%d'%k]={'coords':list(sb),'loo_rho':round(cand[sb],4)}
    b2=out['subsets']['best2']['loo_rho']; b3=out['subsets']['best3']['loo_rho']
    X=[Xall[c] for c in out['subsets']['best2']['coords']]; X=[[xi[j] for xi in X] for j in range(len(y))]
    out['best2_perm_p']=round(perm_p(X,y,b2),4)
    out['marginal_3rd']=round(b3-b2,4)
    # control
    if len(Comp)>=8:
        Xc={c:[r[c][0] for r in Comp] for c in coords}; yc=[r['a'] for r in Comp]
        cb=[]
        for sub in combinations(coords,2):
            X2=[[Xc[sub[0]][j],Xc[sub[1]][j]] for j in range(len(yc))]
            cb.append((sub,loo_spearman(X2,yc)))
        sbc=max(cb,key=lambda t:t[1])
        out['control_best2']={'coords':list(sbc[0]),'loo_rho':round(sbc[1],4)}
    ok = b2>=0.4 and (b3-b2)<0.05
    out['premise_pass']=ok
    return out

def main():
    res={'meta':{'pre_reg':'module docstring 2026-10-18T04:20Z','date':'2026-10-18T04:25Z'}}
    D,C,coords=load_reduction(); res['reduction']=run('reduction',D,C,coords)
    D,C,coords=load_conv();      res['conv']=run('conv',D,C,coords)
    rp=res['reduction']; cp=res['conv']
    both_fail = (not rp.get('premise_pass')) and (not cp.get('premise_pass'))
    contradict = (rp.get('premise_pass') and rp['subsets']['best2']['loo_rho']<0.2) or \
                 (cp.get('premise_pass') and cp['subsets']['best2']['loo_rho']<0.2)
    res['program_verdict']=('FALSIFIED: no family reaches bar => P5 no-collapse, pivot B'
        if both_fail else ('PASS with weak family' if contradict else 'PASS: revival premise survives M0'))
    json.dump(res,open('results/m0_residual.json','w'),indent=1)
    print(json.dumps(res,indent=1)[:2400])

if __name__=='__main__': main()
