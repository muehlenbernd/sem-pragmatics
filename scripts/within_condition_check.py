import pandas as pd
from scipy.stats import ttest_ind
df = pd.read_csv('data/cs1_experiment/processed/experiment1.csv')
print('N =', len(df), '| conditions:', df.groupby(['context','form']).size().to_dict())
MAP = {"k_p":["knewExact","infoAvail"],"k_np":["notKnow","infoNotAvail"],"m_max":["fussyPerson"],
       "m_sit":["needed","purpose","realWorld"],"m_min":["easySpeaker","usualApx"]}
ATT = {"a_comp":"competent","a_like":"likeable","a_ped":"pedantic"}
cent = df.copy()
for a in ATT.values():
    cent[a] = df[a] - df.groupby(['context','form'])[a].transform('mean')
def stats(d):
    out={}
    for v,ps in MAP.items():
        for at,col in ATT.items():
            for p in ps:
                c='t2_'+p
                if d[c].sum()<5: continue
                g1=d.loc[d[c]==1,col]; g2=d.loc[d[c]==0,col]
                t,pv=ttest_ind(g1,g2,equal_var=True)
                out[(v,at,p)]=(g1.mean()-g2.mean(),pv,int(d[c].sum()))
    return out
def table(st,alpha):
    T={}
    for v,ps in MAP.items():
        for at in ATT:
            s=sum((1 if d>0 else -1) if pv<alpha else 0 for (vv,aa,p),(d,pv,n) in st.items() if vv==v and aa==at)
            T[(v,at)]= (s>0)-(s<0)
    return T
P,C=stats(df),stats(cent)
print('\nper-proxy: diff / p  (pooled | within-condition)')
for k in P:
    print(f"{k[0]:6}{k[1]:7}{k[2]:13} n={P[k][2]:3}  {P[k][0]:+.2f} p={P[k][1]:.4f} | {C[k][0]:+.2f} p={C[k][1]:.4f}")
results = {}
for alpha in [.001,.01,.05,.10]:
    tp,tc=table(P,alpha),table(C,alpha)
    print(f"\nalpha={alpha}   pooled | within   (! = differs)")
    for v in MAP:
        print(f"{v:6}"+"  ".join(f"{at[2:]}:{tp[(v,at)]:+d}|{tc[(v,at)]:+d}{'!' if tp[(v,at)]!=tc[(v,at)] else ' '}" for at in ATT))
    nzp=sum(x!=0 for x in tp.values()); nzc=sum(x!=0 for x in tc.values())
    sf=sum(tp[k]*tc[k]==-1 for k in tp)
    print('non-zero pooled', nzp, 'within', nzc, '| sign flips', sf)
    results[str(alpha)] = {"non_zero_pooled": nzp, "non_zero_within": nzc, "sign_flips": sf}

import numpy as np
from sem.comparison import sample_parameters
K=["k_p","k_np"]; M=["m_max","m_sit","m_min"]
def tru(k): return ["v_prc","v_apx"] if k=="k_p" else ["v_apx"]
def strat(m,c,k):
    if k=="k_np": return "v_apx"
    if m=="m_max": return "v_prc"
    if m=="m_sit": return "v_prc" if c=="c_HP" else "v_apx"
    return "v_apx"
def E(T,a,u,c,w,pk,pm):
    zk=sum(pk[k] for k in K if u in tru(k)); tot=0
    for k in K:
        if u not in tru(k): continue
        zm=sum(pm[m] for m in M if strat(m,c,k)==u)
        if zm==0: continue
        for m in M:
            if strat(m,c,k)!=u: continue
            tot+=pk[k]/zk*pm[m]/zm*(w*T[(k if k!='k_np' else 'k_np',a)]+(1-w)*T[(m,a)])
    return tot
def effects(T,w,pk,pm):
    e=lambda a,u,c:E(T,a,u,c,w,pk,pm)
    main=lambda a: e(a,'v_prc','c_HP')+e(a,'v_prc','c_LP')-e(a,'v_apx','c_HP')-e(a,'v_apx','c_LP')
    inter=lambda a: (e(a,'v_prc','c_HP')-e(a,'v_apx','c_HP'))-(e(a,'v_prc','c_LP')-e(a,'v_apx','c_LP'))
    return [main('a_comp')>0, main('a_ped')>0, inter('a_comp')>0, inter('a_like')>0]
samples = sample_parameters("full", n_samples=100, seed=42)
print('\nBalanced model (w=.5, flat) effects 1-4 reproduced? | sampled: all four reproduced in x/100')
for alpha in [.001,.01,.05,.10]:
    for name,st in [('pooled',P),('within',C)]:
        T=table(st,alpha)
        b=effects(T,.5,{'k_p':.5,'k_np':.5},{m:1/3 for m in M})
        s=sum(all(effects(T,samp["omega"],samp["priors_K"],samp["priors_M"])) for samp in samples)
        print(f"alpha={alpha:<5} {name:7} balanced={['✓' if x else '✗' for x in b]}  sampled={s}/100")
        results[str(alpha)][f"balanced_{name}"] = b
        results[str(alpha)][f"sampled_all4_{name}"] = s

import json, os
out_path = 'data/cs1_experiment/processed/within_condition_check_cs1.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults written to {out_path}')
