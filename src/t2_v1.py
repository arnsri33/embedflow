from __future__ import annotations
from pathlib import Path
import hashlib, json

K_VALUES=(10,20,50,100,200,500)
def verify_t2_hash(root:Path)->None:
    p=root/'frozen/T2_V1_FROZEN_SPEC.md'; expected=(root/'frozen/T2_V1_FROZEN_SPEC.sha256').read_text().strip(); actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if expected!=actual: raise RuntimeError(f'T2 frozen hash mismatch: {actual} != {expected}')

def decide(features:dict)->str:
    safe=(features['probe_residual_tail_50_mean']<=.05 and features['deepest_p90']<=200 and features['late_tail_area']<=.20)
    if safe:return 'SAFE'
    guard=(features['probe_residual_tail_50_mean']>.10 or features['stability_to_500_50_mean']<.90 or features['last_shell_any_rate']>.25 or features['fraction_margin_nonpositive']>.50)
    return 'UNSAFE_OR_UNCERTAIN' if guard else 'EXPAND'

def blind_predictions(rows:list[dict], out:Path)->None:
    keys=['pair','dataset','probe_residual_tail_50_mean','stability_to_500_50_mean','deepest_p90','late_tail_area','last_shell_any_rate','fraction_margin_nonpositive']
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        f.write(','.join(keys+['T2_v1_decision'])+'\n')
        for r in rows:f.write(','.join(str(r.get(k,'')) for k in keys+[None])[:-1]+str(decide(r))+'\n')
