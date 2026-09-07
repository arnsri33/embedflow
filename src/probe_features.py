from __future__ import annotations
import numpy as np

def top10(records): return [x[0] for x in records[:10]]
def build_features(candidates:list[list[tuple[str,float,int]]], target_scores:dict[str,float], ks=(10,20,50,100,200,500)):
    # candidates are source-ranked (id, source_score, source_rank); target_scores are finite-pool scores.
    rerank={k:sorted(candidates[:k],key=lambda x:(-target_scores[x[0]],x[2])) for k in ks}; final=top10(rerank[max(ks)])
    def stability(k): return len(set(top10(rerank[k]))&set(final))/10
    def entrants(a,b): return len(set(top10(rerank[b]))-set(top10(rerank[a])))
    e50=entrants(50,100); e100=entrants(100,200); e200=entrants(200,500)
    margins=[]
    for q in [candidates]:
        t10=rerank[50][min(9,len(rerank[50])-1)][1] if not target_scores else target_scores[rerank[50][min(9,len(rerank[50])-1)][0]]
        new=[target_scores[x[0]] for x in q[50:100]]; margins.append(t10-max(new) if new else 0.0)
    deepest=max((x[2] for x in rerank[500][:10]),default=0)
    vals={
      'probe_residual_tail_50_mean':1-stability(50),'stability_to_500_50_mean':stability(50),
      'deepest_p90':float(deepest),'late_tail_area':(e50/10+e100/10+e200/10)/3,
      'last_shell_any_rate':float(e200>0),'fraction_margin_nonpositive':float(np.mean(np.asarray(margins)<=0)),
      'p_any_entrant_50_100':float(e50>0),'p_any_entrant_100_200':float(e100>0),'p_any_entrant_200_500':float(e200>0),
      'mean_entrants_50_100':float(e50),'mean_entrants_100_200':float(e100),'mean_entrants_200_500':float(e200),
      'deepest_source_rank_top10':deepest,'stability_10_20':stability(10),'stability_20_50':stability(20),'stability_50_100':len(set(top10(rerank[50]))&set(top10(rerank[100])))/10,
    }
    return vals
