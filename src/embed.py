from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
from .storage import write_shard, valid_shard
from .utils import progress

class DummyEncoder:
    def __init__(self, dimension:int, seed:int): self.dimension=dimension; self.seed=seed
    def encode(self,texts,role='document'):
        out=[]
        for text in texts:
            import hashlib
            rng=np.random.default_rng(int(hashlib.sha256((str(self.seed)+role+'\0'+text).encode()).hexdigest()[:16],16))
            v=rng.normal(size=self.dimension).astype('float32'); v/=max(np.linalg.norm(v),1e-12); out.append(v)
        return np.asarray(out,dtype='float32'),sum(max(1,len(x.split())) for x in texts)
    def close(self): pass

class TransformersEncoder:
    """Qwen-style encoder following the frozen model contract.

    The implementation deliberately supports CPU fallback for the tiny local
    verification run.  Remote jobs still select CUDA automatically and only
    the batch size may be reduced after an OOM.
    """
    def __init__(self, config:dict, model_dir:Path, device='cuda'):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.c=config; self.device=torch.device(device); self.tok=AutoTokenizer.from_pretrained(str(model_dir),revision=config['revision'],local_files_only=True,use_fast=True)
        self.tok.padding_side=config.get('padding_side','left'); self.tok.truncation_side='right'; self.tok.pad_token=self.tok.pad_token or self.tok.eos_token
        if self.device.type == 'cuda':
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() and config.get('dtype_preference')=='bfloat16' else torch.float16
        else:
            dtype=torch.float32
        self.net=AutoModel.from_pretrained(str(model_dir),revision=config['revision'],local_files_only=True,torch_dtype=dtype,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().to(self.device)
    def encode(self,texts,role='document'):
        import torch
        templ=self.c.get('query_instruction','') if role=='query' else self.c.get('document_instruction','')
        texts=[templ.replace('{text}',x) if templ else x for x in texts]
        e=self.tok(texts,padding=True,truncation=True,max_length=int(self.c.get('max_length',8192)),return_tensors='pt').to(self.device)
        amp = self.device.type == 'cuda'
        context = torch.autocast('cuda',dtype=next(self.net.parameters()).dtype) if amp else torch.autocast('cpu', enabled=False)
        with torch.inference_mode(), context:
            o=self.net(**e); h=o.last_hidden_state; pos=torch.full((h.shape[0],),h.shape[1]-1,dtype=torch.long,device=h.device); v=h[torch.arange(h.shape[0],device=h.device),pos]; v=torch.nn.functional.normalize(v.float(),p=2,dim=1)
        return v.cpu().numpy().astype('float32'),int(e['attention_mask'].sum().item())
    def close(self):
        import torch
        del self.net,self.tok; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

class SentenceEncoder:
    def __init__(self, config:dict, model_dir:Path, device='cuda'):
        from sentence_transformers import SentenceTransformer
        kwargs={'device':device}
        # Older sentence-transformers versions accept local_files_only only
        # through the underlying HF loader; a local snapshot is always passed
        # here, so avoiding the kwarg keeps this compatible across versions.
        self.model=SentenceTransformer(str(model_dir),**kwargs)
        self.model.max_seq_length=int(config.get('max_length',512))
    def encode(self,texts,role='document'):
        v=self.model.encode(texts,batch_size=32,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
        return np.asarray(v,dtype='float32'),sum(max(1,len(x.split())) for x in texts)
    def close(self): del self.model; gc.collect()

def make_encoder(config, model_dir:Path, smoke=False, seed=20270823, device='cuda'):
    if smoke: return DummyEncoder(int(config['dimension']),seed+hash(config['model_id'])%10000)
    return SentenceEncoder(config,model_dir,device) if str(config['model_id']).startswith('sentence-transformers/') else TransformersEncoder(config,model_dir,device)

def is_oom(exc: Exception) -> bool:
    return 'out of memory' in str(exc).lower() or 'cuda oom' in str(exc).lower()

def encode_resilient(encoder, texts:list[str], role:str, batch_size:int, desc: str = '') -> tuple[np.ndarray,int,int]:
    values=[]; tokens=0; i=0; bs=max(1,int(batch_size)); min_bs=1
    bar = None
    if desc:
        try:
            from tqdm.auto import tqdm
            import os
            bar=tqdm(total=len(texts), desc=desc, unit='row', dynamic_ncols=True,
                     disable=os.environ.get('EMBEDFLOW_NO_PROGRESS','').lower() in {'1','true','yes'})
        except Exception:
            bar=None
    try:
        while i<len(texts):
            take=min(bs,len(texts)-i)
            try:
                v,t=encoder.encode(texts[i:i+take],role); values.append(v);tokens+=t;i+=take
                if bar is not None: bar.update(take)
            except Exception as e:
                if not is_oom(e) or bs<=min_bs: raise
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception: pass
                gc.collect(); bs=max(min_bs,bs//2)
    finally:
        if bar is not None: bar.close()
    return (np.concatenate(values) if values else np.empty((0,getattr(encoder,'dimension',0)),dtype='float32')),tokens,bs

def encode_pool(encoder, ids:list[str], texts:list[str], out:Path, dimension:int, shard_rows:int, role:str, metadata:dict, resume=True, batch_size:int=32):
    out.mkdir(parents=True,exist_ok=True); total=0; tokens=0
    for start in progress(range(0,len(ids),shard_rows), desc=f'{metadata.get("model_key", "model")} {role}', unit='shard', total=(len(ids)+shard_rows-1)//shard_rows):
        idx=start//shard_rows; end=min(len(ids),start+shard_rows)
        if resume and valid_shard(out,idx,dimension): total+=end-start; continue
        batch=ids[start:end]; vals,tok,used_bs=encode_resilient(encoder,texts[start:end],role,batch_size); tokens+=tok
        write_shard(out,idx,batch,vals,dict(metadata,start=start,end=end,tokens=tok,role=role,batch_size=used_bs))
        total+=len(batch)
    return {'rows':total,'tokens':tokens,'shards':(len(ids)+shard_rows-1)//shard_rows}

def encode_pool_cached(encoder, ids:list[str], texts:list[str], out:Path, dimension:int, shard_rows:int, role:str, metadata:dict, cache:dict[str,np.ndarray], resume=True, batch_size:int=32):
    """Encode a canonical row order while reusing already-produced vectors."""
    out.mkdir(parents=True,exist_ok=True); total=0; tokens=0
    for start in progress(range(0,len(ids),shard_rows), desc=f'{metadata.get("model_key", "model")} {role} (cached)', unit='shard', total=(len(ids)+shard_rows-1)//shard_rows):
        idx=start//shard_rows; end=min(len(ids),start+shard_rows)
        if resume and valid_shard(out,idx,dimension): total+=end-start; continue
        chunk_ids=ids[start:end]; missing=[i for i,x in enumerate(chunk_ids) if x not in cache]
        vals=np.empty((len(chunk_ids),dimension),dtype='float32')
        for i,x in enumerate(chunk_ids):
            if x in cache: vals[i]=cache[x]
        if missing:
            fresh,tok,used_bs=encode_resilient(encoder,[texts[start+i] for i in missing],role,batch_size); tokens+=tok
            for j,i in enumerate(missing): vals[i]=fresh[j]
        write_shard(out,idx,chunk_ids,vals,dict(metadata,start=start,end=end,tokens=tokens,role=role,batch_size=used_bs if missing else batch_size)); total+=len(chunk_ids)
    return {'rows':total,'tokens':tokens,'shards':(len(ids)+shard_rows-1)//shard_rows}
