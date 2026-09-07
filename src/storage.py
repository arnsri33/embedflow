from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from .utils import sha256_file, atomic_json

def write_shard(root: Path, index: int, ids: list[str], vectors: np.ndarray, metadata: dict) -> dict:
    root.mkdir(parents=True,exist_ok=True); arr=np.asarray(vectors)
    if arr.ndim!=2 or len(ids)!=arr.shape[0] or not np.isfinite(arr).all(): raise ValueError('invalid shard')
    tmp=root/f'.shard-{index:06d}.npy.tmp'; p=root/f'shard-{index:06d}.npy';
    with tmp.open('wb') as f: np.save(f,arr.astype(np.float16),allow_pickle=False)
    tmp.replace(p)
    ip=root/f'shard-{index:06d}.ids.json'; ip.write_text(json.dumps(ids,ensure_ascii=False));
    meta=dict(metadata,shard=index,rows=len(ids),dimension=int(arr.shape[1]),dtype='float16',array_sha256=sha256_file(p),ids_sha256=sha256_file(ip))
    atomic_json(root/f'shard-{index:06d}.json',meta); (root/f'shard-{index:06d}.done').write_text(meta['array_sha256']+'\n'); return meta

def valid_shard(root: Path,index:int,dimension:int|None=None)->bool:
    p=root/f'shard-{index:06d}.npy'; m=root/f'shard-{index:06d}.json'; d=root/f'shard-{index:06d}.done'
    if not (p.exists() and m.exists() and d.exists()): return False
    try:
        meta=json.loads(m.read_text()); a=np.load(p,mmap_mode='r')
        return a.ndim==2 and (dimension is None or a.shape[1]==dimension) and meta.get('array_sha256')==sha256_file(p) and d.read_text().strip()==meta['array_sha256']
    except Exception:return False

def load_shards(root: Path, dimension:int|None=None)->tuple[list[str],np.ndarray]:
    arrays=[];ids=[]
    for p in sorted(root.glob('shard-*.npy')):
        i=int(p.stem.split('-')[1]);
        if not valid_shard(root,i,dimension): raise ValueError(f'invalid shard {p}')
        arrays.append(np.load(p)); ids.extend(json.loads((root/f'shard-{i:06d}.ids.json').read_text()))
    if not arrays: raise ValueError(f'no shards in {root}')
    return ids,np.concatenate(arrays).astype(np.float32)

def iter_shards(root: Path, dimension: int|None=None):
    """Yield (ids, mmap vectors, metadata) without concatenating the corpus."""
    for p in sorted(root.glob('shard-*.npy')):
        i=int(p.stem.split('-')[1])
        if not valid_shard(root,i,dimension): raise ValueError(f'invalid shard {p}')
        ids=json.loads((root/f'shard-{i:06d}.ids.json').read_text())
        yield ids,np.load(p,mmap_mode='r'),json.loads((root/f'shard-{i:06d}.json').read_text())

def vector_lookup(root: Path, ids: list[str], dimension: int|None=None) -> np.ndarray:
    """Read only requested rows, grouped by shard; output follows ``ids``."""
    wanted={str(x):i for i,x in enumerate(ids)}; out=None; found=set()
    for shard_ids, arr, _ in iter_shards(root,dimension):
        local=[(j,wanted[x]) for j,x in enumerate(shard_ids) if x in wanted]
        if local:
            if out is None: out=np.empty((len(ids),arr.shape[1]),dtype='float32')
            for j,pos in local: out[pos]=np.asarray(arr[j],dtype='float32'); found.add(shard_ids[j])
    if out is None or len(found)!=len(wanted): raise KeyError(f'missing {len(wanted)-len(found)} vector IDs')
    return out
