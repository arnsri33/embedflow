from __future__ import annotations
import hashlib, json, os, platform, subprocess, time
from pathlib import Path
from typing import Any

def progress(iterable, *, desc: str = '', total=None, unit: str = 'it', leave: bool = True):
    """Return a tqdm iterator with optional nested-bar cleanup.

    Set ``EMBEDFLOW_NO_PROGRESS=1`` for quiet runs.  ``leave=False`` is useful
    for fine-grained query bars nested inside a shard bar.
    """
    try:
        from tqdm.auto import tqdm
        disabled = os.environ.get('EMBEDFLOW_NO_PROGRESS', '').lower() in {'1', 'true', 'yes'}
        return tqdm(iterable, desc=desc, total=total, unit=unit,
                    disable=disabled, dynamic_ncols=True, leave=leave)
    except Exception:
        return iterable

def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(block), b''): h.update(b)
    return h.hexdigest()

def stable_hash(value: str, seed: int = 0) -> str:
    return hashlib.sha256(f'{seed}\0{value}'.encode()).hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)

def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text())

def command_output(args: list[str]) -> str:
    try: return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e: return f'command failed: {e}'

def environment_snapshot() -> dict[str, Any]:
    import sys
    out = {'python': sys.version, 'platform': platform.platform(), 'hostname': platform.node(), 'cpu': platform.processor()}
    try:
        import torch
        out.update({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_count': torch.cuda.device_count()})
    except Exception as e: out['torch_error'] = repr(e)
    out['nvidia_smi'] = command_output(['nvidia-smi'])
    return out

def now() -> str: return time.strftime('%Y%m%d-%H%M%S')
