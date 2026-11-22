"""Portable numeric policy checkpoints: atomic NPZ writes; no pickling."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np
from tron_ai.engine import TronConfig
from tron_ai.observations import observation_shape

VERSION = 1

def save(path: Path, weights: dict[str,np.ndarray], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays = {k:np.asarray(v) for k,v in weights.items()}
    if not arrays or '__metadata__' in arrays or any(v.dtype.kind not in 'buifc' or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('weights must be finite numeric arrays, with nonreserved names')
    payload = json.dumps({**metadata,'format_version':VERSION},allow_nan=False,sort_keys=True)
    fd, name = tempfile.mkstemp(dir=path.parent,suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,__metadata__=np.array(payload),**arrays)
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def load(path: Path, *, config: TronConfig | None = None, algorithm: str | None = None):
    with np.load(path,allow_pickle=False) as f:
        meta = json.loads(f['__metadata__'].item())
        weights = {k:f[k].copy() for k in f.files if k!='__metadata__'}
    if meta.get('format_version') != VERSION:
        raise ValueError('unsupported checkpoint version')
    if not weights or any(v.dtype.kind not in 'buifc' or not np.isfinite(v).all() for v in weights.values()):
        raise ValueError('invalid weights')
    if algorithm is not None and meta['algorithm'] != algorithm:
        raise ValueError('algorithm mismatch')
    if config is not None:
        from dataclasses import asdict
        if meta['board'] != asdict(config) or meta['observation_shape'] != list(observation_shape(config)):
            raise ValueError('board or observation mismatch')
    return weights,meta
