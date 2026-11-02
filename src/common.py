import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext
import torch

ROOT = Path(__file__).resolve().parent

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def source_id():
    return digest({p.name: file_hash(p) for p in sorted(ROOT.glob('*.py'))})

def autocast(device, config):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and config['precision'] == 'bf16' else nullcontext()

def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    return value

def frozen_teacher(config, device):
    from model import SwapPolicy
    assert file_hash(config['teacher_checkpoint'])==config['teacher_sha256'],'Frozen E006 teacher changed'
    saved=torch.load(config['teacher_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['step']==config['teacher_step']
    policy=SwapPolicy(config['side'],config['width'],config['layers'],config['head_width']).to(device)
    policy.load_state_dict(saved['model'],strict=True);policy.eval();policy.requires_grad_(False)
    return policy
