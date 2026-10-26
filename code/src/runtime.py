"""Training-only helpers: deterministic RNG state, atomic checkpoints, precision."""
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'])


@contextlib.contextmanager
def isolated_seed(seed):
    state = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


@contextlib.contextmanager
def evaluator_precision():
    precision = torch.get_float32_matmul_precision()
    cuda_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32; torch.backends.cudnn.allow_tf32 = cudnn_tf32


def atomic_save(payload, path, previous=None):
    """Commit a durable checkpoint; optionally retain the prior complete file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    previous = Path(previous) if previous is not None else None
    previous_temp = previous.with_name(previous.name+'.tmp') if previous else None
    try:
        with temp.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        if previous is not None and path.exists():
            if previous.parent != path.parent or previous == path:
                raise ValueError('Previous checkpoint must be a different file in the same directory')
            previous_temp.unlink(missing_ok=True)
            os.link(path, previous_temp)
            os.replace(previous_temp, previous)
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temp.unlink(missing_ok=True)
        if previous_temp is not None: previous_temp.unlink(missing_ok=True)


def cpu_state(model):
    return {key:value.detach().cpu() for key,value in model.state_dict().items()}


def append_json(path, row):
    with Path(path).open('a') as stream: stream.write(json.dumps(row)+'\n')


def assert_protected(root):
    root = Path(root); provenance = json.loads((root/'provenance.json').read_text())
    for relative, expected in {**provenance['protected_sha256'],
                              **{'src/original_meanflow/'+k:v for k,v in provenance['copied_backbone_sha256'].items()}}.items():
        if sha256(root/relative) != expected: raise ValueError(f'Protected source changed: {relative}')
