"""Stateful shuffled official-train sampling, with no validation access."""
import json
from pathlib import Path
import numpy as np
import torch

class TrainingStream:
    def __init__(self, root, seed, microbatch):
        root=Path(root)
        self.images=np.load(root/'cache/train.npy',mmap_mode='r')
        self.labels=np.load(root/'cache/labels.npy')
        self.info=json.loads((root/'cache/dataset.json').read_text())
        split=json.loads((root/'cache/split.json').read_text())
        self.indices=torch.tensor(split['optimization_indices'],dtype=torch.long)
        self.generator=torch.Generator().manual_seed(seed)
        self.microbatch=microbatch
        self.order=torch.empty(0,dtype=torch.long)
        self.position=0
        self.epochs=0

    def draw(self, count):
        chunks=[]
        while count:
            if self.position == len(self.order):
                # The supplied DataLoader drops the incomplete final microbatch.
                limit=len(self.indices)//self.microbatch*self.microbatch
                self.order=self.indices[torch.randperm(len(self.indices),generator=self.generator)[:limit]]
                self.position=0;self.epochs+=1
            take=min(count,len(self.order)-self.position)
            chunks.append(self.order[self.position:self.position+take]);self.position+=take;count-=take
        return torch.cat(chunks).numpy()

    def batch(self, indices, device):
        return (torch.from_numpy(np.array(self.images[indices],copy=True)).to(device),
                torch.from_numpy(self.labels[indices].copy()).to(device))

    def state_dict(self):
        return {'rng':self.generator.get_state(),'order':self.order,'position':self.position,'epochs':self.epochs}

    def load_state_dict(self,state):
        self.generator.set_state(state['rng']);self.order=state['order'];self.position=state['position'];self.epochs=state['epochs']
