"""Reporting-only FID with explicit reference identity and no training feedback."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from model import ModelOneNFE,ModelFewNFE
from evaluate import prepare_reference_set,generate_samples,compute_fid
from .runtime import isolated_seed,evaluator_precision,append_json,sha256


def evaluate_reference(ema,run,root,step,seed=1234,batch_size=32):
    with isolated_seed(seed), evaluator_precision():
        return _evaluate_reference(ema,run,root,step,seed,batch_size)


def _evaluate_reference(ema,run,root,step,seed=1234,batch_size=32):
    run,root=Path(run),Path(root)
    info=json.loads((root/'cache/dataset.json').read_text())
    cache=np.load(root/'cache/train.npy',mmap_mode='r')
    reference=run/(info['reference_kind']+'_reference_png')
    if not reference.exists():
        if info['reference_kind']=='official_validation':
            from dataset import PokemonDataset
            data_root=root/'data/pokemon-generation-one-22k'
            dataset=PokemonDataset(data_root,data_root/'val_split.txt')
            assert len(dataset)==3020
            data=SimpleNamespace(val_dataloader=lambda:DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=4))
        else:
            images=torch.from_numpy(np.array(cache[info['reference_indices']],copy=True))
            data=SimpleNamespace(val_dataloader=lambda:DataLoader(TensorDataset(images),batch_size=batch_size,shuffle=False))
        prepare_reference_set(data,reference)
    assert len(list(reference.glob('*.png')))==info['reference_count']
    checkpoint=run/f'checkpoints/ema_{step:07d}.ckpt'
    categories=torch.arange(151,dtype=torch.long).repeat_interleave(20)
    # Lightweight wrappers share this training-only EMA backbone, without a second copy.
    results=[]
    for cls in (ModelOneNFE,ModelFewNFE):
        wrapper=cls.__new__(cls);torch.nn.Module.__init__(wrapper);wrapper.backbone=ema.backbone
        directory=run/f'diagnostics/step_{step:07d}/nfe_{cls.sampling_steps}'
        directory.mkdir(parents=True,exist_ok=False)
        with isolated_seed(seed),evaluator_precision():
            generated=generate_samples(wrapper,directory/'generated',categories,batch_size=batch_size,device=next(ema.parameters()).device)
            score=compute_fid(generated,reference,device=next(ema.parameters()).device,batch_size=batch_size)
        row={'step':step,'fid':score,'nfe':cls.sampling_steps,'seed':seed,'generated_count':3020,'reference_count':info['reference_count'],
             'metric':{'official_validation':'COURSE-VALIDATION FID; local reporting score, not an official TA measurement','development':'DEVELOPMENT FID; held out from official train; not TA FID','training':'TRAINING-reference FID; seen in training; not held-out or TA FID'}[info['reference_kind']],
             'reference_kind':info['reference_kind'],'reference_manifest_sha256':info['reference_manifest_sha256'],'split_sha256':info['split_sha256'],
             'official_validation_used':info['reference_kind']=='official_validation','used_for_training_or_automatic_selection':False,'precision':'FP32, TF32 disabled','batch_size':batch_size,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
             'reference_indices':info['reference_indices'],'reference_train_manifest_sha256':info['identity']['train_manifest']}
        (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n')
        append_json(run/(info['reference_kind']+'_fid.jsonl'),row);results.append(row)
    return results
