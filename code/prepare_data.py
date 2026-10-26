"""Official-train optimization with explicitly configured reporting references."""
import hashlib,json
from pathlib import Path
import numpy as np
from src.runtime import sha256,assert_protected
ROOT=Path(__file__).resolve().parent

def prepare(dev_per_class=0,seed=20260923,fid_reference="official_validation"):
    assert_protected(ROOT);cache=ROOT/'cache';data=ROOT/'data/pokemon-generation-one-22k'
    if fid_reference not in ['official_validation','train_split']:raise ValueError('Unknown FID reference')
    if fid_reference=='official_validation' and dev_per_class:raise ValueError('Official protocol uses all training images')
    from build_cache import prepare as prepare_cache
    source=prepare_cache()
    assert sha256(cache/'train.npy')==source['cache_sha256']
    assert sha256(cache/'labels.npy')==source['labels_sha256']
    old=cache/'dataset.json'
    if old.exists():
        info=json.loads(old.read_text());assert info['dev_per_class']==dev_per_class and info['split_seed']==seed and info['fid_reference']==fid_reference,'Existing split differs; do not silently change it'
        assert info['identity']==source['identity']
        assert sha256(cache/'split.json')==info['split_sha256']
        return info
    labels=np.load(cache/'labels.npy');images=np.load(cache/'train.npy',mmap_mode='r')
    paths=[x.strip() for x in (data/'train_split.txt').read_text().splitlines() if x.strip()]
    val_paths={x.strip() for x in (data/'val_split.txt').read_text().splitlines() if x.strip()}
    assert len(labels)==len(paths)==17079 and set(paths).isdisjoint(val_paths)
    rng=np.random.default_rng(seed)
    if fid_reference=='official_validation':
        optimization=list(range(len(images)));reference=[];kind='official_validation';excluded_duplicates=0
    elif dev_per_class:
        # Recompute pixel hashes of the exact cached TA RGB data. Group duplicates
        # so a held-out reference has no identical preprocessed copy in training.
        hashes=[hashlib.sha256((((np.asarray(x)+1)/2).clip(0,1)*255).astype(np.uint8).transpose(1,2,0).tobytes()).hexdigest() for x in images]
        from torchvision.transforms.functional import to_pil_image
        import torch
        for i in rng.choice(len(images),32,replace=False):
            pixels=np.asarray(to_pil_image(((torch.from_numpy(np.array(images[i],copy=True))+1)/2).clamp(0,1)).convert('RGB'))
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==hashes[i]
        selected=[];dev_hashes=set()
        for c in range(151):
            candidates=rng.permutation(np.flatnonzero(labels==c));chosen=[]
            for i in candidates:
                if hashes[i] not in dev_hashes:
                    chosen.append(int(i));dev_hashes.add(hashes[i])
                if len(chosen)==dev_per_class:break
            assert len(chosen)==dev_per_class
            selected.extend(sorted(chosen))
        optimization=[i for i,h in enumerate(hashes) if h not in dev_hashes]
        reference=selected;kind='development';excluded_duplicates=17079-len(optimization)-len(reference)
        assert not ({hashes[i] for i in optimization}&{hashes[i] for i in reference})
    else:
        optimization=list(range(len(images)));reference=source['diagnostic_indices'];kind='training';excluded_duplicates=0
    assert len(reference)==(0 if kind=='official_validation' else 151*(dev_per_class or 20))
    if dev_per_class:assert set(optimization).isdisjoint(reference)
    split={'seed':seed,'dev_per_class':dev_per_class,'fid_reference':fid_reference,'reference_kind':kind,'optimization_indices':optimization,
           'reference_indices':reference,'duplicate_training_entries_excluded':excluded_duplicates}
    (cache/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    (cache/'optimization_manifest.txt').write_text('\n'.join(paths[i] for i in optimization)+'\n')
    (cache/'reference_manifest.txt').write_text((data/'val_split.txt').read_text() if kind=='official_validation' else '\n'.join(paths[i] for i in reference)+'\n')
    info={k:source[k] for k in ['identity','cache_sha256','labels_sha256']}
    info.update(split_seed=seed,dev_per_class=dev_per_class,fid_reference=fid_reference,optimization_count=len(optimization),reference_count=3020 if kind=='official_validation' else len(reference),
        reference_indices=reference,reference_kind=kind,split_sha256=sha256(cache/'split.json'),
        optimization_manifest_sha256=sha256(cache/'optimization_manifest.txt'),reference_manifest_sha256=sha256(cache/'reference_manifest.txt'),
        optimization_class_counts=np.bincount(labels[optimization],minlength=151).tolist(),
        duplicate_training_entries_excluded=excluded_duplicates,official_validation_used_for_training_or_tuning=False)
    old.write_text(json.dumps(info,indent=2)+'\n')
    print('DATA_READY '+json.dumps({k:v for k,v in info.items() if k not in ['reference_indices','optimization_class_counts']}),flush=True)
    return info

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dev-per-class',type=int,default=0);p.add_argument('--seed',type=int,default=20260923);p.add_argument('--fid-reference',choices=['official_validation','train_split'],default='official_validation');a=p.parse_args();prepare(a.dev_per_class,a.seed,a.fid_reference)
