"""Cache only official training images, decoded with the unchanged TA dataset."""
import json
import os
from pathlib import Path
import numpy as np
from dataset import PokemonDataset
from src.runtime import sha256, assert_protected

ROOT = Path(__file__).resolve().parent

def prepare():
    assert_protected(ROOT)
    data_root = ROOT/'data/pokemon-generation-one-22k'
    mapping = json.loads((data_root/'category_to_id.json').read_text())
    train = PokemonDataset(data_root, data_root/'train_split.txt', return_category=True, category_to_id=mapping)
    val_paths = {x.strip() for x in (data_root/'val_split.txt').read_text().splitlines() if x.strip()}
    assert len(train)==17079 and len(val_paths)==3020 and set(train.paths).isdisjoint(val_paths)
    cache = ROOT/'cache';cache.mkdir(exist_ok=True)
    identity = {'train_manifest':sha256(data_root/'train_split.txt'), 'val_manifest':sha256(data_root/'val_split.txt'),
                'mapping':sha256(data_root/'category_to_id.json'), 'dataset':sha256(ROOT/'dataset.py'),
                'decoder':sha256(ROOT/'download_dataset.py')}
    meta = cache/'decoded_dataset.json'
    if meta.exists():
        existing = json.loads(meta.read_text());assert existing['identity']==identity
        assert sha256(cache/'train.npy')==existing['cache_sha256']
        assert sha256(cache/'labels.npy')==existing['labels_sha256']
        expected=json.loads((ROOT/'reference/cache_identity.json').read_text())
        for key in ['identity','cache_sha256','labels_sha256']:
            assert existing[key] == expected[key], f'Cached {key} differs from the recorded experiment'
        return existing
    temp = cache/'train.npy.tmp'
    array = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(train),3,64,64))
    for i in range(len(train)):
        image,label=train[i];array[i]=image.numpy()
        if i%2000==0: print('DECODED',i,flush=True)
    array.flush();del array;os.replace(temp,cache/'train.npy')
    labels = np.array(train.category_ids, dtype=np.int64)
    np.save(cache/'labels.npy',labels)
    rng=np.random.default_rng(20260923)
    diagnostic_indices=[]
    for c in range(151):
        candidates=np.flatnonzero(labels==c);assert len(candidates)>=20
        diagnostic_indices.extend(sorted(rng.choice(candidates,20,replace=False).tolist()))
    expected=json.loads((ROOT/'reference/cache_identity.json').read_text())
    assert identity == expected['identity'], 'Dataset/source identity differs from the recorded experiment'
    assert sha256(cache/'train.npy') == expected['cache_sha256'], 'Decoded images differ from the recorded experiment; check dataset files and pinned dependencies'
    assert sha256(cache/'labels.npy') == expected['labels_sha256'], 'Class labels differ from the recorded experiment'
    info={'identity':identity,'cache_sha256':sha256(cache/'train.npy'),'labels_sha256':sha256(cache/'labels.npy'),
          'train_count':len(train),'validation_count':len(val_paths),'diagnostic_indices':diagnostic_indices,
          'diagnostic_seed':20260923,'diagnostic_count':len(diagnostic_indices),
          'diagnostic_reference':'TRAINING reference; seen by warm start and fine-tuning, not a held-out FID',
          'official_validation_used_for_training':False}
    meta.write_text(json.dumps(info,indent=2)+'\n')
    print('DATA_READY',json.dumps({k:v for k,v in info.items() if k!='diagnostic_indices'}),flush=True)
    return info

if __name__=='__main__': prepare()
