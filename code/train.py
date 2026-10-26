"""Train class-conditional MeanFlow with built-in CFG from random initialization.

All trainable modules fit on each device independently. Extra GPUs replicate the
model; they do not shard it to evade the course's single-GPU 20 GB constraint.
"""
import argparse
import copy
import json
import os
import signal
import shutil
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from model import ModelFewNFE
from src.cfg_meanflow import make_loss_function
from src.original_meanflow.recipe import augmentation_pipe,update_original_ema
from src.training_data import TrainingStream
from src.runtime import (sha256,seed_all,rng_state,restore_rng,isolated_seed,evaluator_precision,
                         atomic_save,cpu_state,append_json,assert_protected)

ROOT=Path(__file__).resolve().parent


def validate_resume_config(saved_config,config,*,allow_microbatch_change=False):
    """Allow runtime overrides and an explicit, fixed-effective-batch migration."""
    runtime={'output','resume','updates','max_hours','save_every','preview_every','log_every',
             'fid_every','fid_batch','publish','compile','verify_replicas','allow_microbatch_change',
             'optimizer_initialization'}  # Describes state origin, not optimizer hyperparameters.
    for key in ('effective_batch','world_size'):
        if saved_config.get(key)!=config.get(key):raise ValueError(f'Resume mismatch: {key}')
    if allow_microbatch_change:
        for label,options in [('checkpoint',saved_config),('requested',config)]:
            effective,world,micro=(options.get(key) for key in ('effective_batch','world_size','microbatch'))
            if not all(isinstance(value,int) and value>0 for value in (effective,world,micro)):
                raise ValueError(f'Invalid {label} batch configuration')
            if effective%(world*micro) or options.get('gradient_accumulation_per_gpu')!=effective//(world*micro):
                raise ValueError(f'Invalid {label} gradient_accumulation_per_gpu')
        runtime.update({'microbatch','gradient_accumulation_per_gpu'})
    for key,value in config.items():
        if key not in runtime and saved_config.get(key)!=value:raise ValueError(f'Resume mismatch: {key}')
    return {key:{'old':saved_config.get(key),'new':value}
            for key,value in config.items() if saved_config.get(key)!=value}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True,help='Total optimizer updates from random initialization')
    p.add_argument('--resume',type=Path)
    p.add_argument('--allow-microbatch-change',action='store_true',
                   help='Allow full-state resume with a different microbatch/accumulation count; effective batch and world size must stay fixed')
    p.add_argument('--guidance-scale',type=float,default=2.)
    p.add_argument('--label-dropout',type=float,default=.1)
    p.add_argument('--dev-per-class',type=int,default=0)
    p.add_argument('--fid-reference',choices=['official_validation','train_split'],default='official_validation')
    p.add_argument('--dev-seed',type=int,default=20260923)
    p.add_argument('--schedule',choices=['fastgen','original'],default='fastgen')
    p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--effective-batch',type=int,default=512)
    p.add_argument('--lr',type=float)
    p.add_argument('--warmup-steps',type=int)
    p.add_argument('--ema-decay',type=float)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fid-seed',type=int,default=1234)
    p.add_argument('--fid-every',type=int,default=0,help='Reporting FID interval using the explicitly configured reference; no automatic model selection')
    p.add_argument('--fid-batch',type=int,default=32)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--preview-every',type=int,default=100)
    p.add_argument('--log-every',type=int,default=1)
    p.add_argument('--max-hours',type=float,default=0)
    p.add_argument('--compile',action='store_true',help='Compile the fused prediction/JVP loss; native dropout RNG is preserved')
    p.add_argument('--verify-replicas',action='store_true',help='Check initial/final full-state equality across ranks')
    p.add_argument('--disable-tf32',action='store_true')
    p.add_argument('--device',default='cuda')
    p.add_argument('--publish',action='store_true',help='Export EMA to this branch checkpoints/one_nfe.ckpt and few_nfe.ckpt')
    a=p.parse_args()
    defaults={'fastgen':(3e-4,500,.999),'original':(6e-4,10000,.99995)}[a.schedule]
    for key,value in zip(['lr','warmup_steps','ema_decay'],defaults):
        if getattr(a,key) is None:setattr(a,key,value)
    for key in ['updates','microbatch','effective_batch','lr','save_every','preview_every','log_every','fid_batch']:
        if getattr(a,key)<=0:p.error(f'{key} must be positive')
    if a.fid_every<0 or a.warmup_steps<0 or a.max_hours<0 or not 0<=a.ema_decay<1:p.error('Invalid schedule')
    if a.guidance_scale<1 or not 0<=a.label_dropout<1 or not 0<=a.dev_per_class<=20:p.error('Invalid CFG or development split setting')
    return a


def main():
    args=parse_args();assert_protected(ROOT)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device('cuda',local) if world>1 else torch.device(args.device)
    if device.type!='cuda':raise ValueError('Training/memory verification requires CUDA; objective unit tests run on CPU')
    if device.index is None:device=torch.device('cuda',0)
    torch.cuda.set_device(device);torch.set_num_threads(4)
    if args.effective_batch%(world*args.microbatch):raise ValueError('Effective batch must be divisible by world_size * microbatch')
    control=None
    if world>1:
        dist.init_process_group('nccl',device_id=device,timeout=timedelta(hours=2))
        control=dist.new_group(backend='gloo',timeout=timedelta(hours=2))
    if rank==0:
        from prepare_data import prepare
        prepare(args.dev_per_class,args.dev_seed,args.fid_reference)
        if args.output.exists() and any(args.output.iterdir()) and args.resume is None:raise ValueError('Use a fresh output directory')
        args.output.mkdir(parents=True,exist_ok=True)
    if world>1:dist.barrier()
    seed_all(args.seed)  # Identical RANDOM model initialization on every replica.
    torch.set_float32_matmul_precision('highest' if args.disable_tf32 else 'high')
    torch.backends.cudnn.allow_tf32=not args.disable_tf32
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=ModelFewNFE(device=device).train()
    saved=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    source_step=0;done=0
    if saved is not None:
        assert saved['format']=='mf-cfg-training-v1' and saved['source_step']==0
        done=saved['updates'];model.load_state_dict(saved['model'],strict=True)
        if done>args.updates:raise ValueError('Resume checkpoint is past requested update budget')
    # Broadcast guarantees all replicas start from the same parameters/buffers.
    if world>1:
        for value in model.state_dict().values():dist.broadcast(value,src=0)
    seed_all(args.seed+rank)  # Independent rank-local augmentation/dropout/noise streams.
    assert model.num_parameters<100_000_000
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    stream=TrainingStream(ROOT,args.seed,32)  # Preserve source DataLoader drop_last grouping.
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(world_size=world,source_step=source_step,initial_checkpoint_sha256=None,
                  parameters=model.num_parameters,cfg='MeanFlow built-in guidance, equations 18-19',no_auxiliary_head=True,
                  objective='original MeanFlow with guided target AND guided JVP tangent',
                  optimizer_initialization=('Adam state restored from full-state checkpoint' if args.resume else 'fresh Adam; randomly initialized model'),data_identity=stream.info['identity'],
                  checkpoint_selection='predeclared update budget; reporting FID only; no automatic model selection',
                  optimization_count=stream.info['optimization_count'],reference_count=stream.info['reference_count'],
                  reference_kind=stream.info['reference_kind'],split_sha256=stream.info['split_sha256'],
                  cache_sha256=stream.info['cache_sha256'],labels_sha256=stream.info['labels_sha256'],
                  gradient_accumulation_per_gpu=args.effective_batch//(world*args.microbatch))
    if args.resume:
        resume_changes=validate_resume_config(saved['config'],config,allow_microbatch_change=args.allow_microbatch_change)
        ema.load_state_dict(saved['ema'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['stream']);restore_rng(saved['rank_rng'][rank])
        if rank==0:
            append_json(args.output/'resume_changes.jsonl',dict(
                checkpoint=str(args.resume),resumed_at_step=done,changes=resume_changes,
                old_config=saved['config'],new_config=config,optimizer_state_restored=True,
                microbatch_changed=saved['config']['microbatch']!=args.microbatch,
                note='Changing microbatch preserves effective batch, but changes random-number grouping and floating-point accumulation; continuation is not bitwise identical.'))
    sampled_hist=np.array(saved.get('sampled_class_histogram',[0]*151),dtype=np.int64) if args.resume else np.zeros(151,dtype=np.int64)
    prior_elapsed=saved.get('elapsed_seconds',0) if args.resume else 0
    prior_train=saved.get('training_seconds',0) if args.resume else 0
    del saved
    if rank==0:
        (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        source_files=[ROOT/'model.py',ROOT/'train.py',ROOT/'evaluate.py',ROOT/'dataset.py',ROOT/'download_dataset.py',ROOT/'prepare_data.py',ROOT/'provenance.json']+sorted((ROOT/'src').rglob('*.py'))
        source_hashes={}
        for path in source_files:
            relative=path.relative_to(ROOT);target=args.output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);source_hashes[str(relative)]=sha256(path)
        (args.output/'source_sha256.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
        print('CONFIG '+json.dumps(config),flush=True)
        print(('Resumed full training state.' if args.resume else 'Random initialization at step 0; no checkpoint weights loaded.')+' Built-in CFG sampling uses exactly 1 or 4 calls.',flush=True)
        initial_hash=__import__('hashlib').sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        (args.output/('resume_initialization.json' if args.resume else 'initialization.json')).write_text(json.dumps(dict(seed=args.seed,source_step=0,parameter_sha256=initial_hash,checkpoint_loaded=bool(args.resume),null_class=-1),indent=2)+'\n')
    def verify_replicas(stage):
        if not args.verify_replicas:return
        digest=__import__('hashlib').sha256()
        for value in model.state_dict().values():digest.update(value.detach().cpu().numpy().tobytes())
        record={'rank':rank,'hash':digest.hexdigest(),'sampler_epochs':stream.epochs,'sampler_position':stream.position}
        records=[None]*world
        if world>1:dist.all_gather_object(records,record,group=control)
        else:records=[record]
        assert len({r['hash'] for r in records})==1,records
        assert len({(r['sampler_epochs'],r['sampler_position']) for r in records})==1,records
        if rank==0:append_json(args.output/'replica_checks.jsonl',dict(stage=stage,step=done,passed=True,replicas=records))
    verify_replicas('initial')
    loss_function=make_loss_function(guidance_scale=args.guidance_scale,label_dropout=args.label_dropout,compile_enabled=args.compile)
    if rank==0 and args.compile:
        print('Compiled fused JVP enabled; first update includes compilation. Native dropout RNG retained.',flush=True)
    augment=augmentation_pipe()
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    started=time.monotonic();training_seconds=prior_train
    torch.cuda.reset_peak_memory_stats(device)

    def save():
        state=rng_state();states=[None]*world if rank==0 else None
        if world>1:dist.gather_object(state,states,dst=0,group=control)
        else:states=[state]
        if rank==0:
            step=source_step+done
            snapshot={'format':'mf-cfg-training-v1','source_step':source_step,'updates':done,'step':step,
                      'model':cpu_state(model),'ema':cpu_state(ema),'optimizer':optimizer.state_dict(),
                      'rank_rng':states,'stream':stream.state_dict(),'config':config,'sampled_class_histogram':sampled_hist.tolist(),
                      'elapsed_seconds':prior_elapsed+time.monotonic()-started,'training_seconds':training_seconds}
            atomic_save(snapshot,args.output/'training_latest.pt',previous=args.output/'training_previous.pt')
            exported={'state_dict':snapshot['ema'],'step':step,'source_step':source_step,'updates':done,'objective':'mf_cfg','guidance_scale':args.guidance_scale,'label_dropout':args.label_dropout}
            atomic_save(exported,args.output/'ema_latest.ckpt')
            if args.fid_every and done%args.fid_every==0:atomic_save(exported,args.output/f'checkpoints/ema_{step:07d}.ckpt')
            if args.publish:
                atomic_save(exported,ROOT/'checkpoints/one_nfe.ckpt');atomic_save(exported,ROOT/'checkpoints/few_nfe.ckpt')
        if world>1:dist.barrier()

    first_update=True
    status='complete'
    while done<args.updates:
        stop=stopping or (args.output/'STOP').exists() or (args.max_hours and time.monotonic()-started>=args.max_hours*3600)
        flag=torch.tensor(int(bool(stop)),device=device)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item():status='stopped';break
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        global_indices=stream.draw(args.effective_batch)
        sampled_hist+=np.bincount(stream.labels[global_indices],minlength=151)
        logical=global_indices.reshape(-1,world,args.microbatch)
        rounds=len(logical);totals={}
        for batch_indices in logical[:,rank]:
            images,category=stream.batch(batch_indices,device)
            with torch.no_grad():
                flip=torch.rand(len(images),device=device)<.5
                images=torch.where(flip[:,None,None,None],images.flip(-1),images)
                images,aug_cond=augment(images);images=images.contiguous()
            loss,stats=loss_function(model,images,category,aug_cond)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite objective')
            (loss/rounds).backward()
            for k,v in stats.items():totals[k]=totals.get(k,0)+v.detach()/rounds
            del images,category,loss,stats,aug_cond
        if world>1:
            flat=torch.cat([p.grad.flatten() for p in model.parameters()])
            dist.all_reduce(flat);flat.div_(world)
            offset=0
            for p in model.parameters():p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
            del flat
            keys=list(totals);values=torch.stack([totals[k] for k in keys]);dist.all_reduce(values);values.div_(world)
            totals=dict(zip(keys,values))
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        step=source_step+done
        fraction=min(1.,step/args.warmup_steps) if args.warmup_steps else 1.
        lr=1e-8+(args.lr-1e-8)*fraction
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();done+=1;step=source_step+done
        decay=min(args.ema_decay,(step+1)/(step+10)) if args.schedule=='fastgen' else args.ema_decay
        update_original_ema(ema,model,step,decay=decay,period=1 if args.schedule=='fastgen' else 16)
        torch.cuda.synchronize();duration=time.monotonic()-tick;training_seconds+=duration
        peak=torch.tensor(torch.cuda.max_memory_allocated(device),device=device,dtype=torch.int64)
        if world>1:dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if peak.item()>=20_000_000_000:raise RuntimeError('Peak training allocation exceeded 20 GB; reduce microbatch')
        if rank==0 and (first_update or done%args.log_every==0):
            metrics={k:float(v) for k,v in totals.items()}
            metrics.update(step=step,updates=done,source_step=source_step,lr=lr,gradient_norm=float(grad_norm),
                           peak_train_bytes=peak.item(),seconds_per_update=duration,images_per_second=args.effective_batch/duration,
                           elapsed_seconds=prior_elapsed+time.monotonic()-started,training_seconds=training_seconds,
                           compiled=args.compile,includes_compile_warmup=bool(first_update and args.compile))
            append_json(args.output/'metrics.jsonl',metrics);print('TRAIN '+json.dumps(metrics),flush=True)
        do_fid=args.fid_every>0 and done%args.fid_every==0
        if first_update or done%args.save_every==0 or done==args.updates or do_fid:
            save()
            if rank==0:append_json(args.output/'class_histograms.jsonl',dict(step=step,sampled_classes=sampled_hist.tolist(),images_seen=int(sampled_hist.sum()),sampler='uniform shuffled optimization images; empirical class prior'))
        if rank==0 and (first_update or done%args.preview_every==0):
            with isolated_seed(args.fid_seed),evaluator_precision():
                labels=torch.arange(16,device=device)*10
                images=ema.sample((16,3,64,64),device=device,category=labels)
                folder=args.output/'previews';folder.mkdir(exist_ok=True)
                save_image((images+1)/2,folder/f'ema_{step:07d}.png',nrow=4)
        if do_fid:
            if rank==0:
                from src.diagnostic_fid import evaluate_reference
                results=evaluate_reference(ema,args.output,ROOT,step,args.fid_seed,args.fid_batch)
                print('FID '+json.dumps([{k:v for k,v in row.items() if k!='reference_indices'} for row in results]),flush=True)
        if world>1:dist.barrier()
        # Evaluation peaks must not become the next iteration's training peak.
        torch.cuda.reset_peak_memory_stats(device)
        first_update=False
    save()
    verify_replicas('final')
    if rank==0:
        result=dict(status=status,step=source_step+done,updates=done,source_step=source_step,
                    elapsed_seconds=prior_elapsed+time.monotonic()-started,training_seconds=training_seconds)
        (args.output/'status.json').write_text(json.dumps(result,indent=2)+'\n');print('FINISHED '+json.dumps(result),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
