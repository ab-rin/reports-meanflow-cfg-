"""Original MeanFlow with built-in CFG: arXiv:2505.13447, equations 18–19.

Guidance is learned during training. The conditional sampler needs one model
execution per integration step, with no inference-time unconditional pass.
The null condition is an all-zero 151-dimensional class vector, not class zero.
"""
from functools import partial
import torch
from torch.nn import functional as F
from .original_meanflow.recipe import OriginalMeanFlowUNet, sample_times_v0
from .original_meanflow.unet import SongUNet
from .imf_objective import prediction_and_jvp


class CFGMeanFlowUNet(OriginalMeanFlowUNet):
    def forward(self, x, timestep, *, r, category, aug_cond=None):
        # -1 is training-only null conditioning. Sampling validates 0..150.
        labels=F.one_hot(category.clamp_min(0),num_classes=self.num_classes).to(x.dtype)
        labels=labels*(category>=0).to(x.dtype)[:,None]
        return SongUNet.forward(self,x,(timestep,timestep-r),
                                class_labels=labels,aug_cond=aug_cond)


def guided_loss_core(model,images,category,aug_cond,*,noise,r,t,velocity,
                     dropped,fused_jvp=False):
    z=(1-t[:,None,None,None])*images+t[:,None,None,None]*noise
    field=lambda z,r,t:model(z,t,r=r,category=category,aug_cond=aug_cond)
    if fused_jvp:
        prediction,derivative=torch.func.jvp(field,(z,r,t),
            (velocity,torch.zeros_like(r),torch.ones_like(t)))
    else:
        prediction,derivative=prediction_and_jvp(field,z,r,t,velocity)
    # BOTH target and JVP use the guided velocity. This is original MF, not iMF.
    correction=(t-r)[:,None,None,None]*derivative.detach()
    target=(velocity-correction).detach()
    squared=(prediction-target).float().square()
    squared_l2=squared.flatten(1).sum(1)
    loss=(squared_l2/(squared_l2.detach()+1e-3).pow(.75)).mean()
    with torch.no_grad():
        mse=squared.flatten(1).mean(1);fm=r==t;raw=noise-images
        stats=dict(loss=loss.detach(),mse=mse.mean(),squared_l2=squared_l2.mean(),
            fm_error_sum=(mse*fm).sum(),fm_count=fm.float().sum(),
            mf_error_sum=(mse*~fm).sum(),mf_count=(~fm).float().sum(),
            null_fraction=dropped.float().mean(),
            guided_velocity_ms=velocity.square().mean(),raw_velocity_ms=raw.square().mean(),
            guidance_delta_ms=(velocity-raw).square().mean(),
            correction_ms=correction.square().mean(),prediction_ms=prediction.detach().square().mean())
    return loss,stats


def make_loss_function(*,guidance_scale=2.,label_dropout=.1,compile_enabled=False):
    if guidance_scale<1:raise ValueError('guidance_scale must be >= 1')
    if not 0<=label_dropout<=1:raise ValueError('label_dropout must be in [0,1]')
    core=partial(guided_loss_core,fused_jvp=compile_enabled)
    if compile_enabled:
        core=torch.compile(core,fullgraph=True,options={'fallback_random':True})

    def loss_function(model,images,category,aug_cond=None,*,noise=None,r=None,t=None,drop_mask=None):
        if (r is None)!=(t is None):raise ValueError('Supply both r and t, or neither')
        if noise is None:noise=torch.randn_like(images)
        if r is None:r,t=sample_times_v0(len(images),images.device)
        if drop_mask is None:drop_mask=torch.rand(len(images),device=images.device)<label_dropout
        dropped_category=torch.where(drop_mask,torch.full_like(category,-1),category)
        with torch.no_grad():
            raw=noise-images
            if guidance_scale==1:
                velocity=raw
            else:
                z=(1-t[:,None,None,None])*images+t[:,None,None,None]*noise
                devices=[images.device.index or 0] if images.is_cuda else []
                # Official MeanFlow guidance_fn calls v_fn(..., train=False).
                # Keep dropout off for this detached target, then restore training.
                training=model.training
                try:
                    model.eval()
                    with torch.random.fork_rng(devices=devices):
                        unconditional=model(z,t,r=t,category=torch.full_like(category,-1),aug_cond=aug_cond)
                finally:
                    model.train(training)
                guided=guidance_scale*raw+(1-guidance_scale)*unconditional
                # Dropped examples learn the true unconditional field.
                velocity=torch.where(drop_mask[:,None,None,None],raw,guided).detach()
        return core(model,images,dropped_category,aug_cond,noise=noise,r=r,t=t,
                    velocity=velocity,dropped=drop_mask)
    return loss_function
