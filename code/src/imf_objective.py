"""iMF Algorithm 1 (arXiv:2512.02012), boundary-velocity variant, no CFG.

Only the JVP spatial tangent changes relative to the supplied MF objective.
The regression target remains noise - data. The JVP result is stop-gradient;
there is no auxiliary velocity head, auxiliary loss, or EMA teacher.
"""
import torch
from .original_meanflow.recipe import sample_times_v0


def prediction_and_jvp(field, z, r, t, velocity):
    """Compute detached derivative first to avoid overlapping its peak with the reverse graph."""
    devices = [z.device.index if z.device.index is not None else torch.cuda.current_device()] if z.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        _, derivative = torch.func.jvp(field, (z, r, t),
            (velocity, torch.zeros_like(r), torch.ones_like(t)))
    # fork_rng restores the exact pre-JVP state, including CUDA dropout draws.
    prediction = field(z, r, t)
    return prediction, derivative.detach()


def meanflow_loss(model, images, category, aug_cond=None, *, objective='imf', noise=None, r=None, t=None,
                  fused_jvp=False, tangent_velocity=None):
    if objective not in ('imf', 'mf'):
        raise ValueError('objective must be imf or mf')
    if (r is None) != (t is None):
        raise ValueError('Supply both r and t, or neither')
    if noise is None:
        noise = torch.randn_like(images)
    if r is None:
        r, t = sample_times_v0(len(images), images.device)
    interval = (t-r)[:, None, None, None]
    z = (1-t[:, None, None, None])*images + t[:, None, None, None]*noise
    conditional_velocity = noise-images

    def field(z, r, t):
        # Differentiate actual r,t inputs, including the backbone's t-r embedding.
        return model(z, t, r=r, category=category, aug_cond=aug_cond)

    if objective == 'imf' and tangent_velocity is None:
        devices = [z.device.index or 0] if z.is_cuda else []
        # Reuse the same dropout realization for v(z,t)=u(z,t,t), u, and its JVP.
        # No-grad also avoids retaining an unused parameter-gradient graph for v.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            tangent_velocity = field(z, t, t).detach()
    elif objective == 'mf':
        tangent_velocity = conditional_velocity
    if fused_jvp:
        prediction, derivative = torch.func.jvp(
            field, (z, r, t), (tangent_velocity, torch.zeros_like(r), torch.ones_like(t)))
    else:
        prediction, derivative = prediction_and_jvp(field, z, r, t, tangent_velocity)
    correction = interval * derivative.detach()
    compound_velocity = prediction + correction
    error = compound_velocity - conditional_velocity
    squared_error = error.float().square()
    squared_l2 = squared_error.flatten(1).sum(1)
    # Keep the supplied recipe's pixel-sum adaptive L2 weighting unchanged.
    loss = (squared_l2 / (squared_l2.detach()+1e-3).pow(.75)).mean()
    with torch.no_grad():
        mse = squared_error.flatten(1).mean(1)
        fm = r == t
        stats = dict(loss=loss.detach(), mse=mse.mean(), squared_l2=squared_l2.mean(),
                     fm_error_sum=(mse*fm).sum(), fm_count=fm.float().sum(),
                     mf_error_sum=(mse*~fm).sum(), mf_count=(~fm).float().sum(),
                     tangent_velocity_ms=tangent_velocity.square().mean(),
                     conditional_velocity_ms=conditional_velocity.square().mean(),
                     tangent_vs_conditional_mse=(tangent_velocity-conditional_velocity).square().mean(),
                     correction_ms=correction.square().mean(), prediction_ms=prediction.detach().square().mean())
    return loss, stats


def make_loss_function(objective, *, compile_enabled=False):
    """Compile the fused prediction/JVP while keeping RNG replay outside Dynamo.

    Native random operations preserve the dropout realization between iMF's
    no-grad boundary velocity and the fused JVP. Only the loss function is
    compiled; checkpoint keys and the sampling model stay unchanged.
    """
    from functools import partial
    if not compile_enabled:
        return partial(meanflow_loss, objective=objective)
    core = torch.compile(partial(meanflow_loss, objective=objective, fused_jvp=True),
                         fullgraph=True, options={"fallback_random": True})

    def compiled_loss(model, images, category, aug_cond=None, *, noise=None, r=None, t=None):
        if (r is None) != (t is None):
            raise ValueError('Supply both r and t, or neither')
        if noise is None:
            noise = torch.randn_like(images)
        if r is None:
            r, t = sample_times_v0(len(images), images.device)
        velocity = None
        if objective == 'imf':
            with torch.no_grad():
                z = (1-t[:, None, None, None])*images + t[:, None, None, None]*noise
                devices = [images.device.index or 0] if images.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    velocity = model(z, t, r=t, category=category, aug_cond=aug_cond).detach()
        return core(model, images, category, aug_cond, noise=noise, r=r, t=t,
                    tangent_velocity=velocity)

    return compiled_loss
