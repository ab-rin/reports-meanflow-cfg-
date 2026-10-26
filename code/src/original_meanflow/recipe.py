"""Original MeanFlow CIFAR-v0 recipe adapted to the FastGen interface.

Derived from Gsunshine/py-meanflow at 1f6d72d94247c8fdeb89489acca1a8007a6baf6c.
Licensed under CC BY-NC-SA 4.0; see LICENSE and UPSTREAM.md in this directory.
"""

import torch
from torch.nn import functional as F

from .augment import AugmentPipe
from .unet import SongUNet


class OriginalMeanFlowUNet(SongUNet):
    num_classes = 151

    def __init__(self, *, model_channels=128, num_blocks=4, img_resolution=64):
        super().__init__(
            img_resolution=img_resolution, in_channels=3, out_channels=3,
            label_dim=self.num_classes, augment_dim=6, model_channels=model_channels,
            channel_mult_noise=2, resample_filter=[1, 3, 3, 1],
            channel_mult=[2, 2, 2], encoder_type="standard", decoder_type="standard",
            num_blocks=num_blocks, dropout=0.2, label_dropout=0,
        )

    def forward(self, x, timestep, *, r, category, aug_cond=None):
        labels = F.one_hot(category, num_classes=self.num_classes).to(x.dtype)
        return super().forward(x, (timestep, timestep - r), aug_cond=aug_cond,
                               class_labels=labels)


def augmentation_pipe():
    # Exact configuration in the authors' training/train_loop.py.
    return AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0,
                       aniso=1, translate_frac=1)


def sample_times_v0(batch, device):
    t = (torch.randn(batch, device=device) * 2 - 2).sigmoid()
    r = (torch.randn(batch, device=device) * 2 - 2).sigmoid()
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    r = torch.where(torch.rand(batch, device=device) < 0.25, t, r)
    return r, t


def prediction_and_jvp(field, z, r, t, velocity):
    """Same field/dropout realization, without retaining the JVP reverse graph.

    torch.no_grad disables parameter-gradient recording but preserves forward
    AD. fork_rng restores the post-prediction RNG state, so the two forwards
    consume one dropout realization from the training stream.
    """
    before_cpu = torch.get_rng_state()
    before_cuda = torch.cuda.get_rng_state(z.device) if z.is_cuda else None
    prediction = field(z, r, t)
    devices = [z.device.index if z.device.index is not None else torch.cuda.current_device()] if z.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(before_cpu)
        if before_cuda is not None:
            torch.cuda.set_rng_state(before_cuda, z.device)
        with torch.no_grad():
            _, derivative = torch.func.jvp(
                field, (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t))
            )
    return prediction, derivative


def original_meanflow_loss(model, images, category, aug_cond=None, *, noise=None, r=None, t=None,
                           fused_jvp=False):
    if (r is None) != (t is None):
        raise ValueError("Supply both r and t, or neither")
    if noise is None:
        noise = torch.randn_like(images)
    if r is None:
        r, t = sample_times_v0(len(images), images.device)
    broadcast_t = t[:, None, None, None]
    interval = (t - r)[:, None, None, None]
    z = (1 - broadcast_t) * images + broadcast_t * noise
    velocity = noise - images

    def field(z, r, t):
        return model(z, t, r=r, category=category, aug_cond=aug_cond)

    if fused_jvp:
        # The authors' single-forward formulation, suitable for torch.compile.
        prediction, derivative = torch.func.jvp(
            field, (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
    else:
        prediction, derivative = prediction_and_jvp(field, z, r, t, velocity)
    correction = interval * derivative
    target = (velocity - correction).detach()
    squared_error = (prediction - target).float().square()
    # Upstream uses SUM over image dimensions, not pixel-mean MSE.
    squared_l2 = squared_error.flatten(1).sum(1)
    loss = (squared_l2 / (squared_l2.detach() + 1e-3).pow(0.75)).mean()
    with torch.no_grad():
        per_image_mse = squared_error.flatten(1).mean(1)
        fm = r == t
        stats = {
            "loss": loss.detach(), "mse": per_image_mse.mean(),
            "squared_l2": squared_l2.detach().mean(),
            "fm_error_sum": (per_image_mse * fm).sum(), "fm_count": fm.float().sum(),
            "mf_error_sum": (per_image_mse * ~fm).sum(), "mf_count": (~fm).float().sum(),
            "velocity_ms": velocity.square().mean(), "correction_ms": correction.square().mean(),
            "prediction_ms": prediction.detach().square().mean(), "target_ms": target.square().mean(),
        }
    return loss, stats


@torch.no_grad()
def update_original_ema(ema, model, step, decay=0.99995, period=16):
    # Matches upstream ema.py, using the decay reported in the paper.
    if step % period:
        return
    coefficient = 1 - decay ** period
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        updated = ema_parameter.double() + coefficient * (parameter.double() - ema_parameter.double())
        ema_parameter.copy_(updated.float())
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)
