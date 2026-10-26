"""Portable four-call sampler for the unchanged 101k EMA MeanFlow checkpoint."""

import torch


BEST_TIMES = (1.0, 0.6, 0.3, 0.018, 0.0)


@torch.no_grad()
def sample_best_four(model, shape, *, device="cuda", category=None):
    """Return normalized RGB images using exactly four conditional model calls.

    Put the loaded model in eval mode and on ``device`` before calling. Category
    IDs use category_to_id.json, not Pokédex numbers. The built-in guidance needs
    no additional unconditional network pass.
    """
    if model.training:
        raise ValueError("Call model.eval() before sampling")
    if len(shape) != 4 or tuple(shape[1:]) != (3, 64, 64) or shape[0] < 1:
        raise ValueError("Expected shape (batch, 3, 64, 64)")
    device = torch.device(device)
    if category is None:
        category = torch.randint(model.backbone.num_classes, (shape[0],), device=device)
    else:
        category = torch.as_tensor(category, device=device)
        if category.is_floating_point() or category.is_complex():
            raise ValueError("Categories must be integer IDs")
        category = category.long()
    if category.shape != (shape[0],) or ((category < 0) | (category >= 151)).any():
        raise ValueError("Expected one category ID in [0, 150] per image")
    x = torch.randn(shape, device=device)
    for start, end in zip(BEST_TIMES[:-1], BEST_TIMES[1:]):
        t = x.new_full((shape[0],), start)
        r = x.new_full((shape[0],), end)
        x = x - (start - end) * model(x, t, r=r, category=category)
    return x.clamp(-1, 1)
