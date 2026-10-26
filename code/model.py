"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import torch
from torch import nn

from src.utils import count_parameters, parameter_summary


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()





class ModelFewNFE(Model):
    """Conditional MeanFlow with training-time CFG; exactly four sampling calls."""
    sampling_steps = 4
    sampling_rho = 1.0  # Also used by training-only wrappers that bypass __init__.

    def __init__(self, device="cpu", sampling_rho=1.0, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected model options: {sorted(kwargs)}")
        from src.sampling_schedule import power_schedule
        power_schedule(self.sampling_steps, sampling_rho)
        from src.cfg_meanflow import CFGMeanFlowUNet
        super().__init__(CFGMeanFlowUNet())
        self.sampling_rho = float(sampling_rho)
        self.to(device)

    @torch.no_grad()
    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected sampling options: {sorted(kwargs)}")
        if len(shape) != 4 or tuple(shape[1:]) != (3, 64, 64) or shape[0] < 1:
            raise ValueError("Expected shape (batch_size, 3, 64, 64)")
        device = torch.device(device)
        if category is None:
            category = torch.randint(self.backbone.num_classes, (shape[0],), device=device)
        else:
            category = torch.as_tensor(category, device=device)
            if category.is_floating_point() or category.is_complex():
                raise ValueError("category must contain integer class IDs")
            category = category.long()
        if category.shape != (shape[0],):
            raise ValueError("Expected one category ID per sample")
        if ((category < 0) | (category >= self.backbone.num_classes)).any():
            raise ValueError("category IDs must lie in [0, 150]")
        x = torch.randn(shape, device=device)
        from src.sampling_schedule import power_schedule
        times = power_schedule(self.sampling_steps, self.sampling_rho)
        for start, end in zip(times[:-1], times[1:]):
            t = x.new_full((shape[0],), start)
            r = x.new_full((shape[0],), end)
            x = x - (start - end) * self(x, t, r=r, category=category)
        return x.clamp(-1, 1)


class ModelOneNFE(ModelFewNFE):
    """Same backbone/state_dict; exactly one network call during sampling."""
    sampling_steps = 1
