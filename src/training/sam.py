"""Sharpness-Aware Minimization (SAM) optimizer.

Reference:
    Foret et al., 2021: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    https://arxiv.org/abs/2010.01412
"""

import torch
from typing import Dict, Any, Callable, Optional


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.

    Optimizes both the loss value and the loss sharpness by seeking parameters
    whose entire neighborhood has uniformly low training loss.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls: Callable,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Compute perturbation and climb to local loss maximum."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore original parameters and perform base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Standard step using closure (two forward/backward passes)."""
        if closure is None:
            raise ValueError("SAM step requires a closure returning the loss.")

        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step(zero_grad=False)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        """Compute the norm of the gradients across all parameter groups."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p_grad = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    norms.append(p_grad.norm(p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
