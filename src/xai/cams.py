"""Class Activation Mapping (CAM) implementations for MODAL-Net (WP10).

Native PyTorch implementations (zero external dependencies):
  - Grad-CAM: Global average pooling of gradients as channel weights.
  - Grad-CAM++: Pixel-wise alpha-weighted positive gradients for finer multi-instance localization.
  - LayerCAM: Element-wise ReLU gradient weighting across stage-2 (32x32) and stage-3 (16x16).
  - Eigen-CAM: Gradient-free first principal component of channel-centered activations via SVD.

Target layer resolutions for MaxViT-L (512x512 input):
  - stages[-2] (Stage 2): 32 x 32 resolution, 512 channels.
  - stages[-1] (Stage 3): 16 x 16 resolution, 1024 channels.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from src.models.gradenet import probs_from_logits


class ProbWrapper(nn.Module):
    """Wraps classification model to output target class scores for CAM backpropagation."""

    def __init__(self, model: nn.Module, head_type: str = 'softmax'):
        super().__init__()
        self.model = model
        self.head_type = head_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model(x)
        if self.head_type == 'coral':
            p = probs_from_logits(z, head_type='coral')
            return torch.log(torch.clamp(p, min=1e-12))
        return z


class ActivationGradientHook:
    """Registers forward and backward hooks to capture activations and gradients."""

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.fwd_handle = layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, outp):
        self.activations = outp

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def get_target_layers(model: nn.Module) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Resolve target layers for CAM extraction.
    For MaxViT/GradeNet:
      stage_last = backbone.stages[-1] (Stage 3, 16x16)
      stage_penult = backbone.stages[-2] (Stage 2, 32x32)
    """
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'stages') and len(model.backbone.stages) >= 2:
        return model.backbone.stages[-1], model.backbone.stages[-2]
    elif hasattr(model, 'stages') and len(model.stages) >= 2:
        return model.stages[-1], model.stages[-2]
    else:
        # Fallback for toy CNNs in tests
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if len(conv_layers) >= 2:
            return conv_layers[-1], conv_layers[-2]
        elif len(conv_layers) == 1:
            return conv_layers[-1], None
        else:
            modules = list(model.children())
            return modules[-1], (modules[-2] if len(modules) >= 2 else None)


def compute_gradcam(A: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Grad-CAM: w_k = mean_ij g_k(i,j); cam = ReLU(sum_k w_k * A_k)."""
    w = g.mean(dim=(-2, -1), keepdim=True)
    cam = F.relu((w * A).sum(dim=0))
    return cam


def compute_gradcampp(A: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Grad-CAM++: alpha_k(i,j) = g^2 / (2*g^2 + sum_ab A_k(a,b)*g^3 + 1e-7); w_k = sum_ij alpha*ReLU(g)."""
    g2 = g ** 2
    g3 = g2 * g
    sum_a_g3 = (A * g3).sum(dim=(-2, -1), keepdim=True)
    alpha = g2 / (2.0 * g2 + sum_a_g3 + 1e-7)
    w = (alpha * F.relu(g)).sum(dim=(-2, -1), keepdim=True)
    cam = F.relu((w * A).sum(dim=0))
    return cam


def compute_layercam_single(A: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """LayerCAM on a single stage: cam = ReLU(sum_k ReLU(g_k) * A_k)."""
    w = F.relu(g)
    cam = F.relu((w * A).sum(dim=0))
    return cam


def compute_eigencam(A: torch.Tensor) -> torch.Tensor:
    """Eigen-CAM: 1st principal component of channel-centered activations via SVD."""
    C, H, W = A.shape
    A_flat = A.view(C, -1).T  # (HW, C)
    A_cent = A_flat - A_flat.mean(dim=0, keepdim=True)

    U, S, V = torch.linalg.svd(A_cent, full_matrices=False)
    v0 = V[0]  # First principal component (C,)
    proj = A_cent @ v0  # (HW,)

    # Sign alignment: positive correlation with mean activation
    mean_act = A.mean(dim=0).view(-1)
    cov = ((proj - proj.mean()) * (mean_act - mean_act.mean())).sum()
    if cov < 0:
        proj = -proj

    cam = F.relu(proj.view(H, W))
    return cam


def postprocess_cam(
    cam_raw: Union[torch.Tensor, np.ndarray],
    fundus_mask: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Post-processing: ReLU -> min-max to [0,1] -> bilinear upsample to 512x512 -> mask -> min-max."""
    if isinstance(cam_raw, torch.Tensor):
        cam = cam_raw.detach().cpu().numpy()
    else:
        cam = cam_raw.copy()

    cam = np.maximum(cam, 0.0)
    denom = cam.max() - cam.min()
    if denom > 1e-8:
        cam = (cam - cam.min()) / denom
    else:
        cam = np.zeros_like(cam)

    if cam.shape != target_size:
        cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)

    if fundus_mask is not None:
        cam = cam * fundus_mask.astype(np.float32)
        masked_vals = cam[fundus_mask]
        if len(masked_vals) > 0 and (masked_vals.max() - masked_vals.min()) > 1e-8:
            cam = (cam - masked_vals.min()) / (masked_vals.max() - masked_vals.min())
            cam = np.clip(cam * fundus_mask, 0.0, 1.0)
        else:
            cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def compute_model_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    method: str = 'gradcampp',
    head_type: str = 'softmax',
    fundus_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute CAM map for a single model on one input image.
    Model in eval(), fp32, gradients enabled, no autocast.
    """
    model.eval()
    wrapper = ProbWrapper(model, head_type=head_type)
    target_layer1, target_layer2 = get_target_layers(model)

    device = input_tensor.device
    method = method.lower()

    if method == 'eigencam':
        hook = ActivationGradientHook(target_layer1)
        with torch.no_grad():
            _ = wrapper(input_tensor)
        A = hook.activations[0]
        hook.remove()
        cam = compute_eigencam(A)
        return postprocess_cam(cam, fundus_mask=fundus_mask)

    elif method == 'layercam':
        hook1 = ActivationGradientHook(target_layer1)
        hook2 = ActivationGradientHook(target_layer2) if target_layer2 is not None else None

        model.zero_grad()
        out = wrapper(input_tensor)
        score = out[0, target_class]
        score.backward()

        cam1 = compute_layercam_single(hook1.activations[0], hook1.gradients[0])
        map1 = postprocess_cam(cam1, fundus_mask=None, target_size=(512, 512))
        hook1.remove()

        if hook2 is not None and hook2.activations is not None and hook2.gradients is not None:
            cam2 = compute_layercam_single(hook2.activations[0], hook2.gradients[0])
            map2 = postprocess_cam(cam2, fundus_mask=None, target_size=(512, 512))
            hook2.remove()
            cam_avg = 0.5 * (map1 + map2)
        else:
            if hook2 is not None:
                hook2.remove()
            cam_avg = map1

        return postprocess_cam(cam_avg, fundus_mask=fundus_mask)

    elif method == 'gradcampp':
        hook = ActivationGradientHook(target_layer1)
        model.zero_grad()
        out = wrapper(input_tensor)
        score = out[0, target_class]
        score.backward()

        cam = compute_gradcampp(hook.activations[0], hook.gradients[0])
        hook.remove()
        return postprocess_cam(cam, fundus_mask=fundus_mask)

    elif method == 'gradcam':
        hook = ActivationGradientHook(target_layer1)
        model.zero_grad()
        out = wrapper(input_tensor)
        score = out[0, target_class]
        score.backward()

        cam = compute_gradcam(hook.activations[0], hook.gradients[0])
        hook.remove()
        return postprocess_cam(cam, fundus_mask=fundus_mask)

    else:
        raise ValueError(f"Unknown CAM method: {method}")


def compute_ensemble_cam(
    models: List[nn.Module],
    input_tensor: torch.Tensor,
    target_class: int,
    method: str = 'gradcampp',
    head_type: str = 'softmax',
    fundus_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute CAM map across an ensemble of models and average normalized maps."""
    maps = []
    for m in models:
        c_map = compute_model_cam(
            m,
            input_tensor=input_tensor,
            target_class=target_class,
            method=method,
            head_type=head_type,
            fundus_mask=fundus_mask,
        )
        maps.append(c_map)

    avg_map = np.mean(maps, axis=0)
    denom = avg_map.max() - avg_map.min()
    if denom > 1e-8:
        avg_map = (avg_map - avg_map.min()) / denom
    if fundus_mask is not None:
        avg_map = avg_map * fundus_mask.astype(np.float32)
    return avg_map.astype(np.float32)
