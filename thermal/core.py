import copy
from typing import Dict, Tuple, Iterable, Any
import numpy as np
import torch

Tensor = torch.Tensor
Problem = Dict[str, Any]

def to_tensor(x, dtype=torch.float32, device=None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device) if (dtype or device) else x
    return torch.tensor(x, dtype=dtype, device=device)

def gmean(x: Any, eps: float = 1e-12) -> float:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return float(np.exp(np.nanmean(np.log(arr + eps))))

def delta_T(T_bc: Tensor, dirichlet_mask: Tensor, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    dm = dirichlet_mask.bool()
    T_low = T_bc[dm].min()
    dT = (T_bc[dm].max() - T_low).clamp(min=eps)
    return T_low, dT

def domain_L(H: int, W: int, dx: float, dy: float) -> float:
    return max(H * dx, W * dy)

def pi_field(problem: Problem) -> Tensor:
    q: Tensor = problem['q']
    k: Tensor = problem['k']
    T_bc: Tensor = problem['T_bc']
    dm: Tensor = problem['dirichlet_mask']
    dx: float = float(problem['dx'])
    dy: float = float(problem['dy'])
    _, H, W = q.shape
    L = domain_L(H, W, dx, dy)
    T_low, dT = delta_T(T_bc, dm)
    return q * (L ** 2) / (k * dT)

def mask_T_bc_for_plot(problem: Problem) -> Problem:
    d = copy.deepcopy(problem)
    T = d['T_bc'][0].detach().cpu().numpy().copy()
    mask = d['dirichlet_mask'][0].detach().cpu().numpy().astype(bool)
    T[~mask] = np.nan
    d['T_bc'] = torch.from_numpy(T)[None, ...].to(dtype=problem['T_bc'].dtype, device=problem['T_bc'].device)
    return d

def check_problem(problem: Problem):
    required = {'k','q','T_bc','dirichlet_mask','neumann_mask','dx','dy','reg'}
    missing = required - set(problem.keys())
    if missing:
        raise KeyError(f"Problem missing keys: {missing}")
    for key in ['k','q','T_bc','dirichlet_mask','neumann_mask']:
        if not isinstance(problem[key], torch.Tensor):
            raise TypeError(f"{key} must be torch.Tensor")
        if problem[key].ndim != 3 or problem[key].shape[0] != 1:
            raise ValueError(f"{key} must be shape (1,H,W), got {problem[key].shape}")
    if torch.isnan(problem['T_bc']).any():
        dm = problem['dirichlet_mask'].bool()
        if torch.isnan(problem['T_bc'][~dm]).any():
            raise ValueError("T_bc contains NaN in interior; use mask_T_bc_for_plot only for visualization.")


def mae(xx, yy):
    return float(np.mean(np.abs(yy - xx)))


def r2(xx, yy, eps=1e-12):
    ss_res = float(np.sum((yy - xx) ** 2))
    ss_tot = float(np.sum((xx - xx.mean()) ** 2)) + eps
    return 1.0 - ss_res / ss_tot

def label_fn(name, xx, yy):
    return f"{name} (R²={r2(xx,yy):.2f}, MAE={mae(xx,yy):.1g})"