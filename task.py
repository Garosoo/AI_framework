import random
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import torch
from noise import pnoise2

from core import pi_field, to_tensor
from solver import solve as solve_pde


def generate_perlin_terrain(B=1, H=64, W=64, scale=10.0, k_scale=150.0, seed: Optional[int]=0):
    if seed is not None:
        np.random.seed(seed)
    noise_fn = np.vectorize(lambda y, x, b: pnoise2(y / scale, x / scale, base=b), otypes=[np.float32])
    terrain = np.stack([noise_fn(*np.indices((H, W)), (seed or 0) + b) for b in range(B)])  # (B,H,W)
    terrain -= terrain.min() + 1e-3
    terrain /= terrain.max()
    terrain *= k_scale
    return torch.from_numpy(terrain.astype(np.float32))

def make_input(B=1, H=32, W=32, dx=1e-3, dy=1e-3, reg=1e-8,
               seed: Optional[int]=None, T_range=(0.0, 10.0), q_amp_range=(1e6, 2e7)) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    k = generate_perlin_terrain(B=B, H=H, W=W, scale=6.0, k_scale=300.0, seed=seed)
    k = k - k.min() + 1.0
    q = torch.zeros(B, H, W, dtype=torch.float32)
    h0, h1 = H // 3, 2 * H // 3
    w0, w1 = W // 3, 2 * W // 3

    T_bc = torch.zeros(B, H, W, dtype=torch.float32)
    for b in range(B):
        amp = random.uniform(*q_amp_range)
        q[b, h0:h1, w0:w1] = amp
        T_bc[b, :,  0] = random.uniform(*T_range)
        T_bc[b, :, -1] = random.uniform(*T_range)
        T_bc[b,  0, :] = random.uniform(*T_range)
        T_bc[b, -1, :] = random.uniform(*T_range)

    dm = torch.ones(B, H, W, dtype=torch.bool)
    dm[:, 1:-1, 1:-1] = False
    nm = torch.zeros_like(dm)

    params = {
        'dx': float(dx), 'dy': float(dy), 'reg': float(reg),
        'k': k, 'q': q, 'T_bc': T_bc,
        'dirichlet_mask': dm, 'neumann_mask': nm, 'Q_bc': None
    }
    params['pi'] = pi_field(params)
    return params

def mask_T_bc_for_plot(d: Dict[str, Any]):
    from .core import mask_T_bc_for_plot as _mask
    return _mask(d)

def generate_scaled_dataset(N: int, make_kwargs: Dict[str, Any], scaler_ranges: Dict[str, Tuple[float,float]] = None):
    from thermal.scaling import ThermalScaler
    scaler_ranges = scaler_ranges or {}
    alpha_r = scaler_ranges.get("alpha", (0.05, 20.0))
    beta_r  = scaler_ranges.get("beta",  (0.1,  5.0))
    gT_r    = scaler_ranges.get("gamma_T", (0.1,  5.0))
    dx_r    = scaler_ranges.get("delta_x", (0.2,  5.0))
    dy_r    = scaler_ranges.get("delta_y", (0.2,  5.0))

    originals: List[Dict[str, Any]] = []
    scaled:    List[Dict[str, Any]] = []

    for i in range(N):
        inp = make_input(**make_kwargs, seed=i)
        originals.append(inp)

        sol = {
            'alpha':   float(torch.empty(1).uniform_(*alpha_r)),
            'beta':    float(torch.empty(1).uniform_(*beta_r)),
            'gamma_T': float(torch.empty(1).uniform_(*gT_r)),
            'delta_x': float(torch.empty(1).uniform_(*dx_r)),
            'delta_y': float(torch.empty(1).uniform_(*dy_r)),
        }
        scaler = ThermalScaler(); scaler.set_from_solution(sol, mode='enforce_pi')
        inp_scaled, _ = scaler.scale(inp)
        scaled.append(inp_scaled)

    return originals, scaled


def build_input_tensor(sample: Dict[str, Any],
                       use_masks: bool = True,
                       broadcast_dxdy: bool = True,
                       log_kqt: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    C = []
    k = to_tensor(sample['k'][0])
    q = to_tensor(sample['q'][0])
    T = to_tensor(sample['T_bc'][0])
    if log_kqt:
        k = torch.log(k + 1e-4); q = torch.log(q + 1e-4); T = torch.log(T + 1e-4)
    C += [k, q, T]

    if use_masks:
        if 'dirichlet_mask' in sample: C.append(to_tensor(sample['dirichlet_mask'][0], dtype=torch.float32))
        if 'neumann_mask'   in sample: C.append(to_tensor(sample['neumann_mask'][0],   dtype=torch.float32))

    H, W = C[0].shape
    if broadcast_dxdy:
        dx = to_tensor(sample['dx']).expand(H, W)
        dy = to_tensor(sample['dy']).expand(H, W)
        C += [dx, dy]

    x = torch.stack(C, dim=0)   # [C,H,W]
    y = solve_pde(sample)       # [1,H,W]
    return x, y
