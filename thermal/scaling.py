import random
import numpy as np
import torch
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional

from core import delta_T, pi_field


class ThermalScaler:

    def __init__(self, tol: float = 1e-9):
        self.tol = tol
        self.a = self.bq = self.bT = self.dx_s = self.dy_s = None
        self.q_scale = None

    def set_from_solution(self, sol: Dict[str, float], mode: str = 'enforce_pi'):
        a   = float(sol['alpha'])
        bq  = float(sol['beta'])
        bT  = float(sol['gamma_T'])
        dxs = float(sol['delta_x'])
        dys = float(sol['delta_y'])

        lhs = bq * dxs * dys
        rhs = a  * bT
        if abs(lhs - rhs) > self.tol:
            if mode in ('enforce_pi', 'adjust_beta'):
                bq = (a * bT) / (dxs * dys)
            elif mode == 'adjust_gamma_T':
                bT = (bq * dxs * dys) / a
            else:
                raise ValueError(f"Inconsistent scales: beta*dx*dy={lhs} != alpha*gamma_T={rhs}")

        self.a, self.bq, self.bT, self.dx_s, self.dy_s = a, bq, bT, dxs, dys
        self.q_scale = bq

    def scale(self, sample: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        if None in (self.a, self.q_scale, self.bT, self.dx_s, self.dy_s):
            raise RuntimeError("You should first call set_from_solution().")
        d = deepcopy(sample)

        d['k'] = d['k'] * self.a
        d['q'] = d['q'] * self.q_scale

        dm = d['dirichlet_mask'].bool()
        T = d['T_bc'].clone()
        T_low, _ = delta_T(T, dm)
        T[dm] = T_low + self.bT * (T[dm] - T_low)
        d['T_bc'] = T

        d['dx'] = float(d['dx']) * self.dx_s
        d['dy'] = float(d['dy']) * self.dy_s

        unscale_info = {'beta': torch.tensor(self.bT, dtype=T.dtype, device=T.device),
                        'T_low': T_low}
        return d, unscale_info

    @staticmethod
    def unscale_T(T_scaled: torch.Tensor, unscale_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        beta  = unscale_info['beta']
        T_low = unscale_info['T_low']
        return T_low + (T_scaled - T_low) / beta


def buckingham_pi_random_scaling(
    test_set: List[Dict[str, Any]],
    repeats: int = 1,
    alpha_range: Tuple[float, float]   = (0.5, 10.0),
    gamma_T_range: Tuple[float, float] = (0.5, 10.0),
    gamma_range: Tuple[float, float]   = (0.5, 10.0),
    seed: Optional[int] = None,
    add_pi: bool = True,
) -> List[Dict[str, Any]]:

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    out: List[Dict[str, Any]] = []
    for sample in test_set:
        for _ in range(repeats):
            a   = float(torch.empty(1).uniform_(*alpha_range))
            gT  = float(torch.empty(1).uniform_(*gamma_T_range))
            g   = float(torch.empty(1).uniform_(*gamma_range))
            bq  = (a * gT) / (g * g)

            sol = {'alpha': a, 'beta': bq, 'gamma_T': gT, 'delta_x': g, 'delta_y': g}
            scaler = ThermalScaler()
            scaler.set_from_solution(sol, mode='enforce_pi')
            scaled, _ = scaler.scale(sample)
            if add_pi:
                scaled['pi'] = pi_field(scaled)
            out.append(scaled)
    return out


