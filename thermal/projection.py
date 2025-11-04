from typing import Dict, Any, List
import numpy as np
import torch
from .core import gmean, delta_T
from .scaling import ThermalScaler

def summarize_for_projection(sample):
    k_g  = gmean(sample['k'])
    q    = sample['q']
    q_np = q.detach().cpu().numpy() if isinstance(q, torch.Tensor) else np.asarray(q)
    q_pos = q_np[q_np > 0]; q_g = float(np.exp(np.nanmean(np.log(q_pos + 1e-12)))) if q_pos.size>0 else 1e-12
    T_low, dT = delta_T(sample['T_bc'], sample['dirichlet_mask'])
    g = float(sample['dx'])
    return np.array([np.log(k_g), np.log(q_g), float(torch.log(dT).item()), np.log(g)], dtype=float)

def solve_scales_constrained(test_sample, train_samples, weights=None):
    v_t = summarize_for_projection(test_sample)
    V_i = np.stack([summarize_for_projection(p) for p in train_samples], axis=0)
    if weights is None: W = np.eye(4, dtype=float)
    else: w = np.asarray(weights, float).reshape(-1); assert w.size==4; W = np.diag(w)

    C = np.array([[-1.0, 1.0, -1.0, 2.0]], dtype=float)
    KKT = np.block([[W, C.T],[C, np.zeros((1,1), float)]])

    sols, residuals = [], np.empty(V_i.shape[0], float)
    for i in range(V_i.shape[0]):
        s_target = V_i[i] - v_t
        rhs = np.concatenate([W @ s_target, np.array([0.0])])
        sol = np.linalg.solve(KKT, rhs)
        s = sol[:4]
        e = s - s_target
        residuals[i] = float(np.sqrt(e @ (W @ e)))

        a   = float(np.exp(s[0]))
        b   = float(np.exp(s[1]))
        gT  = float(np.exp(s[2]))
        g   = float(np.exp(s[3]))
        sols.append({'alpha':a,'beta':b,'gamma_T':gT,'delta_x':g,'delta_y':g,
                     'residual':residuals[i], 'idx':i})

    best_idx = int(np.argmin(residuals))
    return {'solutions': sols, 'residuals': residuals,
            'best_idx': best_idx, 'best': sols[best_idx]}

def project_to_trainset(test_sample: Dict[str, Any],
                        train_samples: List[Dict[str, Any]],
                        mode: str = 'enforce_pi'):
    data = solve_scales_constrained(test_sample, train_samples, weights=None)
    best = data['best']
    scaler = ThermalScaler()
    scaler.set_from_solution(best, mode=mode)
    scaled_test, unscale_info = scaler.scale(test_sample)
    return {
        'scaled_test': scaled_test,
        'unscale_info': unscale_info,
        'best_idx': data['best_idx'],
        'solution': best,
        'residuals': data['residuals']
    }

def projector(test_sample: Dict[str, Any], train_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    data = project_to_trainset(test_sample, train_set, mode='enforce_pi')
    scaled = data['scaled_test']
    unscale_info = data['unscale_info']

    def unscale_pred(y_pred):
        return ThermalScaler.unscale_T(y_pred, unscale_info)

    return {
        "scaled_sample": scaled,
        "unscale_pred": unscale_pred,
        "meta": {"best_idx": int(data['best_idx'])}
    }

