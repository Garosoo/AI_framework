import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Callable, Optional

from model import predict
Tensor = torch.Tensor

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:
    yp, yt = y_pred.view(-1), y_true.view(-1)
    ss_res = torch.sum((yt - yp) ** 2)
    ss_tot = torch.sum((yt - torch.mean(yt)) ** 2)
    return float(1 - ss_res / (ss_tot + eps))

METRIC_FUNCS: Dict[str, Callable[[Tensor, Tensor], float]] = {
    "mae": lambda y_pred, y_true: torch.nn.functional.l1_loss(y_pred, y_true).item(),
    "mse": lambda y_pred, y_true: torch.nn.functional.mse_loss(y_pred, y_true).item(),
    "rmse": lambda y_pred, y_true: torch.sqrt(torch.nn.functional.mse_loss(y_pred, y_true)).item(),
    "r2":  r2_score,
}

def abs_error_flat_np(y_pred: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
    ae = torch.nn.functional.l1_loss(y_pred, y_true, reduction='none')
    return ae.reshape(-1).detach().cpu().numpy().ravel()

def compute_metrics(y_pred: Tensor, y_true: Tensor,
                    metrics: Dict[str, Callable[[Tensor, Tensor], float]] = METRIC_FUNCS) -> Dict[str, float]:
    return {name: fn(y_pred, y_true) for name, fn in metrics.items()}


def evaluate(model,
             train_set: List[Dict[str, Any]],
             test_set:  List[Dict[str, Any]],
             build_fn:  Callable[[Dict[str,Any]], Tuple[Tensor, Tensor]],
             projector: Optional[Callable[[Dict[str,Any], List[Dict[str,Any]]], Dict[str,Any]]] = None
             ) -> Tuple[Dict[str,float], Optional[Dict[str,float]], List[Dict[str,Any]]]:

    names = list(METRIC_FUNCS.keys())
    raw_agg  = {n: [] for n in names}
    proj_agg = {n: [] for n in names} if projector else None
    details: List[Dict[str,Any]] = []

    for idx, test in enumerate(test_set):
        _, y_true = build_fn(test)  # (C,H,W),(1,H,W)
        # _, y_pred_raw = build_fn(test)  # (C,H,W),(1,H,W)
        y_pred_raw = predict(model, test, build_fn=build_fn)
        raw_metrics = compute_metrics(y_pred_raw, y_true)
        for n in names: raw_agg[n].append(raw_metrics[n])

        rec: Dict[str,Any] = {
            "idx": idx,
            "true": y_true[0],
            "pred_raw": y_pred_raw[0],
            **{f"raw_{n}": raw_metrics[n] for n in names},
            "raw_errors": abs_error_flat_np(y_pred_raw, y_true),
        }

        if projector:
            proj = projector(test, train_set)
            scaled = proj["scaled_sample"]
            unscale_pred = proj.get("unscale_pred", lambda t: t)
            meta = proj.get("meta", {})

            # _, y_pred_scaled = build_fn(scaled)  # (C,H,W),(1,H,W)
            y_pred_scaled = predict(model, scaled, build_fn=build_fn)
            y_pred_proj   = unscale_pred(y_pred_scaled)
            proj_metrics  = compute_metrics(y_pred_proj, y_true)
            for n in names: proj_agg[n].append(proj_metrics[n])

            rec.update({
                "pred_proj": y_pred_proj[0],
                **{f"proj_{n}": proj_metrics[n] for n in names},
                "proj_errors": abs_error_flat_np(y_pred_proj, y_true),
                "meta": meta
            })
        details.append(rec)

    raw_mean = {n: float(np.mean(raw_agg[n])) for n in names}
    proj_mean = None
    if projector:
        proj_mean = {n: float(np.mean(proj_agg[n])) for n in names}

    return raw_mean, proj_mean, details

def evaluate_raw_vs_projected(model,
                              train_set: List[Dict[str, Any]],
                              test_set:  List[Dict[str, Any]]):
                              
    from ..buckingham.thermal.task import build_input_tensor as build_input_tensor_thermal
    from ..buckingham.thermal.projection import projector

    raw_mean, proj_mean, details = evaluate(
        model, train_set, test_set,
        build_fn=build_input_tensor_thermal,
        projector=projector
    )
    return raw_mean, proj_mean, details
