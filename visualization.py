import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Sequence

from visualize import plot_data
from core import mae, r2, label_fn


def _to2d(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim != 2:
        raise TypeError(f"Expect 2D image, got shape {arr.shape}")
    return arr

def show_pipeline_comparison(inputs: List[Dict[str,Any]],
                             T_phys_list: Sequence[np.ndarray],
                             T_star_list=None,
                             keys=('k','q','T_bc','pi'),
                             row_labels=None,
                             cmap_name='viridis',
                             mask_bad_color='gray',
                             save_path=None):
    n_rows = len(inputs)
    row_labels = row_labels or [str(i) for i in range(n_rows)]
    titles = list(keys) + ['T_phys']
    has_star = T_star_list is not None
    if has_star: titles.append('T_star')

    imgs_by_col = []
    for k in keys:
        imgs_by_col.append([_to2d(d[k]) for d in inputs])
    phys_imgs = [_to2d(t) for t in T_phys_list]
    imgs_by_col.append(phys_imgs)
    if has_star:
        star_imgs = [_to2d(t) for t in T_star_list]
        imgs_by_col.append(star_imgs)

    vlims = [None]*len(titles)
    for i in range(len(keys)):
        col = imgs_by_col[i]
        vmin = np.nanmin([np.nanmin(m) for m in col])
        vmax = np.nanmax([np.nanmax(m) for m in col])
        vlims[i] = (vmin, vmax)

    all_T = phys_imgs + (star_imgs if has_star else [])
    Tmin = float(np.nanmin([np.nanmin(m) for m in all_T]))
    Tmax = float(np.nanmax([np.nanmax(m) for m in all_T]))
    vlims[len(keys)] = (Tmin, Tmax)
    if has_star: vlims[len(keys)+1] = (Tmin, Tmax)

    cmap = matplotlib.cm.get_cmap(cmap_name).copy(); cmap.set_bad(color=mask_bad_color)
    cols = len(titles)
    fig, axes = plt.subplots(n_rows, cols, figsize=(1.6*cols, 1.6*n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for c, title in enumerate(titles):
        vmin, vmax = vlims[c]
        for r in range(n_rows):
            ax = axes[r, c]
            im = ax.imshow(imgs_by_col[c][r], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            if r == 0: ax.set_title(title, fontsize=15)
            if c == 0: ax.set_ylabel(row_labels[r], fontsize=15)
            ax.axis('off')
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    else: plt.show()

def visualize_performance_histrogram(ref_errors, scaled_errors, bins=50, title='MAE',
                                     save_path=None, keep_frac=0.90):
    r = np.asarray(ref_errors).ravel()
    s = np.asarray(scaled_errors).ravel()

    abs_all = np.concatenate([np.abs(r), np.abs(s)])
    cutoff = np.quantile(abs_all, keep_frac)

    r_sel = r[np.abs(r) <= cutoff]
    s_sel = s[np.abs(s) <= cutoff]

    vmin = min(r_sel.min(), s_sel.min())
    vmax = max(r_sel.max(), s_sel.max())

    plt.figure(figsize=(6, 4))
    plt.hist(r_sel, bins=bins, range=(vmin, vmax), alpha=0.6, label=f"{title} (raw)", density=True)
    plt.hist(s_sel, bins=bins, range=(vmin, vmax), alpha=0.6, label=f"{title} (projected)", density=True)
    plt.xlabel(title); plt.ylabel("Density"); plt.title(f"Distribution (|x| ≤ {keep_frac*100:.0f}th pct)")
    plt.legend(); plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def visualize_detail(index, detailed_results, save_path=None):
    args = []
    for idx in index:
        res = detailed_results[idx]
        args.append([
            ("true", res["true"]),
            ("pred(projected)", res["pred_proj"]),
            ("pred(raw)", res["pred_raw"])
        ])
    plot_data(args, save_path=save_path, same_clim_row=True, cbar_ticks=True, title_size=12, fig_scale=0.8)

def plot_param_histogram_compare(input1, input2, key='T_bc', bins=50, labels=('Test', 'Train'), log=False, save_path=None):
    def flatten_key(params_list):
        values = []
        for p in params_list:
            tensor = torch.tensor(p[key])
            flat = tensor.flatten().cpu().numpy()
            flat = flat[~np.isnan(flat)]
            flat = flat[flat>0] + 1e-2
            values.append(flat)
        return np.concatenate(values)
    if log:
        values1 = np.log10(np.abs(flatten_key(input1))); values2 = np.log10(np.abs(flatten_key(input2))); key = f"log10({key})"
    else:
        values1 = flatten_key(input1); values2 = flatten_key(input2)
    min_val = min(values1.min(), values2.min()); max_val = max(values1.max(), values2.max())
    bins = np.linspace(min_val, max_val, bins)
    plt.figure(figsize=(5, 4))
    plt.hist(values1, bins=bins, density=True, alpha=0.6, label=labels[0], edgecolor='black')
    plt.hist(values2, bins=bins, density=True, alpha=0.6, label=labels[1], edgecolor='black')
    plt.xlabel(key); plt.ylabel("Probability Density"); plt.title(f"Histogram Comparison of {key}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_true_pred_scatter(
    true_values,
    series,
    *,
    q=0.99,
    scale='linear',
    threshold=0.0,
    sample=None,
    labels=None,
    label_fn=label_fn,
    figsize=(6,6),
    s=8, alpha=0.6,
    symlog_linthresh=1.0,
    save_path=None,
    rng_seed=0
):
    x = np.asarray(true_values, dtype=float).ravel()
    rng = np.random.default_rng(rng_seed)
    data = {}
    pooled = []

    for name, pred in series.items():
        y = np.asarray(pred, dtype=float).ravel()
        n = min(x.size, y.size)
        xx, yy = x[:n], y[:n]

        valid = np.isfinite(xx) & np.isfinite(yy)
        if scale == 'log':
            valid &= (xx > 0) & (yy > 0) & (xx > threshold) & (yy > threshold)
        elif scale == 'symlog':
            valid &= (np.abs(xx) > threshold) & (np.abs(yy) > threshold)
        else:
            valid &= (xx > threshold) & (yy > threshold)

        if not np.any(valid):
            continue

        xx, yy = xx[valid], yy[valid]
        if sample is not None and xx.size > sample:
            idx = rng.choice(xx.size, sample, replace=False)
            xx, yy = xx[idx], yy[idx]

        data[name] = (xx, yy)
        pooled.append(xx); pooled.append(yy)

    if not data:
        raise ValueError("Check the threshold, scale or data")

    if q < 1.0:
        lo_q = (1.0 - q) / 2.0
        hi_q = 1.0 - lo_q
        vals = np.concatenate(pooled)
        if scale == 'log':
            v = np.log10(vals[vals > 0])
            lo_log, hi_log = np.quantile(v, [lo_q, hi_q])
            lo, hi = 10**lo_log, 10**hi_log
            lo = max(lo, np.finfo(float).tiny)
        else:
            lo, hi = np.quantile(vals, [lo_q, hi_q])
    else:
        vals = np.concatenate(pooled)
        lo, hi = np.min(vals), np.max(vals)
    pad = (hi - lo) * 0.02 if hi > lo else 0.0
    lo, hi = lo - pad, hi + pad
    lo = max(lo, threshold)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.0, alpha=0.7)

    for name, (xx, yy) in data.items():
        lbl = (labels.get(name) if labels is not None and name in labels
               else (label_fn(name, xx, yy) if callable(label_fn) else name))
        ax.scatter(xx, yy, s=s, alpha=alpha, label=lbl)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    if scale == 'log':
        ax.set_xscale('log'); ax.set_yscale('log')
    elif scale == 'symlog':
        ax.set_xscale('symlog', linthresh=symlog_linthresh)
        ax.set_yscale('symlog', linthresh=symlog_linthresh)

    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(); plt.close()
