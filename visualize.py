import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt


def visualize_temperature(T_pred, title="Predicted Temperature", output_path=None):
    if T_pred.dim() == 2:
        plt.imshow(T_pred.cpu().numpy(), origin='upper', cmap='jet')
        plt.colorbar()
        plt.title(title)
    else:
        c, h, w = T_pred.shape
        fig, axes = plt.subplots(nrows=1, ncols=c, figsize=(5 * c, 5))

        for i in range(c):
            im = axes[i].imshow(T_pred[i].cpu().numpy(), origin='upper', cmap='jet')
            fig.colorbar(im, ax=axes[i])
            axes[i].set_title(f"{title} - Channel {i}")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def plot_data(named_array_grid, origin='upper', save_path=None,
              same_clim_row=False,       
              cmap='jet',
              titles_top_only=True,       
              title_size=16,              
              fig_scale=1.2,             
              cbar_ticks=True):           
    import numpy as np
    import matplotlib.pyplot as plt

    rows = len(named_array_grid)
    cols = max(len(row) for row in named_array_grid)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(2.6 * cols * fig_scale, 2.0 * rows * fig_scale))

    if rows == 1: axes = np.expand_dims(axes, 0)
    if cols == 1: axes = np.expand_dims(axes, 1)
    axes = np.array(axes)

    vlims_row = [None]*rows

    if same_clim_row:
        for i in range(rows):
            vmins, vmaxs = [], []
            for item in named_array_grid[i]:
                if not item or item[1] is None or item[0] == "": continue
                a = item[1].cpu().numpy() if hasattr(item[1], "cpu") else np.array(item[1])
                a = a[np.isfinite(a)]
                if a.size:
                    vmins.append(np.min(a)); vmaxs.append(np.max(a))
            if vmins:
                vlims_row[i] = (float(min(vmins)), float(max(vmaxs)))

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            try:
                item = named_array_grid[i][j]
                if not item or item[1] is None or item[0] == "":
                    ax.axis('off'); continue

                name, arr = item

                if hasattr(arr, "cpu"): arr = arr.cpu()
                arr = np.array(arr)

                vmin = vmax = None
                if same_clim_row and vlims_row[i] is not None:
                    vmin, vmax = vlims_row[i]

                im = ax.imshow(arr, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)

                if (not titles_top_only) or (i == 0):
                    ax.set_title(name, fontsize=title_size)
                ax.axis('off')

                if not np.isnan(arr).all():
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    if not cbar_ticks:
                        cb.set_ticks([]); cb.ax.tick_params(length=0, labelsize=0)

            except Exception:
                ax.axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
