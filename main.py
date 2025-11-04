from seed import fix_seed
fix_seed(42)
import os
import numpy as np

from model import get_model_lightning
from evaluation import evaluate
from visualization import visualize_performance_histrogram, visualize_detail, plot_true_pred_scatter
from visualization import plot_param_histogram_compare


def save_results(metric: str, details: list, save_dir: str):
    r_raw = [d[f"raw_{metric}"] for d in details]
    r_proj = [d[f"proj_{metric}"] for d in details]
    visualize_performance_histrogram(r_raw, r_proj, bins=None, title=metric,
                                     save_path=os.path.join(save_dir, f"{metric}_perf.png"))

    k = min(5, len(details))
    best_idx = np.argsort(r_proj)[:k]
    worst_idx = np.argsort(r_proj)[-k:]
    visualize_detail(best_idx, details, save_path=os.path.join(save_dir, f"{metric}_best.png"))
    visualize_detail(worst_idx, details, save_path=os.path.join(save_dir, f"{metric}_worst.png"))


def run_task(
        task_name: str,
        *,
        make_input_fn,
        build_fn,
        projector,
        N: int,
        model_class,
        train_ratio: float,
        ckpt_dir: str,
        epochs: int,
        make_kwargs: dict,
        scaling_fn=None,
        train: bool = True,
        ckpt_path: str | None = None,
        lr=1e-3,
        use_two_step: bool = True,
):

    print(task_name, N, model_class, train_ratio, ckpt_dir)
    save_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(save_dir, exist_ok=True)

    originals = [make_input_fn(**make_kwargs, seed=i) for i in range(N)]
    n_train = int(N * train_ratio)
    train_set = originals[:n_train]
    test_set = originals[n_train:]
    if scaling_fn is not None:
        test_set = scaling_fn(test_set)

    model = get_model_lightning(
        model_class,
        samples=train_set,
        epochs=epochs,
        ckpt_dir=ckpt_dir,
        train=train,
        path=(ckpt_path or f"{ckpt_dir}/best.ckpt"),
        build_fn=build_fn,
        lr=lr,
    )

    if task_name == "thermal":
        plot_param_histogram_compare(test_set, train_set, key='dx', bins=50,
                                     save_path=os.path.join(save_dir, "feature_dist_dx.png"))
        plot_param_histogram_compare(test_set, train_set, key='T_bc', bins=100,
                                     save_path=os.path.join(save_dir, "feature_dist_tbc.png"))
        plot_param_histogram_compare(test_set, train_set, key='k', log=True, bins=100,
                                     save_path=os.path.join(save_dir, "feature_dist_k.png"))
        plot_param_histogram_compare(test_set, train_set, key='q', log=True, bins=100,
                                     save_path=os.path.join(save_dir, "feature_dist_q.png"))
    else:
        plot_param_histogram_compare(test_set, train_set, key='dx', bins=50,
                                     save_path=os.path.join(save_dir, "feature_dist_dx.png"))
        plot_param_histogram_compare(test_set, train_set, key='u_bc', bins=100,
                                     save_path=os.path.join(save_dir, "feature_dist_ubc.png"))
        plot_param_histogram_compare(test_set, train_set, key='f', log=True, bins=100,
                                     save_path=os.path.join(save_dir, "feature_dist_f.png"))
        plot_param_histogram_compare(test_set, train_set, key='E', log=True, bins=100,
                                     save_path=os.path.join(save_dir, "feature_dist_E.png"))

    raw_mean, proj_mean, details = evaluate(
        model, train_set, test_set,
        build_fn=build_fn,
        projector=projector
    )
    print(f"[{task_name.upper()} BASELINE       ]", raw_mean)
    print(f"[{task_name.upper()} PROJECTION]", proj_mean)

    true = np.concatenate([d["true"].cpu().numpy().ravel() for d in details])
    series = {
        "baseline": np.concatenate([d["pred_raw"].cpu().numpy().ravel() for d in details]),
        "projection": np.concatenate([d["pred_proj"].cpu().numpy().ravel() for d in details]),
    }
    plot_true_pred_scatter(true, series, sample=100000, scale='symlog', threshold=0.0,
                           save_path=os.path.join(save_dir, "true_pred_scatter_symlog.png"))
    plot_true_pred_scatter(true, series, sample=100000, scale='linear', threshold=0.0,
                           save_path=os.path.join(save_dir, "true_pred_scatter_linear.png"))

    r_raw = np.concatenate([d[f"raw_errors"] for d in details])
    r_proj = np.concatenate([d[f"proj_errors"] for d in details])
    visualize_performance_histrogram(r_raw, r_proj, bins=100, title='MAE',
                                     save_path=os.path.join(save_dir, f"MAE_perf.png"))
    save_results('r2', details, save_dir)
    save_results('mae', details, save_dir)


def run_thermal():
    from model import UNetSmall, TinyCNN, FNO2d
    from thermal.task import make_input, build_input_tensor
    from thermal.projection import projector
    from thermal.scaling import buckingham_pi_random_scaling
    model_path = '/home/seokki/project/pdeflow/250417/numerical_solver/code/buckingham/1step/thermal/cnn/best.ckpt'
    run_task(
        "thermal",
        make_input_fn=make_input,
        build_fn=build_input_tensor,
        projector=projector,
        N=2000,
        model_class=TinyCNN,
        train_ratio=0.5,
        ckpt_dir="./2step/thermal/cnn/",
        epochs=500,
        make_kwargs={"H": 32, "W": 32, "dx": 1e-3, "dy": 1e-3, "reg": 1e-8},
        scaling_fn=buckingham_pi_random_scaling,
        train=True,
        ckpt_path=model_path,
        use_two_step=False,
    )



if __name__ == "__main__":
    run_thermal()


