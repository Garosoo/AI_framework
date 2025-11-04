import os
import random
import numpy as np
import torch


def fix_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        os.environ["TF_DETERMINISTIC_OPS"] = "0"

    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

