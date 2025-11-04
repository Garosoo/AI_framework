import numpy as np
import torch
from typing import Literal, Optional


def check_inputs(k, q, T_bc, dm, nm, Q_bc, dx, dy, mode="cg"):
    dtype = torch.float64 if mode == "cg" else k.dtype
    dev   = k.device
    toD   = lambda t: t.to(device=dev, dtype=dtype)

    k, q, T_bc, Q_bc = map(toD, (k, q, T_bc, Q_bc))
    dm, nm = dm.to(dev), nm.to(dev)

    def info(name, t):
        print(f"{name:10s} shape={tuple(t.shape)} dtype={t.dtype} "
              f"finite={bool(torch.isfinite(t).all())} "
              f"min={t.min().item():.3e} max={t.max().item():.3e}")

    info("k", k); info("q", q); info("T_bc", T_bc); info("Q_bc", Q_bc)
    print(f"dm True={dm.sum().item()}, nm True={nm.sum().item()}, overlap={(dm & nm).any().item()}")

    assert not (dm & nm).any(), "Dirichlet and Neumann are overlapped which is strictly forbidden."
    H, W = k.shape[-2:]
    boundary = torch.zeros_like(dm)
    boundary[..., 0, :] = boundary[..., -1, :] = True
    boundary[..., :, 0] = boundary[..., :, -1] = True
    assert bool((nm & (~boundary)).any() == False), "Neumann is only for boundary regions."
 
    kmin = k.min().item()
    assert kmin > 0, f"k has 0 or negative values: min={kmin}"

    L = max((H-1)*dx, (W-1)*dy)
    k0 = torch.median(k.flatten()).item()
    q0 = torch.max(torch.abs(q)).item()
    T_scale = q0 * (L**2) / max(k0, 1e-30)
    print(f"Est. T scale ~ {T_scale:.3e}  (FP64 is recommended: {T_scale>1e5})")


def _harmonic_mean(a, b, eps=1e-12):
    return 2 * a * b / (a + b + eps)


class Heat2DSolver:
    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        mode: Literal["direct", "cg"] = "direct",
        max_iter: int = 10000,
        tol: float = 1e-6,
        reg: float = 1e-8,
        device: Optional[str] = None,
        dtype=torch.float32,
    ):
        self.dx, self.dy = dx, dy
        self.mode, self.max_iter, self.tol = mode, max_iter, tol
        self.reg = reg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    def solve(
        self,
        k: torch.Tensor,
        q: torch.Tensor,
        T_bc: torch.Tensor,
        dirichlet_mask: torch.Tensor,
        neumann_mask: torch.Tensor,
        Q_bc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        dtype = self.dtype
        device = self.device
        k = k.to(device=device, dtype=dtype)
        q = q.to(device=device, dtype=dtype)
        T_bc = T_bc.to(device=device, dtype=dtype)
        Q_bc = (Q_bc if Q_bc is not None else torch.zeros_like(q)).to(device=device, dtype=dtype)
        dm, nm = dirichlet_mask.to(device), neumann_mask.to(device)


        if self.mode == "direct":
            return self._solve_direct(k, q, T_bc, dm, nm, Q_bc)
        elif self.mode == "cg":
            return self._solve_cg(k, q, T_bc, dm, nm, Q_bc)
        else:
            raise ValueError("mode must be 'direct' or 'cg'")

    def _solve_direct(self, k, q, T_bc, dm, nm, Q_bc):
        B, H, W = k.shape
        N = H * W
        dx2, dy2 = self.dx**2, self.dy**2

        ke = torch.zeros_like(k); kw = torch.zeros_like(k)
        kn = torch.zeros_like(k); ks = torch.zeros_like(k)
        hm = _harmonic_mean
        ke[:, :, :-1] = hm(k[:, :, :-1], k[:, :, 1:])
        kw[:, :, 1:]  = ke[:, :, :-1]
        kn[:, 1:, :]  = hm(k[:, 1:, :], k[:, :-1, :])
        ks[:, :-1, :] = kn[:, 1:, :]

        A = torch.zeros(B, N, N, device=k.device, dtype=k.dtype)
        b = torch.zeros(B, N,      device=k.device, dtype=k.dtype)

        for bidx in range(B):
            b[bidx] = -q[bidx].reshape(N)

            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    if dm[bidx, i, j] or nm[bidx, i, j]:
                        continue
                    ce = ke[bidx, i, j] / dx2
                    cw = kw[bidx, i, j] / dx2
                    cn = kn[bidx, i, j] / dy2
                    cs = ks[bidx, i, j] / dy2
                    A[bidx, idx, idx] = -(ce + cw + cn + cs)
                    if j+1 < W: A[bidx, idx, idx+1] =  ce
                    if j-1 >=0: A[bidx, idx, idx-1] =  cw
                    if i+1 < H: A[bidx, idx, idx+W] =  cs
                    if i-1 >=0: A[bidx, idx, idx-W] =  cn

            flat_Tbc = T_bc[bidx].reshape(N)
            for idx in (dm[bidx].flatten() == True).nonzero(as_tuple=False).view(-1):
                A[bidx, idx, :] = 0
                A[bidx, idx, idx] = 1
                b[bidx, idx]      = flat_Tbc[idx]

            flat_Qbc = Q_bc[bidx]
            for i in range(H):
                for j in range(W):
                    if not nm[bidx, i, j]:
                        continue
                    idx = i * W + j
                    if j == 0:
                        neigh = idx + 1; coeff = ke[bidx, i, j] / self.dx
                    elif j == W-1:
                        neigh = idx - 1; coeff = kw[bidx, i, j] / self.dx
                    elif i == 0:
                        neigh = idx + W; coeff = ks[bidx, i, j] / self.dy
                    else:
                        neigh = idx - W; coeff = kn[bidx, i, j] / self.dy

                    A[bidx, idx, :] = 0
                    A[bidx, idx, idx]   = -coeff
                    A[bidx, idx, neigh] =  coeff
                    b[bidx, idx]        = flat_Qbc[i, j]

        if self.reg > 0:
            eye = torch.eye(N, device=k.device, dtype=k.dtype).unsqueeze(0)
            interior = ~(dm|nm).reshape(B, N)
            A = A + eye * (self.reg * interior.unsqueeze(-1).to(k.dtype))

        T_vec = torch.linalg.solve(A, b)
        return T_vec.view(B, H, W)


    def _solve_cg(self, k, q, T_bc, dm, nm, Q_bc):
        B, H, W = k.shape
        dx2, dy2 = self.dx**2, self.dy**2
        hm = _harmonic_mean

        ke = torch.zeros_like(k); kw = torch.zeros_like(k)
        kn = torch.zeros_like(k); ks = torch.zeros_like(k)
        ke[:, :, :-1] = hm(k[:, :, :-1], k[:, :, 1:]); kw[:, :, 1:] = ke[:, :, :-1]
        kn[:, 1:, :]  = hm(k[:, 1:, :], k[:, :-1, :]); ks[:, :-1, :] = kn[:, 1:, :]

        def A_times(X):
            lap = (
                (ke * torch.roll(X, -1, 2) + kw * torch.roll(X, 1, 2)) / dx2 +
                (kn * torch.roll(X, 1, 1) + ks * torch.roll(X, -1, 1)) / dy2 -
                ((ke + kw) / dx2 + (kn + ks) / dy2) * X
            )
            lap[dm] = X[dm]

            for bidx in range(B):
                for i in range(H):
                    for j in range(W):
                        if not nm[bidx, i, j]:
                            continue
                        idx = i * W + j
                        if j == 0:
                            neigh = (bidx, i, j+1); coeff = ke[bidx, i, j]/self.dx
                        elif j == W-1:
                            neigh = (bidx, i, j-1); coeff = kw[bidx, i, j]/self.dx
                        elif i == 0:
                            neigh = (bidx, i+1, j); coeff = ks[bidx, i, j]/self.dy
                        else:
                            neigh = (bidx, i-1, j); coeff = kn[bidx, i, j]/self.dy

                        lap[bidx, i, j] = -coeff * X[bidx, i, j] + coeff * X[neigh]

            return lap

        X = T_bc.clone()
        b = -q.clone()
        b[dm] = T_bc[dm]
        b[nm] = Q_bc[nm]

        r = b - A_times(X)
        p = r.clone()
        rs = (r * r).flatten(1).sum(1, keepdim=True)

        for _ in range(self.max_iter):
            Ap = A_times(p)
            alpha = rs / (p * Ap).flatten(1).sum(1, keepdim=True)
            X = X + alpha.view(B, 1, 1) * p
            r = r - alpha.view(B, 1, 1) * Ap
            rs_new = (r * r).flatten(1).sum(1, keepdim=True)
            if (rs_new.sqrt() < self.tol).all():
                break
            p = r + (rs_new / rs).view(B, 1, 1) * p
            rs = rs_new
        return X


def solve(inp: torch.Tensor, dx: float = 1e-3, dy: float = 1e-3, idx: int = 0):

    B, ch, H, W = inp.shape
    k    = inp[:, 2, :, :]
    T_bc = inp[:, 3, :, :]
    q    = inp[:, 4, :, :]

    dirichlet_mask = torch.ones((B, H, W), dtype=torch.bool, device=inp.device)
    dirichlet_mask[:, 1:-1, 1:-1]= False

    neumann_mask = torch.zeros_like(dirichlet_mask)
    Q_bc = torch.zeros_like(q)

    solver_d = Heat2DSolver(dx, dy, mode="direct")

    T_direct = solver_d.solve(k, q, T_bc, dirichlet_mask, neumann_mask, Q_bc)
    return T_direct


class ThermalFEMSolver:
    def solve(self, k, T_bc, q, dx: float = 1.0, dy: float = 1.0):
        H, W = k.shape
        dirichlet_mask = torch.ones((1, H, W), dtype=torch.bool, device=k.device)
        dirichlet_mask[:, 1:-1, 1:-1] = False
        neumann_mask = torch.zeros_like(dirichlet_mask)
        Q_bc = torch.zeros_like(q)
        solver = Heat2DSolver(dx, dy)
        T_direct = solver.solve(k.unsqueeze(0), q.unsqueeze(0), T_bc.unsqueeze(0), dirichlet_mask, neumann_mask, Q_bc)
        return T_direct

    def solve_batch(self, sds, dx: float = 1e-3, dy: float = 1e-3):
        batch_inputs = []
        for sd in sds:
            inputs = torch.stack([sd.k_in, sd.T_in, sd.q_in])
            batch_inputs.append(inputs)
        batch_inputs = torch.stack(batch_inputs)

        B, ch, H, W = batch_inputs.shape
        k    = batch_inputs[:, 0]
        T_bc = batch_inputs[:, 1]
        q    = batch_inputs[:, 2]

        dirichlet_mask = torch.ones((B, H, W), dtype=torch.bool, device=batch_inputs.device)
        dirichlet_mask[:, 1:-1, 1:-1]= False

        neumann_mask = torch.zeros_like(dirichlet_mask)
        Q_bc = torch.zeros_like(q)

        solver_d = Heat2DSolver(dx, dy, mode="direct")
        T_direct = solver_d.solve(k, q, T_bc, dirichlet_mask, neumann_mask, Q_bc)
        self.update_sds(sds, T_direct)

    def update_sds(self, sds, t_out):
        for i, sd in enumerate(sds):
            sd.T_out = t_out[i].clone()


def make_dummy_inputs(B, H, W, device=None):
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    shape = (B, H, W)
    k = torch.ones(shape, dtype=torch.double, device=dev)
    q = torch.zeros(shape, dtype=torch.double, device=dev)
    T_bc = torch.zeros(shape, dtype=torch.double, device=dev)
    dm = torch.ones(shape, dtype=torch.bool, device=dev)
    dm[:, 1:-1, 1:-1] = False
    nm = torch.zeros(shape, dtype=torch.bool, device=dev)
    Q_bc = torch.zeros(shape, dtype=torch.double, device=dev)
    return {
        "k": k,
        "q": q,
        "T_bc": T_bc,
        "dirichlet_mask": dm,
        "neumann_mask": nm,
        "Q_bc": Q_bc
    }


def make_input(B=1, H=32, W=32):
    k = torch.ones(B, H, W) * 150.0
    q = torch.zeros(B, H, W)

    h0, h1 = H // 3, 2 * H // 3              
    w0, w1 = W // 3, 2 * W // 3              
    q[:, h0:h1, w0:w1] = 0 

    T_bc = torch.zeros(B, H, W)
    T_bc[:, :, 0] = 320.0
    T_bc[:, :, -1] = 300.0
    T_bc[:, 0, :] = 350.0
    T_bc[:, -1, :] = 390.0

    dirichlet_mask = torch.full((B, H, W), True, dtype=torch.bool)
    dirichlet_mask[:, 1:-1, 1:-1] = False
    neumann_mask = torch.full((B, H, W), False, dtype=torch.bool)
    Q_bc = torch.full((B, H, W), 0, dtype=torch.float32)
    return k, q, T_bc, dirichlet_mask, neumann_mask, Q_bc


def run_demo():
    from pdeflow.utils.visualize import plot_data
    from pdeflow.solver.direct_solver_gpu import Heat2DSolver
    dx = dy = 1e-3
    inputs = make_input()
    Td = Heat2DSolver(dx,dy,"direct").solve(*inputs)
    Tg = Heat2DSolver(dx, dy, "cg", max_iter=1000, tol=1e-8).solve(*inputs)
    Tf = Heat2DSolver(dx, dy).solve(*inputs)

    print(Td.mean(), Tg.mean(), Tf.mean())
    print("max|Td-Tg|=", (Td-Tg).abs().max().item())

    k, q, T_bc, dirichlet_mask, neumann_mask, Q_bc = make_input()
    i = 0
    plot_data([
        [("k", k[i]),
         ("q", q[i]),
         ("T_bc", np.where(dirichlet_mask[0], T_bc[i], np.nan)),
         ("Q_bc", np.where(neumann_mask[0], Q_bc[i], np.nan)),],
        [("Td", Td[i]),
        ("Tcg", Tg[i]),
        ("Tpcg", Tf[i]),
        None,]
    ])


if __name__=="__main__":
    run_demo()
