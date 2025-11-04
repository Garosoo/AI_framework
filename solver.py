from direct_solver import Heat2DSolver
from typing import Dict, Any


def solve(problem: Dict[str, Any]):
    for k in ('k','q','T_bc','dirichlet_mask','neumann_mask','dx','dy','reg'):
        if k not in problem:
            raise KeyError(f"Problem missing key: {k}")

    solver = Heat2DSolver(dx=problem['dx'], dy=problem['dy'], reg=problem['reg'], device='cpu')
    return solver.solve(
        k=problem['k'], q=problem['q'],
        T_bc=problem['T_bc'],
        dirichlet_mask=problem['dirichlet_mask'],
        neumann_mask=problem['neumann_mask'],
        Q_bc=problem.get('Q_bc', None),
    )
