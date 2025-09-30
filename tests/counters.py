import numpy as np

from scipy.optimize import rosen, rosen_der

class Counters:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.f_evals = 0
        self.g_evals = 0
        self.c_evals = 0
        self.j_evals = 0

## Wrappers for optimus
def make_fun_with_counts(cntr: Counters):
    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        cntr.f_evals += 1
        cntr.g_evals += 1
        return rosen(x), rosen_der(x)
    return fun

def make_restr_with_counts(cntr: Counters):
    def restr(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.size
        c = np.concatenate([x, 1 - x])
        A = np.concatenate([np.eye(n), -np.eye(n)])
        cntr.c_evals += 1
        cntr.j_evals += 1
        return c, A

    return restr

## Wrappers for SciPy
def make_scipy_fun(cntr: Counters):
    def fun(x: np.ndarray) -> np.ndarray:
        cntr.f_evals += 1
        return rosen(x)
    return fun

def make_scipy_jac(cntr: Counters):
    def jac(x: np.ndarray) -> np.ndarray:
        cntr.f_evals += 1
        return rosen_der(x)
    return jac

