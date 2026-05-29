import time
import numpy as np
import numpy.testing as npt
import argparse
import counters as cn
import benchmarkres as bmr
import logging

from tqdm import tqdm
from opt_attack.optimus import ls_sqp
from scipy.optimize import rosen, rosen_der, rosen_hess, minimize, Bounds

logger = logging.getLogger()

## Single-run helpers
def run_ls_sqp(n: int, tol: np.float64, hessian, maxiters=200):
    cntr = cn.Counters()
    fun = cn.make_fun_with_counts(cntr)
    restr = cn.make_restr_with_counts(cntr)

    x_0 = np.array([2.] * n)
    lam_0 = np.ones(2 * n)

    t_0 = time.perf_counter()

    x, lam = ls_sqp(
        fun=fun,
        restr=restr,
        x_0=x_0,
        lam_0=lam_0,
        B_0=np.eye(n),
        hessian=hessian,
        eta=0.4,
        tau=0.7,
        maxiters=maxiters,
        tol=tol
    )

    dt = time.perf_counter() - t_0

    f = rosen(x)
    g = rosen_der(x)
    grad_norm = np.linalg.norm(g, ord=2)

    # assert the solution is correct
    npt.assert_allclose(x, np.ones(n), atol=5e-2)

    return bmr.BenchRes(
        solver='ls_sqp',
        n=n,
        hessian=hessian if isinstance(hessian, str) else 'exact',
        time=dt,
        f_final=f,
        grad_norm=grad_norm,
        cntr=cntr,
        success=True
    )

def run_scipy(n: int, method: str, use_exact_hess=False, maxiters=200):
    _use_exact_hess = (method == 'trust-constr' and use_exact_hess)
    cntr = cn.Counters()
    fun = cn.make_scipy_fun(cntr)
    jac = cn.make_scipy_jac(cntr)

    x_0 = np.array([2.] * n)
    bounds = Bounds(0.0, 1.0)

    options = { 'maxiter': maxiters, 'disp': False }
    kwargs = {'hess': rosen_hess} if _use_exact_hess else {}

    t_0 = time.perf_counter()
    res = minimize(fun=fun, x0=x_0, method=method, jac=jac, bounds=bounds, options=options, **kwargs)
    dt = time.perf_counter() - t_0

    grad_norm = np.linalg.norm(rosen_der(res.x), ord=2)

    return bmr.BenchRes(
        solver=f'scipy-{method}',
        n=n,
        hessian='exact' if _use_exact_hess else 'approx',
        time=dt,
        f_final=res.fun,
        grad_norm=grad_norm,
        cntr=cntr,
        success=res.success
    )

# Benchmark ls_sqp
def benchmark_lssqp(n_values, repeats=3, tol=5e-3, maxiters=60):
    hessians = ['BFGS', 'L-BFGS', rosen_hess]

    total = len(n_values) * repeats * len(hessians)
    with tqdm(total=total, desc="Running ls_sqp benchmark") as pbar:
        for n in n_values:
            for i in range(repeats):
                for hessian in hessians:
                    _hessian = hessian if isinstance(hessian, str) else 'exact'
                    pbar.set_postfix_str(f'n: {n}, hessian: {_hessian}, i: {i}')
                    logger.info(f'Starting ls_sqp benchmark for n: {n}, hessian: {_hessian}, i: {i}')
                    try:
                        bench_res = run_ls_sqp(n=n, tol=tol, hessian=hessian, maxiters=maxiters)
                    except AssertionError:
                        bench_res = bmr.BenchRes(
                            solver='ls_sqp',
                            n=n,
                            hessian=_hessian,
                            time=np.nan,
                            f_final=np.nan,
                            grad_norm=np.nan,
                            cntr=cn.Counters(),
                            success=False
                        )
                    logger.info(f'End ls_sqp benchmark')
                    bench_res.write_to_csv()
                    pbar.update(1)

# Benchmark SciPy's methods
def benchmark_scipy(n_values, repeats=3, tol=5e-3, maxiters=200):
    # hessians = ['L-BFGS', 'BFGS', rosen_hess]
    methods = [
        ('L-BFGS-B', False),
        ('SLSQP', False),
        ('trust-constr', True),
        ('trust-constr', False),
    ]

    total = len(n_values) * repeats * len(methods)
    with tqdm(total=total, desc="Running SciPy's benchmarks") as pbar:
        for n in n_values:
            for i in range(repeats):
                for method, use_exact in methods:
                    pbar.set_postfix_str(f'n: {n}, method: {method}, i: {i}')
                    logger.info(f'Starting SciPy\'s benchmark for n: {n}, method: {method}, i: {i}')
                    bench_res = run_scipy(n=n, method=method, use_exact_hess=use_exact, maxiters=maxiters)
                    logger.info(f'End ls_sqp benchmark')
                    bench_res.write_to_csv()
                    pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for the thesis')
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete all the previous results before running again'
    )
    parser.add_argument(
        '--lssqp',
        action='store_true',
        help='Benchmark ls_sqp method'
    )
    parser.add_argument(
        '--scipy',
        action='store_true',
        help='Benchmark SciPy\'s methods'
    )
    args = parser.parse_args()

    if args.clean:
        bmr.clean_outputs()
    else:
        # Make sure parent directory exists
        bmr.PARENT_DIR.mkdir(parents=True, exist_ok=True)

    # Logger setup. Write everything to a file (append)
    logging.basicConfig(
        filename=bmr.PARENT_DIR.joinpath('benchmark.log'),
        filemode='a',
        level=logging.DEBUG,
        format=bmr.LOGGER_FORMAT
    )

    n_values = [10, 20, 50, 100, 200, 500]

    if args.lssqp:
        benchmark_lssqp(n_values=n_values)

    if args.scipy:
        benchmark_scipy(n_values=n_values)
