'''
Comparison of OptimusAttack (line-search SQP, src/opt_attack/optimus.py::ls_sqp)
against SciPyAttack (L-BFGS-B) on a single adversarial-example task.

Both attackers are run with `singleton_attack` from the *same* random initial
guess on the same `(a, t, c, formulation, norm)`. The point is to show that the
objective-function values they land on are comparable — not that the resulting
images are pixel-identical.

Results are written to `logs/comparison/` and never touch `logs/experiments/`.
'''
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
import shutil
import argparse
import logging
from keras import models
from pathlib import Path
from opt_attack.attack import SciPyAttack, OptimusAttack, Dist, SUCCESS, ObjectFun
from opt_attack import utils
from tqdm import tqdm


SHOW_CONSOLE = False
LOGGER_FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s'

logger = logging.getLogger()


@dataclass(frozen=True)
class Experiment():
    a: int
    t: int
    c: float
    formulation: ObjectFun = ObjectFun.szegedy
    norm: Dist = Dist.L2
    attacker_name: str = 'SciPyAttack'

    def __str__(self):
        return '-'.join([
            self.attacker_name,
            self.norm.name,
            self.formulation.name,
            str(self.a),
            str(self.t),
            f'{self.c:.2f}'
        ])

    def get_result_as_row(self, attack_result, distance_res, nits, fun_val,
                          wall_seconds, x_min, x_max):
        row = self.__dict__.copy()
        row['norm'] = self.norm.name
        row['formulation'] = self.formulation.name
        row['success'] = attack_result == SUCCESS
        row['distance'] = distance_res
        row['nits'] = nits
        row['fun_val'] = fun_val
        row['wall_seconds'] = wall_seconds
        row['x_min'] = x_min
        row['x_max'] = x_max
        return row


PARENT_DIR = Path('logs/comparison/')
RESULTS_CSV = PARENT_DIR.joinpath('results.csv')

# Same seed as experiments.py so the random init lines up across files
np.random.seed(7679448)
RANDOM_GUESS = np.random.rand(784)

# The (a, t) pair and c values fixed by the plan
A = 3
T = 8
C_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]
FORMULATION = ObjectFun.szegedy
NORM = Dist.L2

# OptimusAttack params: extend the working setup from
# test_optimus_szegedy_bin_search_L2 (maxiters=50, tol=0.1) with a larger
# iteration budget because the random init is harder than near-source.
OPTIMUS_KWARGS = dict(
    distance=NORM,
    maxiters_method=100,
    eta=0.4,
    tau=0.7,
    tol=0.1,
)

SCIPY_KWARGS = dict(
    distance=NORM,
    method='L-BFGS-B',
    options={'maxiter': 2000, 'disp': 0},
)


def write_row_to_csv(row):
    row_df = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        row_df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        row_df.to_csv(RESULTS_CSV, mode='w', header=True, index=False)


def clean_outputs():
    shutil.rmtree(PARENT_DIR, ignore_errors=True)
    PARENT_DIR.mkdir(parents=True)


def get_done_exps():
    df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    done_exps = set()
    for _, row in df.iterrows():
        exp = Experiment(
            a=int(row['a']),
            t=int(row['t']),
            c=float(row['c']),
            formulation=ObjectFun[row['formulation']],
            norm=Dist[row['norm']],
            attacker_name=row['attacker_name'],
        )
        done_exps.add(hash(exp))
    return done_exps


def run_one(attacker, exp, og_input, done_exps, pbar):
    h_exp = hash(exp)
    pbar.set_postfix_str(f'Exp: {exp}')

    if h_exp in done_exps:
        logger.warning(f'Skipping experiment {exp} because it has already been run.')
        pbar.update(1)
        return

    logger.info(f'Running experiment {exp}')
    t_0 = time.perf_counter()
    result = attacker.singleton_attack(
        original_input=og_input,
        original_class=exp.a,
        target_class=exp.t,
        initial_guess=RANDOM_GUESS,
        obj_fun=exp.formulation,
        c=exp.c,
    )
    wall_seconds = time.perf_counter() - t_0

    x = attacker.res['x']
    row = exp.get_result_as_row(
        attack_result=result,
        distance_res=exp.norm.compute_vec(og_input.flatten(), x),
        nits=attacker.res['nit'],
        fun_val=float(attacker.res['fun']) if attacker.res['fun'] is not None else None,
        wall_seconds=wall_seconds,
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
    )
    write_row_to_csv(row)

    fname = PARENT_DIR.joinpath(f'{exp}.png')
    utils.save_flat_mnist_fig(x, fname=fname)

    done_exps.add(h_exp)
    pbar.update(1)


def run_comparison(scipy_only=False, optimus_only=False):
    done_exps = get_done_exps()
    inputs = utils.get_inputs_tuples()
    og_class, og_input = inputs[A]
    assert og_class == A, f'inputs[{A}] has class {og_class}, expected {A}'

    softmaxmodel = models.load_model('models/softmaxmnist.keras')
    non_softmaxmodel = utils.load_mnist_model()
    model = softmaxmodel if FORMULATION.needs_softmax() else non_softmaxmodel

    run_scipy = not optimus_only
    run_optimus = not scipy_only

    runs_per_c = int(run_scipy) + int(run_optimus)
    total = runs_per_c * len(C_VALUES)
    with tqdm(total=total, desc='Running comparison') as pbar:
        for c in C_VALUES:
            if run_scipy:
                scipy_attacker = SciPyAttack(model, **SCIPY_KWARGS)
                exp_scipy = Experiment(
                    a=A, t=T, c=c,
                    formulation=FORMULATION, norm=NORM,
                    attacker_name='SciPyAttack',
                )
                run_one(scipy_attacker, exp_scipy, og_input, done_exps, pbar)

            if run_optimus:
                optimus_attacker = OptimusAttack(model, **OPTIMUS_KWARGS)
                exp_optimus = Experiment(
                    a=A, t=T, c=c,
                    formulation=FORMULATION, norm=NORM,
                    attacker_name='OptimusAttack',
                )
                run_one(optimus_attacker, exp_optimus, og_input, done_exps, pbar)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare OptimusAttack (ls_sqp) vs SciPyAttack (L-BFGS-B) '
                    'on a fixed adversarial-example task.'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete all previous comparison results before running.'
    )
    parser.add_argument(
        '--scipy-only',
        action='store_true',
        help='Run only the SciPy (L-BFGS-B) half of the comparison.'
    )
    parser.add_argument(
        '--optimus-only',
        action='store_true',
        help='Run only the Optimus (ls_sqp) half of the comparison.'
    )
    args = parser.parse_args()

    if args.scipy_only and args.optimus_only:
        parser.error('--scipy-only and --optimus-only are mutually exclusive.')

    if args.clean:
        clean_outputs()
    else:
        PARENT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=PARENT_DIR.joinpath('comparison.log'),
        filemode='w',
        level=logging.DEBUG,
        format=LOGGER_FORMAT,
    )

    if SHOW_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logging.getLogger().addHandler(console_handler)

    run_comparison(
        scipy_only=args.scipy_only,
        optimus_only=args.optimus_only,
    )
