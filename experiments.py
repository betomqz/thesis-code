from dataclasses import dataclass
import numpy as np
import pandas as pd
import shutil
import argparse
import logging
from keras import models
from pathlib import Path
from opt_attack.attack import SciPyAttack, Dist, SUCCESS, ObjectFun
from opt_attack import utils
from tqdm import tqdm


SHOW_CONSOLE = False
LOGGER_FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s'

logger = logging.getLogger()

@dataclass(frozen=True)
class Experiment():
    a: int                          # original class
    t: int                          # target class
    c: float                        # c value
    formulation: ObjectFun = ObjectFun.szegedy # szegedy, carlini or other
    norm: Dist = Dist.L1            # L1, L2 or L3
    attacker_name: str = type(SciPyAttack).__name__  # 'optimus' or 'scipy'

    def __str__(self):
        return '-'.join([
            self.attacker_name,
            self.norm.name,
            self.formulation.name,
            str(self.a),
            str(self.t),
            f'{self.c:.2f}'
        ])

    def get_result_as_row(self, attack_result, distance_res, nits):
        '''Returns a dictionary with the values to be inserted as a row to de database'''
        row = self.__dict__.copy()        # Copy so we can modify it
        row['norm'] = self.norm.name      # replace it so it looks cleaner
        row['formulation'] = self.formulation.name
        row['success'] = attack_result == SUCCESS
        row['distance'] = distance_res
        row['nits'] = nits
        return row


# Directory to store results
PARENT_DIR = Path('logs/experiments/')
RESULTS_CSV = PARENT_DIR.joinpath('results.csv')
RESULTS_LATEX = PARENT_DIR.joinpath('results.tex')

# Choose a random starting point with seed for reproducibility
np.random.seed(7679448)
RANDOM_GUESS = np.random.rand(784)

def write_row_to_csv(row):
    '''Writes a row to the csv results'''
    row_df = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        row_df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        row_df.to_csv(RESULTS_CSV, mode='w', header=True, index=False)

def clean_outputs():
    '''Cleans the parent directory from all experiments'''
    shutil.rmtree(PARENT_DIR)
    PARENT_DIR.mkdir(parents=True)

def get_done_exps():
    '''Returns a set of Experiment objects from the df'''
    df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()

    done_exps = set()
    for _, row in df.iterrows():
        exp = Experiment(
            a=int(row['a']),
            t=int(row['t']),
            c=float(row['c']),
            formulation=ObjectFun[row['formulation']],
            norm=Dist[row['norm']],
            attacker_name=row['attacker_name']
        )
        h_exp = hash(exp)
        done_exps.add(h_exp)
    return done_exps

def run_all_experiments():
    # Load hashes of done experiments
    done_exps = get_done_exps()

    # Get the ordered tuples
    inputs = utils.get_inputs_tuples()

    # Load the model with the softmax fn
    softmaxmodel = models.load_model('models/softmaxmnist.keras')
    # Load the model without the softmax fn
    non_softmaxmodel = utils.load_mnist_model()

    # Create attacker object - we'll use SciPy for now
    attacker = SciPyAttack(
        softmaxmodel,
        method='L-BFGS-B',
        options={'maxiter':2000, 'disp':0}
    )

    # Define values for c
    c_values = [2**i for i in range(-6, 7)]

    # Choose a formulation for the objective function
    formulations = list(ObjectFun)

    # Choose distance to be used
    distances = list(Dist)

    # Choose original class (a) and target (t)
    a_and_t_values = [(a, b) for a in range(10) for b in range(10) if a != b]

    # Run all
    total = len(formulations) * len(a_and_t_values) * len(distances) * len(c_values)
    with tqdm(total=total, desc="Running experiments") as pbar:
        for formulation in formulations:
            attacker.model = softmaxmodel if formulation.needs_softmax() else non_softmaxmodel

            for a, t in a_and_t_values:
                # Original class will be a
                og_class, og_input = inputs[a]

                for distance in distances:
                    logger.info(f'Using distance {distance.name}')
                    attacker.distance = distance

                    # Run attack for each c
                    for c in c_values:

                        exp = Experiment(
                            a=og_class,
                            t=t,
                            c=c,
                            formulation=formulation,
                            norm=distance,
                            attacker_name=type(attacker).__name__
                        )
                        h_exp = hash(exp)
                        pbar.set_postfix_str(f'Exp: {exp}')

                        # if experiment has already been run, skip
                        if h_exp in done_exps:
                            logger.warning(f'Skipping experiment {exp} because it has already been run.')
                            pbar.update(1)
                            continue

                        # run experiment
                        logger.info(f'Running experiment with c={c}, name={exp}')
                        result = attacker.singleton_attack(
                            original_input=og_input,
                            original_class=exp.a,
                            target_class=exp.t,
                            initial_guess=RANDOM_GUESS,
                            obj_fun=exp.formulation,
                            c=exp.c,
                        )

                        # store experiment results in df
                        row = exp.get_result_as_row(
                            attack_result=result,
                            distance_res=exp.norm.compute_vec(og_input.flatten(), attacker.res['x']),
                            nits=attacker.res['nit']
                        )
                        write_row_to_csv(row)

                        # save the image
                        fname = PARENT_DIR.joinpath(f'{exp}.png')
                        utils.save_flat_mnist_fig(attacker.res['x'], fname=fname)

                        # save done experiment
                        done_exps.add(h_exp)
                        pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for the thesis')
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete all the previous results before running again'
    )
    args = parser.parse_args()

    # Logger setup. Write everything to a file (clear on each run) and only
    # show warnings in console if SHOW_CONSOLE is True
    logging.basicConfig(
        filename=PARENT_DIR.joinpath('experiments.log'),
        filemode='w',
        level=logging.DEBUG,
        format=LOGGER_FORMAT
    )

    if SHOW_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
        logging.getLogger().addHandler(console_handler)

    if args.clean:
        # Clean and create the directory again
        clean_outputs()
        logger.info('Directory cleaned.')
    else:
        # Make sure parent directory exists
        PARENT_DIR.mkdir(parents=True, exist_ok=True)

    run_all_experiments()
