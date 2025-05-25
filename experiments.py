from dataclasses import dataclass
import numpy as np
import pandas as pd
import shutil
import argparse
from keras import models
from pathlib import Path
from opt_attack.attack import SciPyAttack, Dist, SUCCESS
from opt_attack import utils


@dataclass(frozen=True)
class Experiment():
    a: int                          # original class
    t: int                          # target class
    c: float                        # c value
    formulation: str = 'szegedy'    # 'szegedy' or 'carlini'
    norm: Dist = Dist.L1            # L1, L2 or L3
    attacker_name: str = type(SciPyAttack).__name__  # 'optimus' or 'scipy'

    def __str__(self):
        return '-'.join([
            self.attacker_name,
            self.norm.name,
            self.formulation,
            str(self.a),
            str(self.t),
            f'{self.c:.2f}'
        ])

    def get_result_as_row(self, attack_result, distance_res, nits):
        '''Returns a dictionary with the values to be inserted as a row to de database'''
        row = self.__dict__.copy()        # Copy so we can modify it
        row['norm'] = self.norm.name      # replace it so it looks cleaner
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
            formulation=row['formulation'],
            norm=Dist[row['norm']],
            attacker_name=row['attacker_name']
        )
        done_exps.add(exp)
    return done_exps

def run_all_experiments():
    # Get the ordered tuples
    inputs = utils.get_inputs_tuples()

    # Define values for c
    c_values = [2**i for i in range(-6, 7)]
    # c_values = [1.0]

    # Load the model with the softmax fn
    softmaxmodel = models.load_model('models/softmaxmnist.keras')

    # Choose original class (a) and target (t)
    a = 4
    t = 9

    # Create attacker object - we'll use SciPy for now
    attacker = SciPyAttack(
        softmaxmodel,
        method='L-BFGS-B',
        options={'maxiter':2000, 'disp':0}
    )

    # Choose a formulation for the objective function
    formulation = 'szegedy'

    # Choose distance to be used
    for distance in Dist:
        print(f'Using distance {distance.name}')
        attacker.distance = distance

        # Original class will be a
        og_class, og_input = inputs[a]

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

            # if experiment has already been run, skip
            done_exps = get_done_exps()
            if exp in done_exps:
                print(f'Skipping experiment {exp} because it has already been run.')
                continue

            # run experiment
            print(f'Running experiment with c={c}, name={exp}')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for the thesis')
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete all the previous results before running again'
    )
    args = parser.parse_args()

    if args.clean:
        # Clean and create the directory again
        clean_outputs()
    else:
        # Make sure parent directory exists
        PARENT_DIR.mkdir(parents=True, exist_ok=True)

    run_all_experiments()