from dataclasses import dataclass
import numpy as np
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
            formulation,
            str(a),
            str(t),
            f'{self.c:.2f}'
        ])


# Directory to store results
PARENT_DIR = Path('logs/experiments/')
RESULTS_CSV = PARENT_DIR.joinpath('results.csv')
RESULTS_LATEX = PARENT_DIR.joinpath('results.tex')

# Choose a random starting point with seed for reproducibility
np.random.seed(7679448)
RANDOM_GUESS = np.random.rand(784)

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
for distance in [Dist.L2]:
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
        print(f'Attack successful: {result==SUCCESS}')
        dist_res = exp.norm.compute_vec(og_input.flatten(), attacker.res['x'])
        print(f'Distance: {dist_res}')

        # save the image
        fname = PARENT_DIR.joinpath(f'{exp}.png')
        utils.save_flat_mnist_fig(attacker.res['x'], fname=fname)


