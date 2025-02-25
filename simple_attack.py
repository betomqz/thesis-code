import utils
import logging
from datetime import datetime
from attack import Attack, OptimusAttack, SciPyAttack, Dist
import numpy as np
from pathlib import Path


SHOW_CONSOLE = True


# Create the path to save results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
save_path = Path('tmp/').joinpath(timestamp)
save_path.mkdir(parents=True, exist_ok=True)

# Logger setup. Write everything to a file (clear on each run) and only
# show warnings in console if SHOW_CONSOLE is True
logger_dir = save_path.joinpath('simple_attack.log')
logger_format = '%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s'
logging.basicConfig(
    filename=logger_dir,
    level=logging.INFO,
    format=logger_format
)

if SHOW_CONSOLE:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(logger_format))
    logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

# Load model and data
logger.info('Loading model and data')
model = utils.load_mnist_model()
x_train, x_test, y_train, y_test = utils.load_mnist_data()

# Choose inputs
inputs = [
    (0, x_test[3]),
    (1, x_test[2]),
    (2, x_test[1]),
    (3, x_test[18]),
    (4, x_test[4]),
    (5, x_test[8]),
    (6, x_test[11]),
    (7, x_test[0]),
    (8, x_test[61]),
    (9, x_test[7])
]

# Initial guess for the method. Fixed here so that results are reproducible.
np.random.seed(7679448)
random_guess = np.random.rand(784)

def simple_attack(attacker: Attack,
                  original_input,
                  original_class,
                  target_class,
                  close_to_target=False):
    '''
    Perform an attack with a customizable initial guess strategy.
    '''
    logger.info("START")
    if not close_to_target:
        initial_guess = random_guess
    else:
        # Start close to the target
        initial_guess = np.clip(
            inputs[original_class][1].flatten() + 0.1 * random_guess,
            0., 1.
        )

    attacker.attack(
        original_input=original_input,
        original_class=original_class,
        target_class=target_class,
        initial_guess=initial_guess
    )
    attacker.save(path=save_path)
    
    # See if test passed and log
    result = utils.eval_flat_pred(attacker.res['x'], model=model)
    if result == target_class:
        logger.info("The attack was successful")
    else:
        logger.warning("The attack was not successful")
    logger.info("END")



if __name__ == "__main__":

    # attacker = SciPyAttack(
    #     model,
    #     distance=Dist.L2,
    #     method='L-BFGS-B',
    #     options={'maxiter':2000, 'disp':0}
    # )
    # path = 'results/tests/attack/scipy/close/L2'

    logger.info("START")
    attacker = OptimusAttack(
        model,
        distance=Dist.LINF,
        maxiters_bs=5,
        c_right=2.0,
        maxiters_method=500
    )

    og_class, og_input = inputs[0]
    simple_attack(
        attacker=attacker,
        original_input=og_input,
        original_class=og_class,
        target_class=1,
        close_to_target=True
    ) 
    