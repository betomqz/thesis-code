import utils
from attack import OptimusAttack, SciPyAttack, Dist
import numpy as np
import os
from pathlib import Path

# Load model and data
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

def perform_attack(attacker, save_path, close_to_target=False):
    '''
    Perform an attack with a customizable initial guess strategy.

    Parameters:
    - attacker: The attacker instance.
    - save_path: Path to save the results.
    - close_to_target: If True, initial guess will be close to the target; 
                       if False, it will use a fixed random guess.
    '''
    # Create the path and its parent directories if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    if not close_to_target:
        initial_guess = random_guess

    total_tests = 9 * len(inputs)
    tests_ran = 0
    tests_failed = 0
    print(f"Running {total_tests} tests.")

    # For each input
    for original_class, original_input in inputs:
        # For each target
        for target_class in range(10):
            # If the file already exists, don't run this test
            test_name = f'{original_class}-to-{target_class}'
            file_path = f'{save_path}/{test_name}.png'
            if os.path.exists(file_path):
                print(f'{test_name} already exists. Skipping test')
                tests_ran += 1
                continue

            # Only attack if original class is different than target
            if original_class != target_class:

                # Start close to the target
                if close_to_target:
                    initial_guess = np.clip(
                        inputs[original_class][1].flatten() + 0.1 * random_guess,
                        0., 1.
                    )

                print(f"Performing test {test_name}.")
                attacker.attack(
                    original_input=original_input,
                    original_class=original_class,
                    target_class=target_class,
                    initial_guess=initial_guess
                )
                attacker.save(path=save_path)
                
                # See if test passed and log progress
                result = utils.eval_flat_pred(attacker.res['x'], model=model)
                passed = result == target_class
                print(f"{test_name}: {'Passed' if passed else 'Failed'}")
                tests_ran += 1
                if not passed:
                    tests_failed += 1
                print(f"Progress: {tests_ran}/{total_tests} " +
                      f"({(tests_ran/total_tests)*100:.2f}%). " + 
                      f"{tests_failed} test(s) have failed.")
        print("")


if __name__ == "__main__":

    # attacker = SciPyAttack(
    #     model,
    #     distance=Dist.L2,
    #     method='L-BFGS-B',
    #     options={'maxiter':2000, 'disp':0}
    # )
    # path = 'results/tests/attack/scipy/close/L2'

    attacker = OptimusAttack(
        model,
        distance=Dist.L2,
        maxiters_method=500
    )
    path = 'results/tests/attack/optimus/close/L2'

    perform_attack(attacker, path, True)

    ## TODO fix this
    # Save each original image used
    for input_class, input in inputs:    
        utils.vis_flat_mnist(input.flatten(), 
                            save=True, 
                            filename=f'{path}/{input_class}-to-{input_class}.png', 
                            format='png')
    
    # Build big graph
    utils.big_graph(path)
