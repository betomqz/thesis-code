import utils
from attack import OptimusAttack, Dist, SciPyAttack
import numpy as np

np.random.seed(20000325)

model = utils.load_mnist_model()

x_train, x_test, y_train, y_test = utils.load_mnist_data()

inputs = [
    x_test[3],  # 0
    x_test[2],  # 1
    x_test[1],  # 2
    x_test[18], # 3
    x_test[4],  # 4
    x_test[8],  # 5*, mejor 15?
    x_test[11], # 6
    x_test[0],  # 7
    x_test[61], # 8
    x_test[7],  # 9, mejor 16?
]

original_class = 0
original_input = inputs[original_class]
target_class = 1
initial_guess = np.clip(inputs[original_class].flatten() + 0.1 * np.random.rand(784), 0., 1.)

# attacker = OptimusAttack(model=model, maxiters_method=15)
attacker = SciPyAttack(model=model)

attacker.attack(
    original_input=original_input.flatten(),
    original_class=original_class,
    target_class=target_class,
    initial_guess=initial_guess
)

print("EVAL RESULT:")
print(utils.eval_flat_pred(attacker.res['x'], model=model))

attacker.save("tmp", True)