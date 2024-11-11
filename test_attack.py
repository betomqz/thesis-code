import unittest
import utils
from attack import OptimusAttack, SciPyAttack
import numpy as np

class TestAttack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(7679448)

        print("Loading MNIST model...")
        cls.model = utils.load_mnist_model()

        print("Loading MNIST data...")
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = utils.load_mnist_data()

        cls.inputs = [
            cls.x_test[3],
            cls.x_test[2],
            cls.x_test[1],
            cls.x_test[18],
            cls.x_test[4],
            cls.x_test[8],
            cls.x_test[11],
            cls.x_test[0],
            cls.x_test[61],
            cls.x_test[7]
        ]

        cls.original_class = 0
        cls.original_input = cls.inputs[cls.original_class]
        cls.target_class = 1


    def perform_attack_close(self, attack_class):
        '''Helper function to test an attack class with an initial guess close to the original class'''
        self.initial_guess = np.clip(
            self.inputs[self.original_class].flatten() + 0.1 * np.random.rand(784),
            0., 1.
        )
        
        attacker = attack_class(model=self.model)

        attacker.attack(
            original_input=self.original_input.flatten(),
            original_class=self.original_class,
            target_class=self.target_class,
            initial_guess=self.initial_guess
        )

        self.assertEqual(
            utils.eval_flat_pred(attacker.res['x'], model=self.model),
            self.target_class
        )

    def test_scipy_attack_close(self):
        '''Test SciPyAttack with an initial guess close to the original class'''
        self.perform_attack_close(SciPyAttack)


    def test_optimus_attack_close(self):
        '''Test OptimusAttack with an initial guess close to the original class'''
        self.perform_attack_close(OptimusAttack)


if __name__ == "__main__":
    unittest.main()
