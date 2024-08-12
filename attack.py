import utils

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

class Attack:

    def __init__(self, model, distance='L2'):
        self.model = model
        self.distance = distance

        # Bounds of the variables
        self.bounds = [(0.,1.)]*784
        
        self.c = None
        self.input = None
        self.input_class = None
        self._input_tensor = None
        self.target = None
        self.res = None


    def _fun(self, x):
        '''
        Private function for SciPy's minimize().

        Inputs:
        - x: ndarray of shape (784,) that is proposed as Adversarial Example

        Outputs:
        - val: distance from x to the initial input plus c times $f_6(x)$.
        - gradients: the gradients of f(x).
        '''

        # Verify entry
        assert type(x) == np.ndarray, f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: entry must be ndarray"
        assert x.shape == (784,), f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: wrong shape. Must be (784,)"

        # Convert the starting point from ndarray to tensor
        x_t = tf.convert_to_tensor(x.reshape(-1,28,28,1), dtype=tf.float32)
        x_t = tf.Variable(x_t, trainable=True)                

        # Calculate loss and gradients
        with tf.GradientTape() as tape:
            tape.watch(x_t)
            pred = self.model(x_t)[0]
            mask = tf.ones_like(pred, dtype=tf.bool)
        
            # Set the t-th entry of the mask to False
            mask = tf.tensor_scatter_nd_update(mask, indices=[[self.target]], updates=[False])
            
            # Apply the mask to the tensor pred
            masked_pred = tf.boolean_mask(pred, mask)
            
            # Find the maximum value of the masked tensor
            max_z = tf.reduce_max(masked_pred)
            
            # Get distance from x0_t to x_t
            if self.distance == 'L2':
                d = tf.sqrt(tf.reduce_sum(tf.square(x_t - self._input_tensor))) # L2 norm
            elif self.distance == 'L1':
                d = tf.reduce_sum(tf.abs(x_t - self._input_tensor)) # L1 norm
            else:
                d = tf.reduce_max(tf.abs(x_t - self._input_tensor)) #L_infty norm
            val = d + self.c * tf.nn.relu(max_z - pred[self.target])        

        gradients = tape.gradient(val, x_t).numpy().flatten()
        val = val.numpy()

        return val, gradients
    
    
    def save(self, path):
        '''
        Function to save the result of the optimization problem and other variables.
        '''
        raise NotImplementedError(f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: save() method not implemented.")

    def attack(self):
        raise NotImplementedError(f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: attack() method not implemented.")


class SciPyAttack(Attack):

    def attack(self, input, input_class, target, method, x0, right=1e00, left=1e-02, tol=1e-5, maxiters_binary=20, maxiters_scipy=750):
        # Clear previous result
        self.res = None
        
        # Prepare attack
        self.input = input
        self.input_class = input_class
        self._input_tensor = tf.convert_to_tensor(self.input.reshape(-1,28,28,1), dtype=tf.float32)
        self.target = target

        # Options for each method
        options = {}
        if method == 'trust-constr':
            options={
                'maxiter':maxiters_scipy, 
                'verbose':0
            }
        elif method == 'L-BFGS-B':
            options={
                'maxiter':maxiters_scipy, 
                'disp':0
            }
        elif method == 'TNC':
            options={
                'disp':False
            }
        else:
            print(f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: {method} method unknown.")
            return

        # Evaluate on the right
        self.c = right
        # print(f"c: {self.c}")
        res = minimize(
            fun=self._fun, x0=x0, 
            method=method, 
            bounds=self.bounds, jac=True, 
            options=options
        )

        # If res.x on the right isn't classified as the target, there was an error
        if utils.eval_flat_pred(res.x, self.model) != self.target:
            print(f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: c={right} didn't work.")
            return

        # Evaluate on the left
        self.c = left
        # print(f"c: {self.c}")
        res = minimize(
            fun=self._fun, x0=x0, 
            method=method, 
            bounds=self.bounds, jac=True, 
            options=options
        )

        # print("Finding c, iter: ", end="")
        # If res.x on the left is classified as the target, return
        if utils.eval_flat_pred(res.x, self.model) == self.target:        
            self.res = res
            return

        count = 0
        # If right is correct and left wrong:
        while right - left > tol and count < maxiters_binary:
            mid = (left+right) / 2
            self.c = mid
            # print(f"{count+1}", end=",")
            aux_c = ("{0:.2f}").format(self.c)
            print(f"\rPerforming binary search. c:{aux_c}, iter: {count}", end="\r")
            sys.stdout.flush()
            res = minimize(
                fun=self._fun, x0=x0, 
                method=method, 
                bounds=self.bounds, jac=True, 
                options=options
            )

            # If attack succeeds, move right to the middle. If it doesn't,
            # move the left to the middle
            if utils.eval_flat_pred(res.x, self.model) == self.target:
                right = mid
            else:
                left = mid
            count += 1

        self.res = res
        print("")

    def save(self, path, visualize=False):
        '''
        Function to save the result of the optimization problem and other variables.
        If `visualize` is set to `True`, it shows the image.
        '''
        if self.res is None:
            print(f"{utils.TextColors.FAIL}ERROR{utils.TextColors.ENDC}: cannot save empty result.")
            return
        
        with open(f'{path}/{self.input_class}-to-{self.target}.npy', 'wb') as f:
            np.save(f, self.res.x)
            np.save(f, self.res.fun)
            np.save(f, self.res.nit)
            np.save(f, self.c)

        temp = self.res.x.reshape(28,28,1)    
        fig, ax = plt.subplots()
        ax.set_axis_off()
        
        plt.imshow(temp, cmap='gray_r')
        plt.savefig(fname=f'{path}/{self.input_class}-to-{self.target}.png', 
                    format='png', 
                    pad_inches=0, 
                    bbox_inches='tight')
        
        if visualize:
            plt.show()
