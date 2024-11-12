import enum
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from utils import eval_flat_pred
from utils import TextColors as tc
from optimus import ls_sqp
from pathlib import Path


SUCCESS = 0
FAILURE = 1


class Dist(enum.Enum):
    LINF = 'LINF'
    L1 = 'L1'
    L2 = 'L2'


class Attack:

    def __init__(
            self,
            model,
            distance: Dist = Dist.L2,
            maxiters_bs: int = 10,
            c_left: float = 1e-02,
            c_right: float = 1.0
        ):
        '''Init method.'''
        self.model = model
        self.distance = distance
        self.maxiters_bs = maxiters_bs
        self.c_left = c_left
        self.c_right = c_right

        self.c = None
        self.original_input = None
        self.original_input_tensor = None
        self.original_class = None
        self.target_class = None
        self.initial_guess = None
        self.res = {
            'x': None,
            'fun': None,
            'nit': None
        }

    def _minimize(self) -> tuple[np.ndarray, float, int]:
        '''Executes the optimization method.

        Returns (x, fun(x), nit)
        '''
        raise NotImplementedError(
            f"{tc.FAIL}ERROR{tc.ENDC}: _minimize() method not implemented."
        )

    def attack(
            self,
            original_input,
            original_class,
            target_class,
            initial_guess
        ) -> int:
        '''Executes the binary search for the attack and stores the result.'''
        self.original_input = original_input
        self.original_class = original_class
        self.target_class = target_class
        self.initial_guess = initial_guess

        # Clear previous result
        self.res = {
            'x': None,
            'fun': None,
            'nit': None
        }
        
        # Set upper and lower bound for c
        right = self.c_right
        left = self.c_left

        # Convert original input to tensor
        self.original_input_tensor = tf.convert_to_tensor(
            self.original_input.reshape(-1,28,28,1), 
            dtype=tf.float32
        )

        # Evaluate on the right
        self.c = right
        res_x, res_fun, res_nit = self._minimize()

        # If res.x on the right isn't classified as the target, return FAILURE.
        if eval_flat_pred(res_x, self.model) != self.target_class:
            print(f"{tc.FAIL}ERROR{tc.ENDC}: c={self.c} didn't work.")
            return FAILURE
        # Else store successful result and continue
        self.res['x'] = res_x
        self.res['fun'] = res_fun
        self.res['nit'] = res_nit

        # Evaluate on the left
        self.c = left 
        res_x, res_fun, res_nit = self._minimize()

        # If res.x on the left is classified as the target, return
        if eval_flat_pred(res_x, self.model) == self.target_class:        
            self.res['x'] = res_x
            self.res['fun'] = res_fun
            self.res['nit'] = res_nit
            return SUCCESS

        count = 0
        # If right is correct and left wrong:
        while count < self.maxiters_bs:
            self.c = (left + right) / 2

            aux_c = ("{0:.2f}").format(self.c)
            print(f"\rPerforming binary search. c:{aux_c}, iter: {count}",
                  end="\r")
            sys.stdout.flush()
            
            res_x, res_fun, res_nit = self._minimize()

            # If attack succeeds, move right to the middle. If it doesn't, move
            # the left to the middle
            if eval_flat_pred(res_x, self.model) == self.target_class: 
                right = self.c
                self.res['x'] = res_x
                self.res['fun'] = res_fun
                self.res['nit'] = res_nit
            else:
                left = self.c
            count += 1

        # We can guarantee that we succeeded for at least one c
        return SUCCESS

    def _fun(
            self,
            x: np.ndarray
        ) -> tuple[float, np.ndarray]:
        '''Objective function to be minimized by _minimize.'''
        # Convert the starting point from ndarray to tensor
        x_tensor = tf.convert_to_tensor(x.reshape(-1,28,28,1), dtype=tf.float32)
        x_tensor = tf.Variable(x_tensor, trainable=True)                

        # Calculate loss and gradients
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred = self.model(x_tensor)[0]
            mask = tf.ones_like(pred, dtype=tf.bool)
        
            # Set the t-th entry of the mask to False
            mask = tf.tensor_scatter_nd_update(mask, indices=[[self.target_class]], updates=[False])
            
            # Apply the mask to the tensor pred
            masked_pred = tf.boolean_mask(pred, mask)
            
            # Find the maximum value of the masked tensor
            max_z = tf.reduce_max(masked_pred)
            
            # Get distance from x0_t to x_t
            if self.distance == Dist.L2:
                d = tf.sqrt(tf.reduce_sum(tf.square(x_tensor - self.original_input_tensor))) # L2 norm
            elif self.distance == Dist.L1:
                d = tf.reduce_sum(tf.abs(x_tensor - self.original_input_tensor)) # L1 norm
            else:
                d = tf.reduce_max(tf.abs(x_tensor - self.original_input_tensor)) #L_infty norm
            val = d + self.c * tf.nn.relu(max_z - pred[self.target_class])        

        gradients = tape.gradient(val, x_tensor).numpy().flatten()
        val = val.numpy()

        return val, gradients

    def save(self, path, visualize=False):
        '''
        Function to save the result of the optimization problem and other variables.
        If `visualize` is set to `True`, it shows the image.

        TODO: refactor this. I don't like it. But it works, I think.
        '''
        if self.res['x'] is None:
            print(f"{tc.FAIL}ERROR{tc.ENDC}: cannot save empty result.")
            return
        
        # Create the path and its parent directories if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(f'{path}/{self.original_class}-to-{self.target_class}.npy', 'wb') as f:
            np.save(f, self.res['x'])
            np.save(f, self.res['fun'])
            np.save(f, self.res['nit'])
            np.save(f, self.c)

        temp = self.res['x'].reshape(28,28,1)
        fig, ax = plt.subplots()
        ax.set_axis_off()
        
        plt.imshow(temp, cmap='gray_r')
        plt.savefig(fname=f'{path}/{self.original_class}-to-{self.target_class}.png', 
                    format='png', 
                    pad_inches=0, 
                    bbox_inches='tight')
        
        if visualize:
            plt.show()
        plt.close()


class SciPyAttack(Attack):

    def __init__(
            self,
            model,
            distance: Dist = Dist.L2,
            maxiters_bs: int = 10,
            c_left: float = 0.01,
            c_right: float = 1.0,
            method: str = 'L-BFGS-B',
            options: dict = None
        ):
        super().__init__(model, distance, maxiters_bs, c_left, c_right)
        self.method = method
        self.options = options
        self.bounds = [(0.,1.)]*784 # Maybe this shouldn't be hard-coded

    def _minimize(self) -> tuple[np.ndarray, float, int]:
        res = minimize(
            fun=self._fun,
            x0=self.initial_guess,
            method=self.method,
            bounds=self.bounds,
            jac=True,
            options=self.options
        )
        return res.x, res.fun, res.nit


class OptimusAttack(Attack):

    def __init__(
            self,
            model,
            distance: Dist = Dist.L2,
            maxiters_bs: int = 10,
            c_left: float = 0.01,
            c_right: float = 1.0,
            maxiters_method: int = 1000,
            eta: float = 0.4,
            tau: float = 0.7,
            tol: float = 1.1
        ):
        super().__init__(model, distance, maxiters_bs, c_left, c_right)
        self.maxiters = maxiters_method
        self.eta = eta
        self.tau = tau
        self.tol = tol

    def _restr(
            self,
            x: np.ndarray
        ) -> tuple[float, np.ndarray]:
        '''
        Evaluate the restrictions of the problem: c(x) >= 0.

        Since we want that 0 <= x_i <= 1 for every i, then
            c(x) = [x_1, x_2, ..., x_n, 1 - x_1, 1 - x_2, ..., 1 - x_n]
        '''
        c = np.concatenate([x, 1 - x])
        A = np.concatenate([np.eye(x.size), -np.eye(x.size)])
        return c, A

    def _minimize(self) -> tuple[np.ndarray, float, int]:
        '''Minimize using Line Search Sequential Quadratic Programming'''
        res_x, _ = ls_sqp(
            fun=self._fun,
            restr=self._restr,
            x_0=self.initial_guess,
            lam_0=np.ones(2 * self.initial_guess.size),
            B_0=np.eye(self.initial_guess.size),
            eta=self.eta,
            tau=self.tau,
            maxiters=self.maxiters,
            tol=self.tol
        )
        res_fun, _ = self._fun(res_x)
        return res_x, res_fun, -1 #TODO: return iteration count
