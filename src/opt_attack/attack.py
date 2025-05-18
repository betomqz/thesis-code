import enum
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from keras import Model
from opt_attack.utils import eval_flat_pred
from opt_attack.optimus import ls_sqp
from pathlib import Path
from typing import Callable
import multiprocessing


SUCCESS = 0
FAILURE = 1


logger = logging.getLogger(__name__)


class Dist(enum.Enum):
    '''
    Enumeration of vector distance metrics.

    Members
    -------
    LINF : str
        Infinity norm (maximum absolute difference).
    L1 : str
        L1 norm (sum of absolute differences).
    L2 : str
        L2 norm (Euclidean distance).

    Methods
    -------
    compute_vec(x, y)
        Computes the distance between two vectors using the selected norm.
    '''
    LINF = 'LINF'
    L1 = 'L1'
    L2 = 'L2'

    def compute_vec(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the distance between two numpy vectors using the selected norm.

        Parameters
        ----------
        x : np.ndarray
            First input vector.
        y : np.ndarray
            Second input vector.

        Returns
        -------
        float
            The computed distance between the vectors.
        '''
        if self == Dist.L2:
            return np.linalg.norm(x - y, ord=2)
        elif self == Dist.L1:
            return np.linalg.norm(x - y, ord=1)
        else:
            return np.linalg.norm(x - y, ord=np.inf)

    def compute_tens(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the distance between two tensors using the selected norm. The
        calculation is done using TensorFlow to allow the use of GradientTape.

        Parameters
        ----------
        x : np.ndarray
            First input tensor.
        y : np.ndarray
            Second input tensor.

        Returns
        -------
        float
            The computed distance between the tensors.
        '''
        if self == Dist.L2:
            return tf.sqrt(tf.reduce_sum(tf.square(x - y)))
        elif self == Dist.L1:
            return tf.reduce_sum(tf.abs(x - y))
        else:
            return tf.reduce_max(tf.abs(x - y))

class Attack:
    '''
    Abstract class to generate adversarial examples for a given model.

    Attributes
    ----------
    model : Model
        Model for which the adversarial examples will be generated.

    distance : Dist
        Distance metric to compare two images.

    original_input : np.ndarray
        Original input image as a NumPy array.

    original_input_tensor : tf.Tensor
        Original input image converted to a TensorFlow tensor.

    original_class : int
        Class index predicted by the model for the original input.

    target_class : int
        Target class for the adversarial example.

    target_one_hot : tf.Tensor
        One-hot encoded target class label as a TensorFlow tensor.

    initial_guess : np.ndarray
        Initial adversarial guess to start the optimization from.

    res : dict
        Dictionary containing the results of the attack. It includes:
        - `'x'`: the adversarial example generated.
        - `'fun'`: the evaluation of the minimization function on the adversarial
        example.
        - `'nit'`: the number of iterations taken by the optimization method to
        find the example.

    Methods
    -------
    binary_search_attack()
        Perform a binary search to find the optimal constant `c` that balances
        the classification loss against the perturbation distance.  This value
        of `c` is used during optimization to generate effective adversarial
        examples.

    parallel_attack()
        Try different values for `c` using multiple processes at the same time.

    save()
        Save the adversarial example to disk.
    '''

    def __init__(
            self,
            model: Model,
            distance: Dist = Dist.L2
        ):
        '''
        Initialize the Attack object.

        Parameters
        ----------
        model : Model
            The model to attack.

        distance : Dist, optional
            Distance metric used to compare images (default is L2).

        Attributes Initialized
        ----------------------
        original_input : np.ndarray or None
            The original input image (to be set later).

        original_input_tensor : tf.Tensor or None
            TensorFlow tensor version of the original input image (to be set
            later).

        original_class : int or None
            The original predicted class of the input (to be set later).

        target_class : int or None
            The target class for the attack (to be set later).

        initial_guess : np.ndarray or None
            Initial guess for the adversarial example (to be set later).

        res : dict
            Dictionary to store attack results. Contains:
            - `'x'`: The adversarial example (or `None` if not yet computed).
            - `'fun'`: Value of the objective function at the adversarial example.
            - `'nit'`: Number of iterations used to find the adversarial example.
        '''
        self.model = model
        self.distance = distance

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

    def _minimize(self, fun: Callable, c: float) -> tuple[np.ndarray, float, int]:
        '''
        Execute the optimization method to minimize the given objective function.

        This method is intended to be overridden by subclasses. It performs the
        optimization to generate an adversarial example that minimizes a
        trade-off between classification loss and perturbation distance,
        weighted by the constant `c`.

        Parameters
        ----------
        fun : Callable
            The objective function to minimize.

        c : float
            Trade-off constant that balances classification loss and perturbation
            distance in the objective.

        Returns
        -------
        x : np.ndarray
            The adversarial example found by the optimization.

        fun_x : float
            The value of the objective function evaluated at `x`.

        nit : int
            Number of iterations used by the optimization algorithm.

        Raises
        ------
        NotImplementedError
        If the method is not implemented by a subclass.
        '''
        msg = "`_minimize()` method not implemented."
        logger.error(msg)
        raise NotImplementedError(msg)

    def singleton_attack(
            self,
            original_input: np.ndarray,
            original_class: int,
            target_class: int,
            initial_guess: np.ndarray,
            obj_fun: str,
            c: float
        ) -> int:
        '''
        Performs a single optimization attempt to generate an adversarial
        example.

        This method attempts to generate an adversarial example by minimizing
        the selected objective function using a fixed constant `c`. This method
        performs only one optimization run with the provided value. The results
        of the attack are stored in the `res` attribute.

        Parameters
        ----------
        original_input : np.ndarray
            The original input image to be perturbed.

        original_class : int
            The class predicted by the model for the original input.

        target_class : int
            The desired target class for the adversarial attack.

        initial_guess : np.ndarray
            An initial guess for the adversarial example.

        obj_fun : str
            Objective function to use in the optimization. Options are:
            - `'carlini'`: Uses Carlini's objective function.
            - `'szegedy'`: Uses Szegedy's objective function.

        c : float
            Constant that balances the importance of the objective function
            during optimization.

        Returns
        -------
        int
            Status code:
            - SUCCESS = 0
            - FAILURE = 1
        '''
        logger.info("START")
        self.original_input = original_input
        # Convert original input to tensor
        self.original_input_tensor = tf.convert_to_tensor(
            self.original_input.reshape(-1,28,28,1),
            dtype=tf.float32
        )
        self.original_class = original_class
        self.target_class = target_class
        self.initial_guess = initial_guess
        # TODO: again this 10 maybe shouldn't be hardcoded
        self.target_one_hot = tf.one_hot([self.target_class], 10)

        # Choose objective function
        if obj_fun == 'carlini':
            fun = self._fun_carlini
        elif obj_fun == 'szegedy':
            fun = self._fun_szegedy
        else:
            msg = f"`{obj_fun}` function not implemented."
            logger.error(msg)
            raise NotImplementedError(msg)

        # Clear previous result
        self.res = {
            'x': None,
            'fun': None,
            'nit': None
        }

        logger.info(f"Performing singleton attack. c:{c}")
        res_x, res_fun, res_nit = self._minimize(fun, c)

        # Store the results, even it the attack didn't succeed
        self.res['x'] = res_x
        self.res['fun'] = res_fun
        self.res['nit'] = res_nit

        logger.info("END")
        # If res.x isn't classified as the target, return FAILURE.
        if eval_flat_pred(res_x, self.model) != self.target_class:
            logger.error(f"Singleton attack was not successful")
            return FAILURE
        else:
            # Attack was successful
            logger.info(f"Singleton attack was successful!")
            return SUCCESS

    def binary_search_attack(
            self,
            original_input: np.ndarray,
            original_class: int,
            target_class: int,
            initial_guess: np.ndarray,
            obj_fun: str,
            maxiters_bs: int = 10,
            c_left: float = 1e-02,
            c_right: float = 1.0
        ) -> int:
        '''
        Execute a binary search to find the optimal constant `c` for the
        adversarial attack.

        This method searches for the best value of `c` using a binary search
        approach. The selected objective function determines the loss used
        during optimization. The results of the attack are stored in the `res`
        attribute.

        Parameters
        ----------
        original_input : np.ndarray
            The original input image to be perturbed.

        original_class : int
            The class predicted by the model for the original input.

        target_class : int
            The desired target class for the adversarial attack.

        initial_guess : np.ndarray
            An initial guess for the adversarial example.

        obj_fun : str
            Objective function to use in the optimization. Options are:
            - `'carlini'`: Uses Carlini's objective function.
            - `'szegedy'`: Uses Szegedy's objective function.

        maxiters_bs : int, optional
            Maximum number of binary search iterations (default is 10).

        c_left : float, optional
            Lower bound of the binary search interval for the constant `c`
            (default is 1e-2).

        c_right : float, optional
            Upper bound of the binary search interval for the constant `c`
            (default is 1.0).

        Returns
        -------
        int
            Status code:
            - SUCCESS = 0
            - FAILURE = 1

        Raises
        ------
        ValueError
            If an unsupported objective function is provided.
        '''
        logger.info("START")
        self.original_input = original_input
        # Convert original input to tensor
        self.original_input_tensor = tf.convert_to_tensor(
            self.original_input.reshape(-1,28,28,1),
            dtype=tf.float32
        )
        self.original_class = original_class
        self.target_class = target_class
        self.initial_guess = initial_guess
        # TODO: again this 10 maybe shouldn't be hardcoded
        self.target_one_hot = tf.one_hot([self.target_class], 10)

        # Choose objective function
        if obj_fun == 'carlini':
            fun = self._fun_carlini
        elif obj_fun == 'szegedy':
            fun = self._fun_szegedy
        else:
            msg = f"`{obj_fun}` function not implemented."
            logger.error(msg)
            raise NotImplementedError(msg)

        # Clear previous result
        self.res = {
            'x': None,
            'fun': None,
            'nit': None
        }

        # Set upper and lower bound for c
        right = c_right
        left = c_left

        # Evaluate on the right
        c = right
        logger.info(f"Performing binary search. c:{c}")
        res_x, res_fun, res_nit = self._minimize(fun, c)

        # If res.x on the right isn't classified as the target, return FAILURE.
        if eval_flat_pred(res_x, self.model) != self.target_class:
            logger.error(f"c={c} didn't work.")
            return FAILURE
        # Else store successful result and continue
        self.res['x'] = res_x
        self.res['fun'] = res_fun
        self.res['nit'] = res_nit

        # Evaluate on the left
        c = left
        logger.info(f"Performing binary search. c:{c}")
        res_x, res_fun, res_nit = self._minimize(fun, c)

        # If res.x on the left is classified as the target, return
        if eval_flat_pred(res_x, self.model) == self.target_class:
            self.res['x'] = res_x
            self.res['fun'] = res_fun
            self.res['nit'] = res_nit
            logger.info("END")
            return SUCCESS

        count = 0
        # If right is correct and left wrong:
        while count < maxiters_bs:
            c = (left + right) / 2

            aux_c = ("{0:.2f}").format(c)
            logger.info(f"Performing binary search. c:{aux_c}, iter: {count}")

            res_x, res_fun, res_nit = self._minimize(fun, c)

            # If attack succeeds, move right to the middle. If it doesn't, move
            # the left to the middle
            if eval_flat_pred(res_x, self.model) == self.target_class:
                right = c
                self.res['x'] = res_x
                self.res['fun'] = res_fun
                self.res['nit'] = res_nit
            else:
                left = c
            count += 1

        # We can guarantee that we succeeded for at least one c
        logger.info("END")
        return SUCCESS

    def parallel_attack(
            self,
            original_input: np.ndarray,
            original_class: int,
            target_class: int,
            initial_guess: np.ndarray,
            obj_fun: str,
            c_start: float = 1e-02,
            c_stop: float = 1.0,
            c_num: int = 10
        ) -> int:
        '''
        Execute the adversarial attack in parallel over a range of constants
        `c`.

        This method evaluates the attack across multiple values of `c`,
        generated using `np.linspace` from `c_start` to `c_stop` with `c_num`
        steps. Each attack is run in a separate process (using multiprocessing,
        not true parallelism). The result corresponding to the adversarial
        example with the smallest distance (as defined by the configured `Dist`
        metric) but classified as `target_classs` is selected and stored in the
        `res` attribute.

        Parameters
        ----------
        original_input : np.ndarray
            The original input image to be perturbed.

        original_class : int
            The class predicted by the model for the original input.

        target_class : int
            The desired target class for the adversarial attack.

        initial_guess : np.ndarray
            An initial guess for the adversarial example.

        obj_fun : str
            Objective function to use in the optimization. Options are:
            - `'carlini'`: Uses Carlini's objective function.
            - `'szegedy'`: Uses Szegedy's objective function.

        c_start : float, optional
            Starting value of the range of `c` values (default is 1e-2).

        c_stop : float, optional
            Ending value of the range of `c` values (default is 1.0).

        c_num : int, optional
            Number of `c` values to evaluate between `c_start` and `c_stop`
            (default is 10).

        Returns
        -------
        int
            Status code:
            - SUCCESS = 0
            - FAILURE = 1

        Raises
        ------
        ValueError
            If an unsupported objective function is provided.
        '''
        logger.info("START")
        self.original_input = original_input
        # Convert original input to tensor
        self.original_input_tensor = tf.convert_to_tensor(
            self.original_input.reshape(-1,28,28,1),
            dtype=tf.float32
        )
        self.original_class = original_class
        self.target_class = target_class
        self.initial_guess = initial_guess
        # TODO: again this 10 maybe shouldn't be hardcoded
        self.target_one_hot = tf.one_hot([self.target_class], 10)

        # Choose objective function
        if obj_fun == 'carlini':
            fun = self._fun_carlini
        elif obj_fun == 'szegedy':
            fun = self._fun_szegedy
        else:
            msg = f"`{obj_fun}` function not implemented."
            logger.error(msg)
            raise NotImplementedError(msg)

        # Clear previous result
        self.res = {
            'x': None,
            'fun': None,
            'nit': None
        }

        # Create the values of c
        cs = np.linspace(start=c_start, stop=c_stop, num=c_num)

        # List with tuples of arguments to pass to starmap
        pool_args = [(fun, c) for c in cs]

        # Let multiprocessing handle the pool
        with multiprocessing.Pool(processes=len(cs)) as pool:
            results = pool.starmap(self._minimize, pool_args)

        # Retrieve the best result
        best_dist = np.inf
        for c, res in zip(cs, results):
            res_x, res_fun, res_nit = res
            if eval_flat_pred(res_x, self.model) == self.target_class:
                dist = self.distance.compute_vec(res_x, self.original_input.flatten())
                logger.info(f"Attack with c={c} was successful with distance={dist}")

                if dist < best_dist:
                    self.res['x'] = res_x
                    self.res['fun'] = res_fun
                    self.res['nit'] = res_nit

        if self.res['x'] is None:
            logger.error("No value for c worked.")
            logger.info("END")
            return FAILURE

        logger.info("END")
        return SUCCESS

    def _fun_carlini(
            self,
            x: np.ndarray,
            c: float
        ) -> tuple[float, np.ndarray]:
        '''
        Objective function based on Carlini's formulation. Function to be
        minimized by `_minimize`.

        Parameters
        ----------
        x : np.ndarray
            The input vector (current adversarial candidate).

        c : float
            Constant to weight the distance metric and the classification.

        Returns
        -------
        val : float
            The result of the objective function evaluated at `x`.

        grad : np.ndarray
            Gradient of the objective function with respect to `x`.
        '''
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
            d = self.distance.compute_tens(x_tensor, self.original_input_tensor)
            val = d + c * tf.nn.relu(max_z - pred[self.target_class])

        gradients = tape.gradient(val, x_tensor).numpy().flatten()
        val = val.numpy()

        return val, gradients

    def _fun_szegedy(
            self,
            x: np.ndarray,
            c: float
        ) -> tuple[float, np.ndarray]:
        '''
        Objective function based on Szegedy's formulation. Function to be
        minimized by `_minimize`.

        Parameters
        ----------
        x : np.ndarray
            The input vector (current adversarial candidate).

        c : float
            Constant to weight the distance metric and the classification.

        Returns
        -------
        val : float
            The result of the objective function evaluated at `x`.

        grad : np.ndarray
            Gradient of the objective function with respect to `x`.
        '''
        # Convert the starting point from ndarray to tensor
        x_tensor = tf.convert_to_tensor(x.reshape(-1,28,28,1), dtype=tf.float32)
        x_tensor = tf.Variable(x_tensor, trainable=True)

        # Calculate loss and gradients
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred = self.model(x_tensor)
            ce = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(self.target_one_hot, pred)
            )

            # Get distance from x0_t to x_t
            d = self.distance.compute_tens(x_tensor, self.original_input_tensor)
            val = c * d + ce

        gradients = tape.gradient(val, x_tensor).numpy().flatten()
        val = val.numpy()

        return val, gradients

    def save(self, path: str | Path, visualize: bool = False) -> None:
        '''
        Save the result of the adversarial attack to disk.

        This function saves the adversarial example, the value of the objective
        function, and the number of iterations used by the optimizer to a `.npy`
        binary file.  It also generates and saves a PNG image of the adversarial
        example.

        If `visualize` is set to `True`, the adversarial image is displayed
        after saving.

        Parameters
        ----------
        path : str or Path
            Directory path where the result files will be saved. The directory
            and its parents will be created if they do not exist.

        visualize : bool, optional
            If True, display the adversarial image using matplotlib (default is
            False).

        Files Saved
        -----------
        - `{original_class}-to-{target_class}.npy`:
            Binary file containing:
            1. The adversarial example (`np.ndarray`)
            2. The objective function value (`float`)
            3. The number of iterations (`int`)

        - `{original_class}-to-{target_class}.png`:
            Grayscale PNG image of the adversarial example.
        '''
        # TODO: refactor this. I don't like it. But it works, I think.
        if self.res['x'] is None:
            logger.error("Cannot save empty result.")
            return

        # Create the path and its parent directories if they don't exist
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(f'{path}/{self.original_class}-to-{self.target_class}.npy', 'wb') as f:
            np.save(f, self.res['x'])
            np.save(f, self.res['fun'])
            np.save(f, self.res['nit'])

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
    '''
    Adversarial attack implementation using SciPy's `minimize` function.

    This class extends the `Attack` base class and uses a SciPy optimizer (e.g.,
    L-BFGS-B, trust-constr) to solve the adversarial optimization problem.

    Attributes
    ----------
    method : str
        Optimization method to use (e.g., 'L-BFGS-B', 'trust-constr').

    options : dict or None
        Dictionary of solver-specific options to pass to
        `scipy.optimize.minimize`.

    bounds : list of tuple
        List of (min, max) bounds for each input dimension. Currently hard-coded
        to constrain each pixel to [0.0, 1.0].

    Methods
    -------
    _minimize(fun, c)
        Minimize the given objective function using SciPy's optimizer.
    '''

    def __init__(
            self,
            model,
            distance: Dist = Dist.L2,
            method: str = 'L-BFGS-B',
            options: dict = None
        ):
        '''
        Initialize a SciPyAttack instance.

        Parameters
        ----------
        model : Model
            The model to attack.

        distance : Dist, optional
            Distance metric to use for comparing images (default is L2).

        method : str, optional
            Optimization method to use in `scipy.optimize.minimize` (default is
            'L-BFGS-B').

        options : dict, optional
            Dictionary of additional options to pass to the optimizer (default
            is None).
        '''
        super().__init__(model, distance)
        self.method = method
        self.options = options
        self.bounds = [(0.,1.)]*784 # Maybe this shouldn't be hard-coded

    def _minimize(self, fun: Callable, c: float) -> tuple[np.ndarray, float, int]:
        '''
        Minimize the given objective function using SciPy's optimizer.

        Parameters
        ----------
        fun : Callable
            The objective function to minimize. Must return a tuple `(val,
            grad)`.

        c : float
            Trade-off constant weighting classification loss versus distance.

        Returns
        -------
        x : np.ndarray
            The adversarial example found by the optimizer.

        fun : float
            The final value of the objective function.

        nit : int
            Number of iterations taken by the optimizer.
        '''
        res = minimize(
            fun=fun,
            x0=self.initial_guess,
            args=(c,),
            method=self.method,
            bounds=self.bounds,
            jac=True,
            options=self.options
        )
        return res.x, res.fun, res.nit


class OptimusAttack(Attack):
    '''
    Adversarial attack implementation using Line Search Sequential Quadratic
    Programming (SQP).

    This class extends the `Attack` base class and solves the adversarial
    optimization problem using a custom SQP method with line search and box
    constraints.

    Attributes
    ----------
    maxiters : int
        Maximum number of iterations allowed for the optimization algorithm.

    eta : float
        Step size parameter for accepting candidate steps.

    tau : float
        Step size reduction factor for line search.

    tol : float
        Convergence tolerance for the optimization.

    Methods
    -------
    _restr(x)
        Evaluate inequality constraints to ensure inputs remain in [0, 1]^n.

    _minimize(fun, c)
        Minimize the adversarial objective using line search SQP.
    '''

    def __init__(
            self,
            model,
            distance: Dist = Dist.L2,
            maxiters_method: int = 50,
            eta: float = 0.4,
            tau: float = 0.7,
            tol: float = 1.1
        ):
        '''
        Initialize an OptimusAttack instance.

        Parameters
        ----------
        model : Model
            The model to attack.

        distance : Dist, optional
            Distance metric to use for comparing images (default is L2).

        maxiters_method : int, optional
            Maximum number of iterations for the SQP optimizer (default is 50).

        eta : float, optional
            Step size parameter for accepting candidate steps (default is 0.4).

        tau : float, optional
            Step size reduction factor for line search (default is 0.7).

        tol : float, optional
            Convergence tolerance for the optimization (default is 1.1).
        '''
        super().__init__(model, distance)
        self.maxiters = maxiters_method
        self.eta = eta
        self.tau = tau
        self.tol = tol

    def _restr(
            self,
            x: np.ndarray
        ) -> tuple[float, np.ndarray]:
        '''
        Evaluate the inequality constraints for the optimization problem.

        Enforces box constraints on `x` such that `0 <= x_i <= 1` for each pixel.
        This is represented as a vector `c(x)` where all elements must be >= 0:
            c(x) = [x_1, x_2, ..., x_n, 1 - x_1, 1 - x_2, ..., 1 - x_n]

        Parameters
        ----------
        x : np.ndarray
            Input vector to evaluate the constraints on.

        Returns
        -------
        c : np.ndarray
            Constraint values `c(x)`, which must all be non-negative.

        A : np.ndarray
            Jacobian matrix of the constraints.
        '''
        c = np.concatenate([x, 1 - x])
        A = np.concatenate([np.eye(x.size), -np.eye(x.size)])
        return c, A

    def _minimize(self, fun: Callable, c: float) -> tuple[np.ndarray, float, int]:
        '''
        Minimize the given objective function using Line Search Sequential
        Quadratic Programming (SQP).

        Parameters
        ----------
        fun : Callable
            The objective function to minimize. Must return a tuple `(val,
            grad)`.

        c : float
            Trade-off constant weighting classification loss versus distance.

        Returns
        -------
        x : np.ndarray
            The adversarial example found by the optimizer.

        fun : float
            The final value of the objective function.

        nit : int
            Number of iterations taken by the optimizer.
        '''
        res_x, _ = ls_sqp(
            fun=fun,
            restr=self._restr,
            args=(c,),
            x_0=self.initial_guess,
            lam_0=np.ones(2 * self.initial_guess.size),
            B_0=np.eye(self.initial_guess.size),
            hessian='L-BFGS',
            eta=self.eta,
            tau=self.tau,
            maxiters=self.maxiters,
            tol=self.tol
        )
        res_fun, _ = fun(res_x, c)
        return res_x, res_fun, -1 #TODO: return iteration count
