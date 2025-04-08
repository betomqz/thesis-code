# Thesis - Code

Author: José Alberto Márquez Luján
Institution: ITAM
Degree: Bsc in Applied Mathematics and Bsc in Computer Engineering
Advisor: Dr. Andreas Wachtel

## Overview

This project implements a numerical optimization algorithm using Sequential
Quadratic Programming (SQP) with an interior-point method. The goal is to find
adversarial perturbations that cause a neural network to misclassify images,
minimizing the change required to the images.

It was developed as part of my undergraduate thesis in Applied Mathematics and
Computer Engineering.

## Use cases

The code found here can be used for two main purposes:

1. Adversarial examples generation: The infrastructure to generate adversarial
examples for a given model is provided here, allowing the user to change
objective functions, optimization methods, custom strategies, and distance
metrics.
2. Numerical optimization research: A big part of the effort in this project was
put into understanding and implementing a SQP algorithm with inequiality
constraints. The algorithm was obtained from Jorge Nocedal and Stephen J.
Wright's book, _Numerical Optimization_. This algorithm can be used in a
different context than the generation of adversarial examples.

## Instalation

This project was developed using Python 3.10.16.

1. Clone the repository

```bash
git clone https://github.com/betomqz/thesis-code.git
cd thesis-code
```

2. Create and activate a virtual environment

```bash
python -m venv thesisenv
source thesisenv/bin/activate
```

3. Install the project

```
pip install -e .
```

## Unit tests

Example on how to run `test_scipy_szegedy_parallel_L2`:

```
python -m unittest tests.test_attack.TestAttack.test_scipy_szegedy_parallel_L2
```


# `opt_attack` documentation

## [`optimus`](src/opt_attack/optimus.py)

This submodule has the numerical optimization algorithms necessary to minimize a
function. It was all implemented following Nocedal's book.

### `int_point_qp`

```python
def int_point_qp(
    G: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x_0: np.ndarray,
    y_0: np.ndarray = None,
    lam_0: np.ndarray = None,
    maxiters: int = 50,
    tol: np.float64 = np.finfo(np.float64).eps
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Solves a convex quadratic programming (QP) problem with inequality
constraints using the predictor-corrector interior-point method described by
Nocedal's book in algorithm 16.4 (p. 484).

The problem is formulated as:
```
min q(x) = 1/2 x^T * G * x + x^T * c
```

subject to
```
A * x >= b
```

#### Parameters

- **G** : _ndarray_

    Symmetric and positive semidefinite `nxn` matrix.

- **c** : _ndarray_

    Coefficient vector of size $n$.

- **A** : _ndarray_

    Constraint matrix of size $m \times n$.

- **b** : _ndarray_

    Constraint vector of size $m$.

- **x_0** : _ndarray_

    Initial guess for **x**.

- **y_0** : _ndarray_, _optional_

    Initial guess for the slack variable **y**. Default is None.

- **lam_0** : _ndarray_, _optional_

    Initial guess for the Lagrange multipliers. Default is None.

- **maxiters** : _int_, _optional_

    Maximum number of iterations. Default is 50.

- **tol** : _float_, _optional_

    Tolerance for the convergence test. Default is machine epsilon for `np.float64`.

#### Returns

- **x** : _ndarray_

    The optimal solution.

- **y** : _ndarray_

    Slack variables at the optimal point.

- **lam** : _ndarray_

    Lagrange multipliers associated with the constraints.

### Hessian approximations

```python
def _bfgs(s_k: np.ndarray, y_k: np.ndarray, B_k: np.ndarray) -> np.ndarray:
```

Calculates an update to the Hessian using a damped BFGS approach described
by Nocedal in Procedure 18.2 (p. 537) to guarantee that the update is s.p.d.

#### Parameters

- **s_k** : _ndarray_

    Vector representing the change for x in current iteration (alpha_k *
    p_k)

- **y_k** : _ndarray_

    Vector representing the change for the lagrangian in current iteration

- **B_k** : _ndarray_

    Approximation to be updated.

#### Returns

**B_k** : _ndarray_
    Updated approximation to the Hessian

```python
def _l_bfgs(S_k: np.ndarray, Y_k: np.ndarray) -> np.ndarray:
```

Calculates an approximation `B_k` to the Hessian using a limited-memory
updating approach described by Nocedal (eq. 7.29, p. 182)

#### Parameters

- **S_k** : _ndarray_

    `n x m` matrix with the `m` most recent `s_i` vectors

- **Y_k** : _ndarray_

    `n x m` matrix with the `m` most recent `y_i` vectors

#### Returns

- **B_k** : _ndarray_

    Approximation to the Hessian

### Line Search Sequential Quadratic Programming

```python
def ls_sqp(
    fun: Callable[[np.ndarray, tuple], tuple[float, np.ndarray]],
    restr: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    x_0: np.ndarray,
    lam_0: np.ndarray,
    B_0: np.ndarray,
    hessian: str | Callable[[np.ndarray], np.ndarray],
    eta: float,
    tau: float,
    maxiters: int,
    args: tuple = (),
    tol: np.float64 = np.finfo(np.float64).eps
) -> tuple[np.ndarray, np.ndarray]:
```

Solves a constrained optimization problem using Sequential Quadratic
Programming (SQP) with a line search approach. Based on Algorithm 18.3 (p.
545) from Nocedal's book.

#### Parameters

- **fun** : _callable_

    Function to minimize. Must return f(x) and its gradient.

- **restr** : _callable_

    Constraint function (denoted as c(x) in Nocedal). Must return c(x) and
    the Jacobian (denoted as A(x) in Nocedal).

- **x_0** : _ndarray_

    Initial point for the optimization.

- **lam_0** : _ndarray_

    Initial guess for the Lagrange multipliers.

- **B_0** : _ndarray_

    Initial approximation of the Hessian matrix. Must be symmetric positive
    definite (s.p.d.).

- **hessian** : _str_ or _callable_

    Approximation method for the Hessian matrix. Supported options:
    - 'BFGS' : Damped BFGS method.
    - 'L-BFGS' : Limited-memory BFGS method.
    - A callable function if the exact Hessian is available.

- **eta** : _float_

    Step size parameter. Must be strictly between 0 and 0.5.

- **tau** : _float_

    Line search parameter. Must be strictly between 0 and 1.

- **maxiters** : _int_

    Maximum number of iterations allowed.

- **args** : _tuple_, _optional_

    Arguments passed to ``fun``.

- **tol** : _float_, _optional_

    Tolerance for the convergence test. Default is machine epsilon for
    ``np.float64``.

#### Returns

- **x_opt** : _ndarray_

    The optimal solution found by the algorithm, or the value at the last
    iteration.

- **lam_opt** : _ndarray_

    The corresponding Lagrange multipliers at the optimal point.

## [`attack`](src/opt_attack/attack.py)

### `Dist` enumeration

```python
class Dist(enum.Enum):
```

Enumeration of vector distance metrics.

#### Members

- **LINF** : _str_

    Infinity norm (maximum absolute difference).

- **L1** : _str_

    L1 norm (sum of absolute differences).

- **L2** : _str_

    L2 norm (Euclidean distance).

#### `compute_vec`

```python
def compute_vec(self, x: np.ndarray, y: np.ndarray) -> float:
```

Compute the distance between two numpy vectors using the selected norm.

##### Parameters

- **x** : _np.ndarray_

    First input vector.

- **y** : _np.ndarray_

    Second input vector.

##### Returns

- **d** : _float_

    The computed distance between the vectors.

#### `compute_tens`

```python
def compute_tens(self, x: np.ndarray, y: np.ndarray) -> float:
```

Compute the distance between two tensors using the selected norm. The
calculation is done using TensorFlow to allow the use of GradientTape.

##### Parameters

- **x** : _np.ndarray_

    First input tensor.

- **y** : _np.ndarray_

    Second input tensor.

##### Returns

- **d** : _float_

    The computed distance between the vectors.

### `Attack` class

```python
class Attack:
```

Abstract class to generate adversarial examples for a given model.

#### Attributes

- **model** : _Model_

    Model for which the adversarial examples will be generated.

- **distance** : _Dist_

    Distance metric to compare two images.

- **original_input** : _np.ndarray_

    Original input image as a NumPy array.

- **original_input_tensor** : _tf.Tensor_

    Original input image converted to a TensorFlow tensor.

- **original_class** : _int_

    Class index predicted by the model for the original input.

- **target_class** : _int_

    Target class for the adversarial example.

- **target_one_hot** : _tf.Tensor_

    One-hot encoded target class label as a TensorFlow tensor.

- **initial_guess** : _np.ndarray_

    Initial adversarial guess to start the optimization from.

- **res** : _dict_

    Dictionary containing the results of the attack. It includes:
    - `'x'`: the adversarial example generated.
    - `'fun'`: the evaluation of the minimization function on the adversarial
    example.
    - `'nit'`: the number of iterations taken by the optimization method to
    find the example.

#### `__init__`

```python
def __init__(
    self,
    model: Model,
    distance: Dist = Dist.L2
):
```

Initialize the Attack object.

##### Parameters

- **model** : _Model_

    The model to attack.

- **distance** : _Dist_, _optional_

    Distance metric used to compare images (default is L2).

#### `_minimize`

```python
def _minimize(self, fun: Callable, c: float) -> tuple[np.ndarray, float, int]:
```

Execute the optimization method to minimize the given objective function.

This method is intended to be overridden by subclasses. It performs the
optimization to generate an adversarial example that minimizes a
trade-off between classification loss and perturbation distance,
weighted by the constant `c`.

##### Parameters

- **fun** : _Callable_

    The objective function to minimize.

- **c** : _float_

    Trade-off constant that balances classification loss and perturbation
    distance in the objective.

##### Returns

- **x** : _np.ndarray_

    The adversarial example found by the optimization.

- **fun_x** : _float_

    The value of the objective function evaluated at `x`.

- **nit** : _int_

    Number of iterations used by the optimization algorithm.

##### Raises

- `NotImplementedError`

    If the method is not implemented by a subclass.

#### binary_search_attack

```python
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
```
Execute a binary search to find the optimal constant `c` for the
adversarial attack.

This method searches for the best value of `c` using a binary search
approach. The selected objective function determines the loss used
during optimization. The results of the attack are stored in the `res`
attribute.

##### Parameters

- **original_input** : _np.ndarray_

    The original input image to be perturbed.

- **original_class** : _int_

    The class predicted by the model for the original input.

- **target_class** : _int_

    The desired target class for the adversarial attack.

- **initial_guess** : _np.ndarray_

    An initial guess for the adversarial example.

- **obj_fun** : _str_

    Objective function to use in the optimization. Options are:
    - `'carlini'`: Uses Carlini's objective function.
    - `'szegedy'`: Uses Szegedy's objective function.

- **maxiters_bs** : _int_, _optional_

    Maximum number of binary search iterations (default is 10).

- **c_left** : _float_, _optional_

    Lower bound of the binary search interval for the constant `c`
    (default is 1e-2).

- **c_right** : _float_, _optional_

    Upper bound of the binary search interval for the constant `c`
    (default is 1.0).

##### Returns

- **status**: _int_

    Status code:
    - SUCCESS = 0
    - FAILURE = 1

##### Raises

- `ValueError`

    If an unsupported objective function is provided.

#### parallel_attack

```python
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
```

Execute the adversarial attack in parallel over a range of constants
`c`.

This method evaluates the attack across multiple values of `c`,
generated using `np.linspace` from `c_start` to `c_stop` with `c_num`
steps. Each attack is run in a separate process (using multiprocessing,
not true parallelism). The result corresponding to the adversarial
example with the smallest distance (as defined by the configured `Dist`
metric) but classified as `target_classs` is selected and stored in the
`res` attribute.

##### Parameters

- **original_input** : _np.ndarray_

    The original input image to be perturbed.

- **original_class** : _int_

    The class predicted by the model for the original input.

- **target_class** : _int_

    The desired target class for the adversarial attack.

- **initial_guess** : _np.ndarray_

    An initial guess for the adversarial example.

- **obj_fun** : _str_

    Objective function to use in the optimization. Options are:
    - `'carlini'`: Uses Carlini's objective function.
    - `'szegedy'`: Uses Szegedy's objective function.

- **c_start** : _float_, _optional_

    Starting value of the range of `c` values (default is 1e-2).

- **c_stop** : _float_, _optional_

    Ending value of the range of `c` values (default is 1.0).

- **c_num** : _int_, _optional_

    Number of `c` values to evaluate between `c_start` and `c_stop`
    (default is 10).

##### Returns

- **status**: _int_

    Status code:
    - SUCCESS = 0
    - FAILURE = 1

##### Raises

- `ValueError`

    If an unsupported objective function is provided.

#### _fun_carlini

```python
def _fun_carlini(
    self,
    x: np.ndarray,
    c: float
) -> tuple[float, np.ndarray]:
```

Objective function based on Carlini's formulation. Function to be
minimized by `_minimize`.

##### Parameters

- **x** : _np.ndarray_

    The input vector (current adversarial candidate).

- **c** : _float_

    Constant to weight the distance metric and the classification.

##### Returns

- **val** : _float_

    The result of the objective function evaluated at `x`.

- **grad** : _np.ndarray_

    Gradient of the objective function with respect to `x`.

#### _fun_szegedy

```python
def _fun_szegedy(
    self,
    x: np.ndarray,
    c: float
) -> tuple[float, np.ndarray]:
```

Objective function based on Szegedy's formulation. Function to be
minimized by `_minimize`.

##### Parameters

- **x** : _np.ndarray_

    The input vector (current adversarial candidate).

- **c** : _float_

    Constant to weight the distance metric and the classification.

##### Returns

- **val** : _float_

    The result of the objective function evaluated at `x`.

- **grad** : _np.ndarray_

    Gradient of the objective function with respect to `x`.

#### save

```python
def save(self, path: str | Path, visualize: bool = False) -> None:
```

Save the result of the adversarial attack to disk.

This function saves the adversarial example, the value of the objective
function, and the number of iterations used by the optimizer to a `.npy`
binary file.  It also generates and saves a PNG image of the adversarial
example.

If `visualize` is set to `True`, the adversarial image is displayed
after saving.

##### Parameters

- **path** : _str_ or _Path_

    Directory path where the result files will be saved. The directory
    and its parents will be created if they do not exist.

- **visualize** : _bool_, _optional_

    If `True`, display the adversarial image using matplotlib (default is
    `False`).

##### Files Saved

- `{original_class}-to-{target_class}.npy`:

    Binary file containing:
    1. The adversarial example (`np.ndarray`)
    2. The objective function value (`float`)
    3. The number of iterations (`int`)

- `{original_class}-to-{target_class}.png`:

    Grayscale PNG image of the adversarial example.

### `SciPyAttack` subclass

```python
class SciPyAttack(Attack):
```

Adversarial attack implementation using SciPy's `minimize` function.

This class extends the `Attack` base class and uses a SciPy optimizer (e.g.,
L-BFGS-B, trust-constr) to solve the adversarial optimization problem.

#### Attributes

- **method** : _str_

    Optimization method to use (e.g., 'L-BFGS-B', 'trust-constr').

- **options** : _dict_ or None

    Dictionary of solver-specific options to pass to
    `scipy.optimize.minimize`.

- **bounds** : _list of tuple_

    List of `(min, max)` bounds for each input dimension. Currently hard-coded
    to constrain each pixel to $[0.0, 1.0]$.

#### __init__

```python
def __init__(
    self,
    model,
    distance: Dist = Dist.L2,
    method: str = 'L-BFGS-B',
    options: dict = None
):
```

Initialize a SciPyAttack instance.

##### Parameters

- **model** : _Model_

    The model to attack.

- **distance** : _Dist_, _optional_

    Distance metric to use for comparing images (default is L2).

- **method** : _str_, _optional_

    Optimization method to use in `scipy.optimize.minimize` (default is
    'L-BFGS-B').

- **options** : _dict_, _optional_

    Dictionary of additional options to pass to the optimizer (default
    is None).

#### _minimize

```python
def _minimize(self, fun: Callable, c: float) -> tuple[np.ndarray, float, int]:
```

Minimize the given objective function using SciPy's optimizer.

##### Parameters

- **fun** : _Callable_

    The objective function to minimize. Must return a tuple `(val,
    grad)`.

- **c** : _float_

    Trade-off constant weighting classification loss versus distance.

##### Returns

- **x** : _np.ndarray_

    The adversarial example found by the optimizer.

- **fun** : _float_

    The final value of the objective function.

- **nit** : _int_

    Number of iterations taken by the optimizer.

### `OptimusAttack` subclass

```python
class OptimusAttack(Attack):
```

Adversarial attack implementation using Line Search Sequential Quadratic
Programming (SQP).

This class extends the `Attack` base class and solves the adversarial
optimization problem using a custom SQP method with line search and box
constraints.

#### Attributes

- **maxiters** : _int_

    Maximum number of iterations allowed for the optimization algorithm.

- **eta** : _float_

    Step size parameter for accepting candidate steps.

- **tau** : _float_

    Step size reduction factor for line search.

- **tol** : _float_

    Convergence tolerance for the optimization.

#### __init__

```python
def __init__(
    self,
    model,
    distance: Dist = Dist.L2,
    maxiters_method: int = 50,
    eta: float = 0.4,
    tau: float = 0.7,
    tol: float = 1.1
):
```

Initialize an OptimusAttack instance.

##### Parameters

- **model** : _Model_

    The model to attack.

- **distance** : _Dist_, _optional_

    Distance metric to use for comparing images (default is L2).

- **maxiters_method** : _int_, _optional_

    Maximum number of iterations for the SQP optimizer (default is 50).

- **eta** : _float_, _optional_

    Step size parameter for accepting candidate steps (default is 0.4).

- **tau** : _float_, _optional_

    Step size reduction factor for line search (default is 0.7).

- **tol** : _float_, _optional_

    Convergence tolerance for the optimization (default is 1.1).

#### _restr

```python
def _restr(
    self,
    x: np.ndarray
) -> tuple[float, np.ndarray]:
```

Evaluate the inequality constraints for the optimization problem.

Enforces box constraints on `x` such that `0 <= x_i <= 1` for each pixel.
This is represented as a vector `c(x)` where all elements must be >= 0:

$$c(x) = [x_1, x_2, ..., x_n, 1 - x_1, 1 - x_2, ..., 1 - x_n]$$

##### Parameters

- **x** : _np.ndarray_

    Input vector to evaluate the constraints on.

##### Returns

- **c** : _np.ndarray_

    Constraint values `c(x)`, which must all be non-negative.

- **A** : _np.ndarray_

    Jacobian matrix of the constraints.

#### _minimize

```python
def _minimize(self, fun: Callable, c: float) -> tuple[np.ndarray, float, int]:
```

Minimize the given objective function using Line Search Sequential
Quadratic Programming (SQP).

##### Parameters

- **fun** : _Callable_

    The objective function to minimize. Must return a tuple `(val,
    grad)`.

- **c** : _float_

    Trade-off constant weighting classification loss versus distance.

##### Returns

- **x** : _np.ndarray_

    The adversarial example found by the optimizer.

- **fun** : _float_

    The final value of the objective function.

- **nit** : _int_

    Number of iterations taken by the optimizer.
