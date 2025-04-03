# Thesis - Code

This is the repository for the code for my bachelor's thesis.
Everything is implemented using Python.

# `opt_attack`

## [`optimus`](src/opt_attack/optimus.py)

This submodule has the numerical optimization algorithms necessary to minimize a
function. It was all implemented following Nocedal's book.

### `int_point_qp`

```python
def int_point_qp(G: np.ndarray,
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

Parameters
----------
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

Returns
-------
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

Parameters
----------
- **s_k** : _ndarray_

    Vector representing the change for x in current iteration (alpha_k *
    p_k)

- **y_k** : _ndarray_

    Vector representing the change for the lagrangian in current iteration

- **B_k** : _ndarray_

    Approximation to be updated.

Returns
-------
**B_k** : _ndarray_
    Updated approximation to the Hessian

```python
def _l_bfgs(S_k: np.ndarray, Y_k: np.ndarray) -> np.ndarray:
```

Calculates an approximation `B_k` to the Hessian using a limited-memory
updating approach described by Nocedal (eq. 7.29, p. 182)

Parameters
----------
- **S_k** : _ndarray_

    `n x m` matrix with the `m` most recent `s_i` vectors

- **Y_k** : _ndarray_

    `n x m` matrix with the `m` most recent `y_i` vectors

Returns
-------
- **B_k** : _ndarray_

    Approximation to the Hessian

### Line Search Sequential Quadratic Programming

```python
def ls_sqp(fun: Callable[[np.ndarray, tuple], tuple[float, np.ndarray]],
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

Parameters
----------
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

Returns
-------
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

#### Methods

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


## `utils`

## Unit tests

Example on how to run `test_scipy_szegedy_parallel_L2`:

```
python -m unittest tests.test_attack.TestAttack.test_scipy_szegedy_parallel_L2
```
