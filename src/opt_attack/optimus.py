import numpy as np
import logging
from collections import deque
from scipy import linalg as la
from typing import Callable


# Logger
logger = logging.getLogger(__name__)

# Capture warnings (for LinAlgWarning)
logging.captureWarnings(True)


def _find_alpha(y: np.ndarray,
               d_y: np.ndarray,
               tau: float):
    '''
    Find maximum alpha in (0,1] such that y + alpha * d_y >= (1 - tau) * y
    '''

    # Calculate - tau * y / d_y except when d_y is 0. If this is the case,
    # simply choose 1.
    props = np.where(d_y != 0, - tau * np.divide(y, d_y, where=(d_y != 0)), 1.)

    # If d_y > 0, we need alpha >= - tau * y / d_y, so there's an error if
    # - tau * y / d_y is greater than 1.
    # TODO: I think this comes down to if any element in y is < 0.
    if np.any((d_y > 0) & (props > 1.)):
        return 0.0
    # else choose 1. in this case
    props = np.where(d_y > 0, 1., props)

    # Return the minimum valid value
    return np.min([1., np.min(props)])


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
    '''
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
    G : ndarray
        Symmetric and positive semidefinite `nxn` matrix.

    c : ndarray
        Coefficient vector of size `n`.

    A : ndarray
        Constraint matrix of size `mxn`.

    b : ndarray
        Constraint vector of size `m`.

    x_0 : ndarray
        Initial guess for `x`.

    y_0 : ndarray, optional
        Initial guess for the slack variable `y`. Default is None.

    lam_0 : ndarray, optional
        Initial guess for the Lagrange multipliers. Default is None.

    maxiters : int, optional
        Maximum number of iterations. Default is 50.

    tol : float, optional
        Tolerance for the convergence test. Default is machine epsilon for `np.float64`.

    Returns
    -------
    x : ndarray
        The optimal solution.

    y : ndarray
        Slack variables at the optimal point.

    lam : ndarray
            Lagrange multipliers associated with the constraints.
    '''
    logger.info("START")
    n = G.shape[0]
    m = A.shape[0]

    k = 0
    x_k = x_0
    e = np.ones(m)
    if y_0 is None:
        y_k = np.ones(m)
    else:
        y_k = y_0
    if lam_0 is None:
        lam_k = np.ones(m)
    else:
        lam_k = lam_0

    # Matrix M is the left side of (16.58)
    M = np.zeros((n+2*m, n+2*m))
    M[:n,:n] = G
    M[:n,n+m:] = -A.T
    M[n:n+m,:n] = A
    M[n:n+m,n:n+m] = -np.eye(m)

    while k < maxiters:
        # Update values for Y and Lam
        Y = np.diag(y_k)
        Lam = np.diag(lam_k)
        M[n+m:,n:n+m] = Lam
        M[n+m:,n+m:] = Y

        # Right side of (16.58) with sigma = 0
        r_d = np.dot(G,x_k) - np.dot(A.T,lam_k) + c
        r_p = np.dot(A,x_k) - y_k - b
        DYe = np.dot(np.dot(Lam, Y), e)

        r = np.zeros(n+2*m)
        r[:n] = - r_d
        r[n:n+m] = -r_p
        r[n+m:] = -DYe

        # Solve (16.58) with sigma = 0
        deltas_aff = la.solve(M,r)
        d_x_aff = deltas_aff[:n]
        d_y_aff = deltas_aff[n:n+m]
        d_lam_aff = deltas_aff[n+m:]

        # Caclulate mu
        mu = np.dot(y_k, lam_k) / m

        # Calculate alpha_aff
        alpha_aff = _find_alpha(
            np.concatenate([y_k, lam_k]),
            np.concatenate([d_y_aff, d_lam_aff]),
            1
        )

        # Calculate mu_aff
        mu_aff = np.dot(y_k+alpha_aff*d_y_aff, lam_k+alpha_aff*d_lam_aff) / m

        # Set centering parameter
        sigma = (mu_aff / mu) ** 3

        # Right side of (16.67)
        Delta_Lam_aff = np.diag(d_lam_aff)
        Delta_Y_aff = np.diag(d_y_aff)
        pert = -np.dot(np.dot(Delta_Lam_aff, Delta_Y_aff), e) + sigma * mu * e
        r[n+m:] = -DYe + pert

        # Solve (16.67)
        deltas = la.solve(M,r)
        d_x = deltas[:n]
        d_y = deltas[n:n+m]
        d_lam = deltas[n+m:]

        # If stopping criteria is met, return.
        if la.norm(d_x, np.infty) <= tol:
            logger.info("Solution found!")
            logger.info(f"iter {k}:\n - x: {x_k}\n - y: {y_k}\n - l: {lam_k}")
            logger.info("END")
            return x_k, y_k, lam_k

        # Step length selection from (16.66)

        # Choose tau in (0,1) to approach 1 as iterations go on.
        # TODO: This is completely arbitrary but I don't want to think of
        #       something else rn. Seems legit.
        tau_k = 1 / (1 + np.exp(-0.1 * k))
        alpha_pri = _find_alpha(y_k, d_y, tau_k)
        alpha_dual = _find_alpha(lam_k, d_lam, tau_k)
        alpha = np.min([alpha_pri, alpha_dual])

        if alpha == 0:
            logger.warning(f"alpha is 0 for iter {k}")
            alpha = 0.1

        # Update x_k, y_k, and lam_k
        x_k += alpha * d_x
        y_k += alpha * d_y
        lam_k += alpha * d_lam

        k += 1
        logger.info(f"iter {k}:\n" +
            f" - ||d_x||_infty: {la.norm(d_x, np.infty)}\n" +
            f" - ||d_y||_infty: {la.norm(d_y, np.infty)}\n" +
            f" - ||d_lam||_infty: {la.norm(d_lam, np.infty)}\n" +
            f" - alpha: {alpha}")

    logger.warning(f"Maximum number of iterations achieved: {maxiters}")
    logger.info("END")
    return x_k, y_k, lam_k


def _bfgs(s_k: np.ndarray, y_k: np.ndarray, B_k: np.ndarray) -> np.ndarray:
    '''
    Calculates an update to the Hessian using a damped BFGS approach described
    by Nocedal in Procedure 18.2 (p. 537) to guarantee that the update is s.p.d.

    Parameters
    ----------
    s_k : ndarray
        Vector representing the change for x in current iteration (alpha_k *
        p_k)

    y_k : ndarray
        Vector representing the change for the lagrangian in current iteration

    B_k : ndarray
        Approximation to be updated.

    Returns
    -------
    B_k : ndarray
        Updated approximation to the Hessian
    '''
    # Damped BFGS updating (Procedure 18.2)
    sy = np.dot(s_k, y_k)
    Bs = np.dot(B_k, s_k)
    sBs = np.dot(s_k, Bs)

    # (18.15)
    theta_k = 1
    if sy < 0.2 * sBs:
        theta_k = 0.8 * sBs / (sBs - sy)

    r_k = theta_k * y_k + (1 - theta_k) * Bs

    # Update B_k with (18.16) to guarantee that it is s.p.d.
    BssB = np.outer(Bs, Bs)
    rrT = np.outer(r_k, r_k)
    return B_k - BssB / sBs + rrT / np.dot(s_k, r_k)


def _l_bfgs(S_k: np.ndarray, Y_k: np.ndarray) -> np.ndarray:
    '''
    Calculates an approximation `B_k` to the Hessian using a limited-memory
    updating approach described by Nocedal (eq. 7.29, p. 182)

    Parameters
    ----------
    S_k : ndarray
        `n x m` matrix with the `m` most recent `s_i` vectors

    Y_k : ndarray
        `n x m` matrix with the `m` most recent `y_i` vectors

    Returns
    -------
    B_k : ndarray
        Approximation to the Hessian
    '''
    n = S_k.shape[0]

    # L_k and D_k are m x m matrices adapted from (7.26) & (7.27)
    sTy = np.dot(S_k.T, Y_k)
    L_k = np.tril(sTy, -1)
    D_k = np.diag(np.diag(sTy))

    # delta_k is the inverse of (7.20)
    Y_k_mius_1 = Y_k[:,-1]
    delta_k = np.dot(Y_k_mius_1, Y_k_mius_1) / np.dot(S_k[:,-1], Y_k_mius_1)

    # middle matrix of dimension 2*m in (7.29)
    M = np.block([
        [delta_k * np.dot(S_k.T, S_k), L_k],
        [L_k.T, -D_k]
    ])

    dSY = np.block([
        [delta_k * S_k, Y_k]
    ])

    # calculate M^{-1} dSY^T
    X = la.solve(M, dSY.T)

    return delta_k * np.eye(n) - np.dot(dSY, X)


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
    '''
    Solves a constrained optimization problem using Sequential Quadratic
    Programming (SQP) with a line search approach. Based on Algorithm 18.3 (p.
    545) from Nocedal's book.

    Parameters
    ----------
    fun : callable
        Function to minimize. Must return f(x) and its gradient.

    restr : callable
        Constraint function (denoted as c(x) in Nocedal). Must return c(x) and
        the Jacobian (denoted as A(x) in Nocedal).

    x_0 : ndarray
        Initial point for the optimization.

    lam_0 : ndarray
        Initial guess for the Lagrange multipliers.

    B_0 : ndarray
        Initial approximation of the Hessian matrix. Must be symmetric positive
        definite (s.p.d.).

    hessian : str or callable
        Approximation method for the Hessian matrix. Supported options:
        - 'BFGS' : Damped BFGS method.
        - 'L-BFGS' : Limited-memory BFGS method.
        - A callable function if the exact Hessian is available.

    eta : float
        Step size parameter. Must be strictly between 0 and 0.5.

    tau : float
        Line search parameter. Must be strictly between 0 and 1.

    maxiters : int
        Maximum number of iterations allowed.

    args : tuple, optional
        Arguments passed to ``fun``.

    tol : float, optional
        Tolerance for the convergence test. Default is machine epsilon for
        ``np.float64``.

    Returns
    -------
    x_opt : ndarray
        The optimal solution found by the algorithm, or the value at the last
        iteration.

    lam_opt : ndarray
        The corresponding Lagrange multipliers at the optimal point.
    '''
    logger.info("START")

    # Verify valid method for quasi-Newton approx
    if (not callable(hessian)) and hessian != 'BFGS' and hessian != 'L-BFGS':
        raise Exception("Invalid method for quasi-Newton approximation.")

    # Evaluate f_0, ∇f_0, c_0, A_0;
    x_k = x_0
    lam_k = lam_0
    f_k, grad_k = fun(x_k, *args) # fun must return both f and its gradient
    c_k, A_k = restr(x_k) # restr must return both c(x) and A(x)

    # || [c_k]^- ||_1: see (15.24)
    c_k_norm = la.norm(np.maximum(0, -c_k), ord=1)

    # Choose initial nxn s.p.d. Hessian approximation B_0
    B_k = B_0

    # Queues to store S_k and Y_k
    S_k = deque(maxlen=10)
    Y_k = deque(maxlen=10)

    # mu_k and rho for (18.36)
    rho = 0.1
    mu_k = 1.0

    # Initial "old" info
    phi_old = f_k + mu_k * c_k_norm
    grad_k_old = grad_k
    A_k_old = A_k

    # Iteration counter
    k = 0

    # Repeat until convergence test is satisfied
    while k < maxiters:
        # Compute p_k by solving (18.11);
        # let lambda_hat be the corresponding mult.
        p_k, _, lam_hat = int_point_qp(G=B_k,
                                       c=grad_k,
                                       A=A_k,
                                       b=-c_k,
                                       x_0=np.ones(x_k.size),
                                       tol=10e-5,
                                       maxiters=15)

        # Set p_lambda <- lambda_hat - lambda_k
        p_lam = lam_hat - lam_k

        # Choose mu_k to satisfy (18.36) with sigma=1
        if c_k_norm > 0:
            mess = np.dot(grad_k, p_k) + 0.5 * np.dot(p_k, np.dot(B_k, p_k))
            mess /= (1 - rho) * c_k_norm
            if mu_k < mess:
                mu_k = mess + 1e-4

        # Set alpha_k <- 1
        alpha_k = 1

        # Compute the directional derivative of last phi
        deriv = np.dot(grad_k, p_k) - mu_k * c_k_norm

        count_ls = 0
        while count_ls < 50:
            # Evaluate possible f_k+1, ∇f_k+1, c_k+1, A_k+1
            s_k = alpha_k * p_k
            f_k, grad_k = fun(x_k + s_k, *args)
            c_k, A_k = restr(x_k + s_k)
            c_k_norm = la.norm(np.maximum(0, -c_k), ord=1)

            # Compute phi_1
            phi = f_k + mu_k * c_k_norm

            # If line search criteria is met, interrupt while
            if phi <= phi_old + eta * alpha_k * deriv:
                break
            # else reset alpha_k <- tau_alpha * alpha_k for some tau_alpha in
            # (0,tau]. TODO (maybe?) choose a better tau_alpha.
            alpha_k *= tau
            count_ls += 1

        # Set x_k+1 and lambda_k+1
        x_k += s_k
        lam_k += alpha_k * p_lam

        # kkt conditions
        kkt = grad_k - np.dot(A_k.T, lam_k)

        # Define y_k as in (18.13)
        y_k = kkt - (grad_k_old - np.dot(A_k_old.T,lam_k))

        if hessian == 'BFGS':
            B_k = _bfgs(s_k=s_k, y_k=y_k, B_k=B_k)
        elif hessian == 'L-BFGS':
            S_k.append(s_k)
            Y_k.append(y_k)
            B_k = _l_bfgs(
                S_k=np.array(S_k).T,
                Y_k=np.array(Y_k).T
            )
        else:
            # We know it must be a callable
            B_k = hessian(x_k)

        # Update old info
        phi_old = phi
        grad_k_old = grad_k.copy()
        A_k_old = A_k.copy()

        # If something went wrong with line search, warn
        if count_ls == 30:
            logger.warning("Maximum number of iterations achieved for line \
                           search")

        k += 1
        # If stopping criteria is met, return.
        kkt_norm = la.norm(kkt, np.infty)
        if kkt_norm <= tol:
            logger.info("Solution found!")
            logger.info(f"iter {k}:\n - x: {x_k}\n - l: {lam_k}")
            logger.info("END")
            return x_k, lam_k

        logger.info(f"iter {k}:\n" +
              f" - ||kkt||: {kkt_norm}\n" +
              f" - mu_k: {mu_k}\n" +
              f" - alpha_k: {alpha_k}")

    logger.warning(f"Maximum number of iterations achieved: {maxiters}")
    logger.info("END")
    return x_k, lam_k
