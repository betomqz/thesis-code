import numpy as np
from scipy import linalg as la
from typing import Callable


def find_alpha(y: np.ndarray,
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
                 sigma: float = 0.5,
                 maxiters: int = 50,
                 tol: np.float64 = np.finfo(np.float64).eps,
                 verbose = True
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Solve quadratic problem
        $$ \\min  q(x) = \\frac{1}{2} x^T G x + x^T c $$
        s.t. $$Ax \\geq b.$$
    Using the interior point method described by Nocedal.

    Parameters
    - `G` (np.ndarray): symmetric and positive semidefinite nxn matrix
    - `c` is a vector of size n
    - `A` is an mxn matrix
    - `b` is a vector of size m

    Returns
    - `x`: the solution
    - `y`: slack variable
    - `lam`: lagrange multiplier
    '''
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

    while k < maxiters: 
        # Matrix M is the left side of (16.58)
        M = np.zeros((n+2*m, n+2*m))
        Y = np.diag(y_k)
        Lam = np.diag(lam_k)

        M[:n,:n] = G
        M[:n,n+m:] = -A.T
        M[n:n+m,:n] = A
        M[n:n+m,n:n+m] = -np.eye(m)
        M[n+m:,n:n+m] = Lam
        M[n+m:,n+m:] = Y

        # Right side of (16.58)
        mu = np.dot(y_k,lam_k) / m
        r_d = np.dot(G,x_k) - np.dot(A.T,lam_k) + c
        r_p = np.dot(A,x_k) - y_k - b
        pert = -np.dot(np.dot(Lam,Y),e) + sigma * mu * e

        r = np.zeros(n+2*m)
        r[:n] = - r_d
        r[n:n+m] = -r_p
        r[n+m:] = pert

        # Solve (16.58)
        deltas = la.solve(M,r)
        d_x = deltas[:n]
        d_y = deltas[n:n+m]
        d_lam = deltas[n+m:]

        # If stopping criteria is met, return. 
        if la.norm(d_x) <= tol:
            if verbose:
                print("Solution found")
                print(f"iter {k}:\n - x: {x_k}\n - y: {y_k}\n - l: {lam_k}")
            return x_k, y_k, lam_k

        # Step length selection from (16.66)

        # Choose tau in (0,1) to approach 1 as iterations go on.
        # TODO: This is completely arbitrary but I don't want to think of
        #       something else rn. Seems legit.
        tau_k = 1 / (1 + np.exp(-0.1 * k))
        alpha_pri = find_alpha(y_k, d_y, tau_k)
        alpha_dual = find_alpha(lam_k, d_lam, tau_k)
        alpha = np.min([alpha_pri, alpha_dual])

        if alpha == 0:
            if verbose:
                print(f"INT POINT WARNING: alpha is 0 for iter {k}")
            alpha = 0.1

        # Update x_k, y_k, and lam_k
        x_k += alpha * d_x
        y_k += alpha * d_y
        lam_k += alpha * d_lam

        # make sigma tend to zero
        sigma *= 0.8

        k += 1
        # TODO: better use a logger.
        if verbose:
            print(f"iter {k}:\n" +
                f" - x: {x_k}\n" +
                f" - y: {y_k}\n" +
                f" - l: {lam_k}\n" +
                f" - ||d_x||: {la.norm(d_x)}\n" +
                f" - ||d_y||: {la.norm(d_y)}\n" +
                f" - ||d_lam||: {la.norm(d_lam)}\n" +
                f" - alpha: {alpha}\n")

    if verbose:
        print(f"INT POINT WARNING: Maximum number of iterations achieved: {maxiters}")
    return x_k, y_k, lam_k


def ls_sqp(fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
           restr: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
           x_0: np.ndarray,
           lam_0: np.ndarray,
           B_0: np.ndarray,
           eta: float,
           tau: float,
           maxiters: int,
           tol: np.float64 = np.finfo(np.float64).eps
           ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Parameters:
    - `fun` is the function to minimize. Must return f(x) and its gradient
    - `retr` are the restrictions (denoted by c(x) in Nocedal). Must return
             c(x) and the Jacobian as well (denoted by A(x) in Nocedal).
    - `x_0` is the starting point
    - `lam_0` is the initial guess for the multipliers
    - `B_0` is the initial guess for the Hessian matrix. Must be s.p.d.
    - `eta` must lie strictly between 0 and 0.5
    - `tau` must lie strictly between 0 and 1
    - `maxiters` is the maximum number of iterations allowed
    - `tol` is the tolerance for the convergence test
    '''

    # Evaluate f_0, ∇f_0, c_0, A_0;
    x_k = x_0
    lam_k = lam_0
    f_k, grad_k = fun(x_k) # fun must return both f and its gradient
    c_k, A_k = restr(x_k) # restr must return both c(x) and A(x)

    # || [c_k]^- ||_1: see (15.24)
    c_k_norm = la.norm(np.maximum(0, -c_k), ord=1)

    # Choose initial nxn s.p.d. Hessian approximation B_0
    B_k = B_0

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
                                       maxiters=15,
                                       verbose=False)

        # Set p_lambda <- lambda_hat - lambda_k
        p_lam = lam_hat - lam_k

        # Choose mu_k to satisfy (18.36) with sigma=1
        count_mu = 0
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
        while count_ls < 30:
            # Evaluate possible f_k+1, ∇f_k+1, c_k+1, A_k+1
            s_k = alpha_k * p_k
            f_k, grad_k = fun(x_k + s_k)
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
        B_k = B_k - BssB / sBs + rrT / np.dot(s_k, r_k)

        # Update old info
        phi_old = phi
        grad_k_old = grad_k.copy()
        A_k_old = A_k.copy()

        # If something went wrong with finding mu_k, reset it to 1
        if count_mu == 15:
            print("WARNING: maximum value for mu reached.")
            mu_k = 1

        # If something went wrong with line search, warn
        if count_ls == 30:
            print("WARNING: maximum number of iterations achieved for line "
                  "search")
        
        k += 1
        # If stopping criteria is met, return. 
        # TODO: is this really the best criteria?
        if la.norm(kkt) <= tol:
            print("Solution found")
            print(f"iter {k}:\n - x: {x_k}\n - l: {lam_k}")
            return x_k, lam_k

        # TODO: better use a logger.
        print(f"iter {k}:\n" +
              f" - x: {x_k}\n" +
              f" - l: {lam_k}\n" +
              f" - ||kkt||: {la.norm(kkt)}\n" +
              f" - alpha_k: {alpha_k}\n")

    print(f"WARNING: maximum number of iterations achieved: {maxiters}")
    return x_k, lam_k
