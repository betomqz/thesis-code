import numpy as np

def int_point_qp(G: np.array,
				 c: np.array,
				 A: np.array,
				 b: np.array,
                 x_0: np.array,
				 y_0: np.array = None,
				 lam_0: np.array = None,
				 sigma: float = 0.5,
                 maxiters: int = 10,
                 ) -> tuple[np.array, np.array, np.array]:
    '''
    Solve quadratic problem
        $$ \\min  q(x) = \\frac{1}{2} x^T G x + x^T c $$
        s.t. $$Ax \\geq b.$$
    Using the interior point method described by Nocedal.

    Parameters
    - `G` (np.array): symmetric and positive semidefinite nxn matrix
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
		# TODO: better use a logger.
        print(f"iter {k}:\n - x: {x_k}\n - y: {y_k}\n - l: {lam_k}")

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
        deltas = np.linalg.solve(M,r)
        d_x = deltas[:n]
        d_y = deltas[n:n+m]
        d_lam = deltas[n+m:]

        # Choose alpha TODO: implement a good strategy
        alpha = 0.1

        # Update x_k, y_k, and lam_k
        x_k += alpha * d_x
        y_k += alpha * d_y
        lam_k += alpha * d_lam

        k += 1

    return x_k, y_k, lam_k
