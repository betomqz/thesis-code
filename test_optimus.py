import unittest
import numpy as np
import numpy.testing as npt
from optimus import int_point_qp, ls_sqp, _find_alpha
from scipy.optimize import rosen, rosen_der, rosen_hess, minimize, LinearConstraint

class TestOptimus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(7679448)

    def perform_int_point_random(self, n, tol=0.01, verbose=False):
        '''Helper function to test the interior point method with random values.'''
        m = 2 * n

        G = np.random.rand(n,n)
        G = np.dot(G,G.T)
        c = np.random.rand(n) * 10 - 5
        A = np.concatenate([np.eye(n), -np.eye(n)])
        b = np.concatenate([-np.ones(n), -2*np.ones(n)])

        def fun(x):
            '''Function to minimize for int_point'''
            return 0.5*np.dot(x,np.dot(G,x)) + np.dot(c, x)

        constr = LinearConstraint(A, b, [np.inf]*m)

        x0 = np.random.randn(n)

        # We will test against SciPy's `minimize`
        res = minimize(fun=fun, x0=x0, method='SLSQP', constraints=constr)

        x, y, lam = int_point_qp(
            G=G,
            c=c,
            A=A,
            b=b,
            x_0=x0.copy(),
            sigma=0.1,
            tol=10e-5,
            verbose=verbose
        )

        if verbose:
            print(f"SciPy: {res.x}")
            print(f"Optimus: {x}")
            print(f"SciPy: {fun(res.x)}")
            print(f"Optimus: {fun(x)}")
            print(f"SciPy: {np.linalg.norm(A@res.x - b, np.infty)}")
            print(f"Optimus: {np.linalg.norm(A@x - b, np.infty)}")

        npt.assert_allclose(x, res.x, atol=tol)

    def test_int_point_qp_random_10(self):
        '''Test interior point method with random values for n=10'''
        self.perform_int_point_random(10)

    def test_int_point_qp_random_20(self):
        '''Test interior point method with random values for n=20'''
        self.perform_int_point_random(20)

    def test_int_point_qp_random_50(self):
        '''Test interior point method with random values for n=50'''
        self.perform_int_point_random(50)

    def test_int_point_qp_random_100(self):
        '''Test interior point method with random values for n=100'''
        self.perform_int_point_random(100)
    
    def test_int_point_qp_random_200(self):
        '''Test interior point method with random values for n=200'''
        self.perform_int_point_random(200)

    def test_int_point_nocedal_475(self):
        '''Test for problem on p. 475 from (Nocedal)'''
        G = np.array([[2,0],[0,2]])
        c = np.array([-2,-5])
        A = np.array([
            [ 1, -2],
            [-1, -1],
            [-1,  2],
            [ 1,  0],
            [ 0,  1]
        ])
        b = np.array([-2,-6,-2,0,0])

        x0 = np.array([2.,0.])
        x, y, lam = int_point_qp(G=G, c=c, A=A, b=b, x_0=x0.copy(), sigma=0.1,
                                tol=10e-10, verbose=False)
        
        npt.assert_allclose(x, np.array([1.4,1.7]))

    def test_hs21(self):
        '''
        Perform the following test:
        Model hs21
          Variables
            x[1] = -1
            x[2] = 1
            obj
          End Variables
        
          Equations
            10*x[1] - x[2] >= 10
            2 <= x[1] <= 50
            -50 <= x[2] <= 50    
        
            ! best known objective = -99.96
            obj = x[1]^2/100 + x[2]^2 - 100
          End Equations
        End Model
        '''
        m = 5

        G = np.array([[1/50, 0], [0, 2]])
        c = np.array([0, 0])
        A = np.array([
            [10, -1],
            [ 1,  0],
            [-1,  0],
            [ 0,  1],
            [ 0, -1]
        ])
        b = np.array([10,2,-50,-50,-50])

        x0 = np.array([-1.,1.])

        def fun(x):
            return x[0]**2/100 + x[1]**2 - 100
        
        x, y, lam = int_point_qp(
            G=G,
            c=c,
            A=A,
            b=b,
            x_0=x0.copy(),
            sigma=0.1,
            tol=10e-5,
            verbose=False
        )

        self.assertAlmostEqual(fun(x), -99.96, places=5)

    def test_positive_direction(self):
        '''Test when d_y is positive and should return a valid alpha'''
        y = np.array([1.0, 2.0, 3.0])
        d_y = np.array([0.5, 1.0, 1.5])
        tau = 0.1
        expected = 1.0
        self.assertAlmostEqual(_find_alpha(y, d_y, tau), expected)

    def test_negative_direction(self):
        '''Test when d_y is negative'''
        y = np.array([1.0, 2.0, 3.0])
        d_y = np.array([-0.5, -1.0, -1.5])
        tau = 0.1
        expected = 0.2
        self.assertAlmostEqual(_find_alpha(y, d_y, tau), expected)

    def test_d_y_zeros(self):
        '''Test when d_y contains zeros'''
        y = np.array([1.0, 2.0, 3.0])
        d_y = np.array([0.0, 1.0, -1.0])
        tau = 0.1
        expected = 0.3
        self.assertAlmostEqual(_find_alpha(y, d_y, tau), expected)

    def test_invalid_case(self):
        '''Test when d_y > 0 and -tau * y / d_y > 1, which should return 0'''
        y = np.array([-1.0, 2.0, 3.0])
        d_y = np.array([0.01, 0.02, 0.03])
        tau = 0.1
        self.assertEqual(_find_alpha(y, d_y, tau), 0.0)

#   ----------------------------------------------------------------------------
    def perform_rosenbrock_ls_sqp(self, n, tol, hessian='L-BFGS'):
        '''
        Helper function to test the line search SQP method with the
        n-dimension Rosenbrock function.
        '''
        def fun(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return rosen(x), rosen_der(x)

        def restr(x: np.ndarray) -> tuple[float, np.ndarray]:
            '''
            Evaluate the restrictions of the problem: c(x) >= 0.

            Since we want that 0 <= x_i <= 1 for every i, then
                c(x) = [x_1, x_2, ..., x_n, 1 - x_1, 1 - x_2, ..., 1 - x_n]
            '''
            c = np.concatenate([x, 1 - x])
            A = np.concatenate([np.eye(x.size), -np.eye(x.size)])
            return c, A
        
        x0 = np.array([2.]*n)

        x, lam = ls_sqp(
            fun=fun,
            restr=restr,
            x_0=x0,
            lam_0=np.ones(2 * x0.size),
            B_0=np.eye(x0.size),
            hessian=hessian,
            eta=0.4,
            tau=0.7,
            maxiters=60,
            tol=0.005
        )

        npt.assert_allclose(x, np.ones(n), atol=tol)

    def test_ls_sqp_rosenbrock_10(self):
        '''Test line search SQP with 10-dimension Rosenbrock function'''
        self.perform_rosenbrock_ls_sqp(10, 0.1)

    def test_ls_sqp_rosenbrock_20(self):
        '''Test line search SQP with 20-dimension Rosenbrock function'''
        self.perform_rosenbrock_ls_sqp(20, 0.1)

    def test_ls_sqp_rosenbrock_50(self):
        '''Test line search SQP with 50-dimension Rosenbrock function'''
        self.perform_rosenbrock_ls_sqp(50, 0.1)

    def test_ls_sqp_rosenbrock_100(self):
        '''Test line search SQP with 100-dimension Rosenbrock function'''
        self.perform_rosenbrock_ls_sqp(100, 0.1)

    def test_ls_sqp_rosenbrock_200(self):
        '''Test line search SQP with 200-dimension Rosenbrock function'''
        self.perform_rosenbrock_ls_sqp(200, 0.1)

    def test_ls_sqp_rosenbrock_100_true_hess(self):
        '''
        Test line search SQP with 100-dimension Rosenbrock function using the
        true hessian. We can do this since the restrictions are linear, so the
        hessian of the lagrangian is the same as the hessian of f
        '''
        self.perform_rosenbrock_ls_sqp(100, 0.1, hessian=rosen_hess)


if __name__ == "__main__":
    unittest.main()
