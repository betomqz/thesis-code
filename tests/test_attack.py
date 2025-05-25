import unittest
import numpy as np
import logging
from opt_attack.attack import Attack, OptimusAttack, SciPyAttack, Dist, SUCCESS, ObjectFun
from keras import models
from opt_attack import utils
from pathlib import Path


SHOW_CONSOLE = True
LOGGER_FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if SHOW_CONSOLE:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))

    # Add console handler to root logger
    logger.addHandler(console_handler)


class TestAttack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(7679448)
        cls.random_guess = np.random.rand(784)

        cls.model = utils.load_mnist_model()
        cls.softmaxmodel = models.load_model('models/softmaxmnist.keras')
        x_train, x_test, y_train, y_test = utils.load_mnist_data()
        # Choose inputs
        cls.inputs = [
            (0, x_test[3]),
            (1, x_test[2]),
            (2, x_test[1]),
            (3, x_test[18]),
            (4, x_test[4]),
            (5, x_test[8]),
            (6, x_test[11]),
            (7, x_test[0]),
            (8, x_test[61]),
            (9, x_test[7])
        ]

        # Directory to save the logs of each test
        cls.parent_path = Path('logs/test_attack/')
        cls.parent_path.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        '''Set up a log file for each test case'''
        # Get the name and remove the 'test_' part
        self.test_name = self.id().split('.')[-1][5:]
        self.log_path = self.parent_path.joinpath(self.test_name)
        self.log_path.mkdir(exist_ok=True) # Parent dir should be created by now

        # File handler & format
        self.file_handler = logging.FileHandler(
            filename=self.log_path.joinpath('attack.log'),
            mode='w'
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(logging.Formatter(LOGGER_FORMAT))

        # Add file handler to root logger
        logger.addHandler(self.file_handler)

    def tearDown(self):
        '''Remove file handler from root logger after test'''
        logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def _perform_singleton(self, attacker: Attack, obj_fun, c=1.0):
        '''Helper function to run singleton attacks'''
        # Original class will be 0
        og_class, og_input = self.inputs[0]

        # Random start
        initial_guess = self.random_guess

        result = attacker.singleton_attack(
            original_input=og_input,
            original_class=og_class,
            target_class=1,
            initial_guess=initial_guess,
            obj_fun=obj_fun,
            c=c,
        )
        attacker.save(path=self.log_path)

        return result == SUCCESS

    def _perform_bin_search(self, attacker: Attack, obj_fun: ObjectFun, c_left=0.01, c_right=2.0):
        '''Helper function to run binary search attacks'''
        # Original class will be 0

        og_class, og_input = self.inputs[0]

        # Start close to the target
        initial_guess = np.clip(
            self.inputs[og_class][1].flatten() + 0.1 * self.random_guess,
            0., 1.
        )

        attacker.binary_search_attack(
            original_input=og_input,
            original_class=og_class,
            target_class=1,
            initial_guess=initial_guess,
            obj_fun=obj_fun,
            maxiters_bs=5,
            c_left=c_left,
            c_right=c_right
        )
        attacker.save(path=self.log_path)

        # See if test passed and log
        result = utils.eval_flat_pred(
            attacker.res['x'],
            model=self.softmaxmodel if obj_fun.needs_softmax() else self.model
        )
        return result == 1

    def _perform_parallel_attack(self, attacker: Attack, obj_fun: ObjectFun, c_start=0.01, c_stop=1.0, c_num=10):
        '''Helper function to run parallel attacks'''
        # Original class will be 0

        og_class, og_input = self.inputs[0]

        # Start close to the target
        initial_guess = np.clip(
            self.inputs[og_class][1].flatten() + 0.1 * self.random_guess,
            0., 1.
        )

        attacker.parallel_attack(
            original_input=og_input,
            original_class=og_class,
            target_class=1,
            initial_guess=initial_guess,
            obj_fun=obj_fun,
            c_start=c_start,
            c_stop=c_stop,
            c_num=c_num
        )
        attacker.save(path=self.log_path)

        # See if test passed and log
        result = utils.eval_flat_pred(
            attacker.res['x'],
            model=self.softmaxmodel if obj_fun.needs_softmax() else self.model
        )
        return result == 1

    # Tests using Optimus ------------------------------------------------------
    def test_optimus_szegedy_bin_search_L2(self):
        '''
        Test attack using SQP, Szegedy's objective function, binary search, and
        L2 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.softmaxmodel,
            distance=Dist.L2,
            maxiters_method=50,
            tol=0.1
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.szegedy)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_szegedy_bin_search_L1(self):
        '''
        Test attack using SQP, Szegedy's objective function, binary search, and
        L1 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.softmaxmodel,
            distance=Dist.L1,
            maxiters_method=20,
            tol=0.1
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.szegedy)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_szegedy_parallel_L2(self):
        '''
        Test attack using SQP, Szegedy's objective function, parallel attack,
        and L2 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.softmaxmodel,
            distance=Dist.L2,
            maxiters_method=50,
            tol=0.1
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.szegedy, c_num=5
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_szegedy_parallel_L1(self):
        '''
        Test attack using SQP, Szegedy's objective function, parallel attack,
        and L1 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.softmaxmodel,
            distance=Dist.L1,
            maxiters_method=20,
            tol=0.1
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.szegedy, c_num=5
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_carlini_bin_search_L2(self):
        '''
        Test attack using SQP, Carlini's objective function, binary search, and
        L2 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.model,
            distance=Dist.L2,
            maxiters_method=20,
            tol=0.2
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.carlini)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_carlini_bin_search_L1(self):
        '''
        Test attack using SQP, Carlini's objective function, binary search, and
        L1 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.model,
            distance=Dist.L1,
            maxiters_method=20,
            tol=0.2
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.carlini)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_carlini_parallel_L2(self):
        '''
        Test attack using SQP, Carlini's objective function, parallel attack,
        and L2 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.model,
            distance=Dist.L2,
            maxiters_method=20,
            tol=0.1
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.carlini, c_stop=0.35, c_num=2
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_optimus_carlini_parallel_L1(self):
        '''
        Test attack using SQP, Carlini's objective function, parallel attack,
        and L1 distance.
        '''
        logger.info('START')
        attacker = OptimusAttack(
            self.model,
            distance=Dist.L1,
            maxiters_method=20,
            tol=0.1
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.carlini, c_num=5
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    # Tests using SciPy --------------------------------------------------------
    def test_scipy_szegedy_singleton_L2(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Szegedy's objective function,
        singleton attack, and L2 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.softmaxmodel,
            distance=Dist.L2,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_singleton(attacker=attacker, obj_fun=ObjectFun.szegedy, c=0.01)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_szegedy_bin_search_L2(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Szegedy's objective function,
        binary search, and L2 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.softmaxmodel,
            distance=Dist.L2,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.szegedy)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_szegedy_bin_search_L1(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Szegedy's objective function,
        binary search, and L1 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.softmaxmodel,
            distance=Dist.L1,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.szegedy)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_szegedy_parallel_L2(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Szegedy's objective function,
        parallel attack, and L2 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.softmaxmodel,
            distance=Dist.L2,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.szegedy, c_stop=10.0
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_szegedy_parallel_L1(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Szegedy's objective function,
        parallel attack, and L1 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.softmaxmodel,
            distance=Dist.L1,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.szegedy, c_stop=1.0
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_carlini_bin_search_L2(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Carlini's objective function,
        binary search, and L2 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.model,
            distance=Dist.L2,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_bin_search(
            attacker=attacker, obj_fun=ObjectFun.carlini, c_left=0.2, c_right=1.0
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_carlini_bin_search_L1(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Carlini's objective function,
        binary search, and L1 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.model,
            distance=Dist.L1,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_bin_search(attacker=attacker, obj_fun=ObjectFun.carlini)
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_carlini_parallel_L2(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Carlini's objective function,
        parallel attack, and L2 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.model,
            distance=Dist.L2,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.carlini, c_stop=1.0
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)

    def test_scipy_carlini_parallel_L1(self):
        '''
        Test attack using SciPy's L-BFGS-B method, Carlini's objective function,
        parallel attack, and L1 distance.
        '''
        logger.info('START')
        attacker = SciPyAttack(
            self.model,
            distance=Dist.L1,
            method='L-BFGS-B',
            options={'maxiter':2000, 'disp':0}
        )
        passed = self._perform_parallel_attack(
            attacker=attacker, obj_fun=ObjectFun.carlini, c_stop=10.0
        )
        if passed:
            logger.info("The attack was successful")
        else:
            logger.warning("The attack was not successful")
        logger.info("END")
        self.assertTrue(passed)
