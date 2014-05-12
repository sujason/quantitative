import unittest
import numpy as np
from autodiff.optimize import fmin_l_bfgs_b, fmin_cg, fmin_ncg


def L2(x, y):
    return ((x - y) ** 2).mean()


def l2_loss(p):
    l2_x = np.arange(20).reshape(4, 5)
    l2_b = np.arange(3) - 1.5
    loss = L2(np.dot(l2_x, p) - l2_b, np.arange(3))
    return loss


def simple_loss_multiple_args(p, q):
    p_err = ((p - np.arange(2.))**2).sum()
    q_err = ((q - np.arange(3.))**2).sum()
    return p_err + q_err


def subtensor_loss(x):
    if np.any(x < -100):
        return float('inf')
    x2 = x.copy()
    x2[0] += 3.0
    x2[1] -= 4.0
    rval = (x2 ** 2).sum()
    rval += 1.3
    rval *= 1.0
    return rval


class TestOptimizers(unittest.TestCase):
    def test_subtensor(self):
        x0 = np.zeros(2)
        ans = [-3, 4]

        opt = fmin_l_bfgs_b(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_cg(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_ncg(subtensor_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

    def test_L2(self):
        x0 = np.zeros((5, 3))
        ans = np.array([[+3.0, -1.0, -5.0],
                        [+1.5, -0.5, -2.5],
                        [+0.0,  0.0,  0.0],
                        [-1.5,  0.5,  2.5],
                        [-3.0,  1.0,  5.0]]) / 10.0

        opt = fmin_l_bfgs_b(l2_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_cg(l2_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

        opt = fmin_ncg(l2_loss, x0)
        self.assertTrue(np.allclose(opt, ans))

    def test_simple_loss_multiple_args(self):
        # test that args and kwargs are both handled by optimizers
        x0 = np.zeros(2), np.zeros(3)
        ans = np.arange(2.), np.arange(3.)

        opt = fmin_l_bfgs_b(simple_loss_multiple_args, x0)
        opt = fmin_l_bfgs_b(simple_loss_multiple_args,
                            init_kwargs=dict(p=x0[0], q=x0[1]))
        opt = fmin_l_bfgs_b(simple_loss_multiple_args,
                            init_args=x0[0],
                            init_kwargs=dict(q=x0[1]))
        self.assertTrue(np.allclose(opt[0], ans[0]))
        self.assertTrue(np.allclose(opt[1], ans[1]))

        opt = fmin_cg(simple_loss_multiple_args, x0)
        opt = fmin_cg(simple_loss_multiple_args,
                      init_kwargs=dict(p=x0[0], q=x0[1]))
        opt = fmin_cg(simple_loss_multiple_args,
                      init_args=x0[0],
                      init_kwargs=dict(q=x0[1]))
        self.assertTrue(np.allclose(opt[0], ans[0]))
        self.assertTrue(np.allclose(opt[1], ans[1]))

        opt = fmin_ncg(simple_loss_multiple_args, x0)
        opt = fmin_ncg(simple_loss_multiple_args,
                       init_kwargs=dict(p=x0[0], q=x0[1]))
        opt = fmin_ncg(simple_loss_multiple_args,
                       init_args=x0[0],
                       init_kwargs=dict(q=x0[1]))
        self.assertTrue(np.allclose(opt[0], ans[0]))
        self.assertTrue(np.allclose(opt[1], ans[1]))


class TestSVM(unittest.TestCase):
    """
    adopted from pyautodiff v0.0.1 tests.

    should correspond to examples/svm.py
    """
    def test_svm(self):
        rng = np.random.RandomState(1)

        # -- create some fake data
        x = rng.rand(10, 5)
        y = 2 * (rng.rand(10) > 0.5) - 1
        l2_regularization = 1e-4

        def loss_fn(weights, bias):
            margin = y * (np.dot(x, weights) + bias)
            loss = np.maximum(0, 1 - margin) ** 2
            l2_cost = 0.5 * l2_regularization * np.dot(weights, weights)
            loss = np.mean(loss) + l2_cost
            print 'ran loss_fn(), returning', loss
            return loss

        w, b = fmin_l_bfgs_b(loss_fn, (np.zeros(5), np.zeros(())))
        final_loss = loss_fn(w, b)
        assert np.allclose(final_loss, 0.7229)
