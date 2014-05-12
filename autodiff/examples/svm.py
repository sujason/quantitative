"""
Linear SVM
==========

This script fits a linear support vector machine classifier to random data.  It
illustrates how a function defined purely by NumPy operations can be minimized
directly with a gradient-based solver.

"""
import numpy as np
from autodiff.optimize import fmin_l_bfgs_b


def test_svm():
    rng = np.random.RandomState(1)

    # -- create some fake data
    x = rng.rand(10, 5)
    y = 2 * (rng.rand(10) > 0.5) - 1
    l2_regularization = 1e-4

    # -- loss function
    def loss_fn(weights, bias):
        margin = y * (np.dot(x, weights) + bias)
        loss = np.maximum(0, 1 - margin) ** 2
        l2_cost = 0.5 * l2_regularization * np.dot(weights, weights)
        loss = np.mean(loss) + l2_cost
        print 'ran loss_fn(), returning', loss
        return loss

    # -- call optimizer
    w_0, b_0 = np.zeros(5), np.zeros(())
    w, b = fmin_l_bfgs_b(loss_fn, init_args=(w_0, b_0))

    final_loss = loss_fn(w, b)

    assert np.allclose(final_loss, 0.7229)

    print 'optimization successful!'


if __name__ == '__main__':
    test_svm()
