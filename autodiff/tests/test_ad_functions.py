import unittest
import numpy as np

from autodiff.functions import constant, tag
from autodiff.symbolic import Function
from autodiff.decorators import function


def check(fn, *args, **kwargs):
    F = Function(fn)
    py_result = fn(*args, **kwargs)
    sym_result = F(*args, **kwargs)
    return np.allclose(py_result, sym_result)


class TestConstant(unittest.TestCase):
    def test_range(self):
        def f(x):
            for i in range(3):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x):
            for i in range(x):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x, r):
            for i in range(r):
                x = x + x
            return x
        self.assertTrue(check(f, np.ones(3), 3))

        def f(x):
            for i in range(constant(3)):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x):
            for i in range(constant(x)):
                x = x + x
            return x
        self.assertTrue(check(f, 1))

        def f(x, r):
            for i in range(constant(r)):
                x = x + x
            return x
        self.assertTrue(check(f, np.ones(3), 3))

    def test_sum(self):
        def f(x):
            return x.sum(1)
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x):
            a = 1
            return x.sum(a)
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x, a):
            return x.sum(a)
        self.assertRaises(TypeError, check, f, np.ones((3, 4)), 1)

        def f(x):
            a = np.int_(1)
            return x.sum(a)
        self.assertRaises(TypeError, check, f, np.ones((3, 4)))

        def f(x):
            return x.sum(constant(1))
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x):
            a = constant(1)
            return x.sum(a)
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x):
            a = 1
            return x.sum(constant(a))
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x):
            a = np.int_(1)
            return x.sum(constant(a))
        self.assertTrue(check(f, np.ones((3, 4))))

        def f(x, a):
            return x.sum(constant(a))
        self.assertTrue(check(f, np.ones((3, 4)), 1))

    def test_closure_sum(self):
        a = 1

        def f(x):
            return x.sum(a)
        self.assertTrue(check(f, np.ones((3, 4))))

    def test_float_range(self):
        import theano
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float32'

        def f():
            return np.arange(5.0)
        self.assertTrue(check(f))

        theano.config.floatX = old_floatX


class TestTag(unittest.TestCase):
    def test_tag(self):
        def f(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        F = Function(f)
        self.assertFalse('y' in F.s_vars)
        F(10)
        self.assertTrue('y' in F.s_vars)

    def test_tag_decorator(self):
        @function
        def F(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        self.assertFalse('y' in F.s_vars)
        F(10)
        self.assertTrue('y' in F.s_vars)
