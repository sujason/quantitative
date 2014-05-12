import unittest
import numpy as np

from autodiff.decorators import function, gradient, hessian_vector
from autodiff.functions import tag

#========= Tests


class TestFunction(unittest.TestCase):
    def test_basic_fn(self):
        @function
        def fn(x):
            return x
        self.assertTrue(np.allclose(fn(3), 3))
        self.assertTrue(len(fn.cache) == 1)

    def test_fn_cache(self):
        @function
        def fn(x):
            return x

        self.assertTrue(np.allclose(fn(3), 3))

        # check that fn was cached
        self.assertTrue(len(fn.cache) == 1)

        # check that arg of new input dim was cached
        fn(np.ones(10))
        self.assertTrue(len(fn.cache) == 2)

        # check that another arg of same input dim was not cached
        fn(np.ones(10) + 15)
        self.assertTrue(len(fn.cache) == 2)


class TestGradient(unittest.TestCase):
    def test_basic_grad(self):
        @gradient
        def fn(x):
            return x
        self.assertTrue(np.allclose(fn(3), 1.0))

    def test_nonscalar_grad(self):
        @gradient
        def fn(x):
            return x
        self.assertRaises(TypeError, fn, np.ones(1))

    def test_grad_wrt(self):
        @gradient(wrt='x')
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), 5.0))

        @gradient(wrt=('x', 'y'))
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), [5.0, 3.0]))

        @gradient(wrt=('y', 'x'))
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), [3.0, 5.0]))

        @gradient()
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(3.0, 5.0), [5.0, 3.0]))

        a = np.array(3.0)
        b = np.array(5.0)

        @gradient(wrt=a)
        def f(x, y):
            return x * y
        self.assertTrue(np.allclose(f(a, 5.0), 5.0))

        @gradient(wrt=b)
        def f(x, y):
            return x * y
        self.assertRaises(ValueError, f, a, 5.0)
        self.assertTrue(np.allclose(f(a, b), 3.0))
        self.assertTrue(np.allclose(f(3.0, b), 3.0))

        @gradient(wrt=(a, b))
        def f(x, y):
            return x * y
        self.assertRaises(ValueError, f, a, 5.0)
        self.assertTrue(np.allclose(f(a, b), [5.0, 3.0]))


class TestHV(unittest.TestCase):
    def test_hv_missing_vectors(self):
        @hessian_vector
        def fn(x):
            return x
        self.assertRaises(ValueError, fn, np.array([[1, 1]]))

    def test_hv_no_scalar(self):
        @hessian_vector
        def fn(x):
            return np.dot(x, x)
        x = np.ones((3, 3))
        self.assertRaises(TypeError, fn, x, _vectors=x[0])

    def test_hv(self):
        @hessian_vector
        def fn(x):
            return np.dot(x, x).sum()
        x = np.ones((3, 3))
        self.assertTrue(np.allclose(x * 6, fn(x, _vectors=x)))
        self.assertTrue(np.allclose(x * 2, fn(x[0], _vectors=x[0])))


class TestClass(unittest.TestCase):
    def setUp(self):
        class AutoDiff(object):

            a = 100.0

            def __init__(self):
                self.x = 100.0
                self.y = np.ones((3, 4))
                self.grad_f4 = gradient(self.f4, wrt=self.y)

            @function
            def f1(self):
                return self.x + self.y

            @function
            def f2(self, x):
                return self.x + x

            @gradient
            def f3(self, x):
                return np.sum(self.y + x)

            @gradient(wrt='y')
            def f4(self, x):
                return np.sum(tag(self.y, 'y') * x)

            def passthrough(self, *args, **kwargs):
                return self.f2(*args, **kwargs)

            @function
            def passthrough_fn(self, *args, **kwargs):
                return self.f2(*args, **kwargs)

            @classmethod
            @function
            def class_method(cls, x):
                return cls.a + x

            @staticmethod
            @function
            def static_method(x):
                return x + 100.0

        self.AD = AutoDiff()

    def test_decorated_method(self):
        self.assertTrue(np.allclose(self.AD.f1(), self.AD.x + self.AD.y))
        self.assertTrue(np.allclose(self.AD.f2(1.0), self.AD.x + 1.0))
        self.assertTrue(np.allclose(self.AD.f3(1.0), np.sum(self.AD.y)))
        self.assertTrue(np.allclose(self.AD.f4(2.0), self.AD.y * 2.0))
        self.assertTrue(np.allclose(self.AD.grad_f4(2.0), self.AD.y * 2.0))
        self.assertTrue(np.allclose(self.AD.passthrough(2.0), self.AD.f2(2.0)))
        self.assertTrue(np.allclose(self.AD.passthrough_fn(3.0),
                                    self.AD.f2(3.0)))

    def test_class_method(self):
        self.assertTrue(np.allclose(self.AD.class_method(1.0), 101.0))

    def test_static_method(self):
        self.assertTrue(np.allclose(self.AD.static_method(1.0), 101.0))
