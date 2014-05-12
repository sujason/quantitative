import unittest
import numpy as np
import theano.tensor

from autodiff.symbolic import Symbolic, Function, Gradient
from autodiff import tag


def checkfn(symF, *args, **kwargs):
    py_result = symF.pyfn(*args, **kwargs)
    ad_result = symF(*args, **kwargs)
    return np.allclose(ad_result, py_result)


#========= Tests


class TestFunction(unittest.TestCase):
    def test_sig_no_arg(self):
        # single arg, no default
        def fn():
            return np.ones((3, 4)) + 2
        f = Function(fn)
        self.assertTrue(checkfn(f))

    def test_sig_one_arg(self):
        # single arg, no default
        def fn(x):
            return x
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a=2)
        self.assertTrue(checkfn(f, 2))
        self.assertTrue(checkfn(f, x=2))

    def test_sig_mult_args(self):
        # multiple args, no default
        def fn(x, y):
            return x * y
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, 2)
        self.assertRaises(TypeError, f, a=2, b=2)
        self.assertTrue(checkfn(f, 2, 3))
        self.assertTrue(checkfn(f, y=4, x=5))

    def test_sig_var_args(self):
        # var args, no default
        def fn(x, y, *z):
            return x * y * sum(z)
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, 2)
        self.assertRaises(TypeError, f, a=2, b=2)
        self.assertTrue(checkfn(f, 2, 3))
        self.assertTrue(checkfn(f, 2, 3, 4))
        self.assertTrue(checkfn(f, 2, 3, 4, 5))

        # make sure function recompiles for different numbers of varargs
        f = Function(fn)
        self.assertTrue(checkfn(f, 2, 3, 4, 5, 6))
        self.assertTrue(checkfn(f, 2, 3, 4))
        self.assertTrue(checkfn(f, 2, 3, 4, 5))

    def test_sig_default_args(self):
        # multiple args, one default
        def fn(x, y=2):
            return x * y
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, y=3)
        self.assertTrue(checkfn(f, 2))
        self.assertTrue(checkfn(f, 2, 3))
        self.assertTrue(checkfn(f, y=4, x=5))
        self.assertTrue(checkfn(f, x=5))

        # multiple args, all default
        def fn(x=1, y=2):
            return x * y
        f = Function(fn)
        self.assertTrue(checkfn(f))
        self.assertTrue(checkfn(f, 1))
        self.assertTrue(checkfn(f, 1, 2))
        self.assertTrue(checkfn(f, y=2, x=1))
        self.assertTrue(checkfn(f, x=5))
        self.assertTrue(checkfn(f, y=5))

    def test_sig_default_var_args(self):
        # multiple var args, all default
        def fn(x=1, y=2, *z):
            return x * y * sum(z)
        f = Function(fn)
        self.assertTrue(checkfn(f))
        self.assertTrue(checkfn(f, 1))
        self.assertTrue(checkfn(f, 1, 2))
        self.assertTrue(checkfn(f, 1, 2, 3))
        self.assertTrue(checkfn(f, 1, 2, 3, 4))

    def test_sig_kwargs(self):
        # kwargs
        def fn(**kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z
        f = Function(fn)
        self.assertRaises(KeyError, f)
        self.assertRaises(TypeError, f, 1)
        self.assertTrue(checkfn(f, x=1, y=2, z=3))

    def test_sig_varargs_kwargs(self):
        # varargs and kwargs
        def fn(a, *b, **kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z
        f = Function(fn)
        self.assertRaises(TypeError, f)
        self.assertRaises(KeyError, f, 1)
        self.assertRaises(TypeError, f, x=1, y=2, z=3)
        self.assertTrue(checkfn(f, 1, x=1, y=2, z=3))
        self.assertTrue(checkfn(f, 1, 2, 3, x=1, y=2, z=3))

        # varargs and kwargs, use varargs
        def fn(a, *b, **kwargs):
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            return x * y * z * b[0]
        f = Function(fn)
        self.assertTrue(checkfn(f, 1, 2, x=1, y=2, z=3))
        self.assertTrue(checkfn(f, 1, 2, 3, x=1, y=2, z=3))

    def test_nested_fn_call(self):
        def f(x, y):
            return x + y

        def g(x):
            return f(x, x + 1) - x ** 2

        h = Function(g)
        self.assertTrue(checkfn(h, 10))

    def test_nested_fn_def(self):

        def g(x):
            def f(x, y):
                return x + y
            return f(x, x + 1) - x ** 2

        h = Function(g)
        self.assertTrue(checkfn(h, 10))

    def test_nested_fn_kwargs_call(self):
        def f(x, y):
            return x + y

        def g(x):
            return f(y=x, x=x+1) - x ** 2

        h = Function(g)
        self.assertTrue(checkfn(h, 10))

    def test_nested_fn_kwargs_def(self):
        def g(x):
            def f(x, y):
                return x + y
            return f(y=x, x=x+1) - x ** 2

        h = Function(g)
        self.assertTrue(checkfn(h, 10))

    def test_fn_constants(self):
        # access constant array
        def fn(x):
            return np.dot(x, np.ones((3, 4)))
        f = Function(fn)
        self.assertTrue(checkfn(f, np.ones((2, 3))))

    def test_caching(self):
        def fn(x, switch):
            if switch > 0:
                return x * 1
            else:
                return x * 0

        f_cached = Function(fn, use_cache=True)
        c_result_1 = f_cached(1, 1)
        c_result_2 = f_cached(1, -1)

        self.assertTrue(np.allclose(c_result_1, 1))
        self.assertTrue(np.allclose(c_result_2, 1))

        f_uncached = Function(fn, use_cache=False)
        uc_result_1 = f_uncached(1, 1)
        uc_result_2 = f_uncached(1, -1)

        self.assertTrue(np.allclose(uc_result_1, 1))
        self.assertTrue(np.allclose(uc_result_2, 0))

    def test_function_of_function(self):
        # single arg, no default
        def fn():
            return np.ones((3, 4)) + 2
        f = Function(fn)
        f2 = Function(f)
        self.assertTrue(checkfn(f2))

    def test_function_of_nested_vargs_kwargs(self):
        def fn(*args, **kwargs):
            return args[1] + kwargs['kw']

        def fn2(*args, **kwargs):
            return fn(*args, **kwargs)

        f = Function(fn2)
        self.assertTrue(checkfn(f, 1.0, 2.0, kw=3.0, kw2=4.0))

    def test_function_of_nested_def_vargs_kwargs(self):
        def fn2(*args, **kwargs):
            def fn(*args, **kwargs):
                return args[1] + kwargs['kw']
            return fn(*args, **kwargs)

        f = Function(fn2)
        self.assertTrue(checkfn(f, 1.0, 2.0, kw=3.0, kw2=4.0))

    def test_dict_arg(self):
        def f(x):
            return x + 1

        def g(x):
            return f(x[1])

        F = Function(g)
        self.assertTrue(checkfn(F, {1.0: 5.0}))


class TestGradient(unittest.TestCase):
    def test_simple_gradients(self):
        # straightforward gradient
        g = Gradient(lambda x: x ** 2)
        self.assertTrue(np.allclose(g(3.0), 6))

        # gradient of two arguments
        fn = lambda x, y: x * y
        g = Gradient(fn)
        self.assertTrue(np.allclose(g(3.0, 5.0), [5.0, 3.0]))

    def test_wrt(self):
        fn = lambda x, y: x * y

        # function of two arguments, gradient of one
        g = Gradient(fn, wrt='x')
        self.assertTrue(np.allclose(g(3.0, 5.0), 5.0))

        g = Gradient(fn, wrt='y')
        self.assertTrue(np.allclose(g(3.0, 5.0), 3.0))

        # test object tracking for wrt
        a = 3.0
        b = 5.0
        g = Gradient(fn, wrt=[a, b])
        self.assertTrue(np.allclose(g(a, b), [b, a]))

        g = Gradient(fn, wrt=a)
        self.assertTrue(np.allclose(g(a, b), b))

        g = Gradient(fn, wrt=b)
        self.assertTrue(np.allclose(g(a, b), a))


class TestSymbolic(unittest.TestCase):
    def test_symbolic(self):
        def f1(x):
            return x + 1.0

        def f2(x):
            return x * 2.0

        def f3(x):
            return x ** 2
        s = Symbolic()
        x = np.random.random((3, 4))
        o1 = s.trace(f1, x)
        o2 = s.trace(f2, o1)
        o3 = s.trace(f3, o2)

        # test function
        f = s.compile_function(x, o3)
        self.assertTrue(np.allclose(f(x), f3(f2(f1(x)))))

        # test gradient
        o4 = s.trace(lambda x: x.sum(), o3)
        g = s.compile_gradient(x, o4, wrt=x)
        self.assertTrue(np.allclose(g(x), 8 * (x+1)))

    def test_symbolic_readme(self):
        """ the README example"""

        # -- a vanilla function
        def f1(x):
            return x + 2

        # -- a function referencing a global variable
        y = np.random.random(10)

        def f2(x):
            return x * y

        # -- a function with a local variable
        def f3(x):
            z = tag(np.ones(10), 'local_var')
            return (x + z) ** 2

        # -- create a general symbolic tracer and apply
        #    it to the three functions
        x = np.random.random(10)
        tracer = Symbolic()

        out1 = tracer.trace(f1, x)
        out2 = tracer.trace(f2, out1)
        out3 = tracer.trace(f3, out2)

        # -- compile a function representing f(x, y, z) = out3
        new_fn = tracer.compile_function(inputs=[x, y, 'local_var'],
                                         outputs=out3)

        # -- compile the gradient of f(x) = out3, with respect to y
        fn_grad = tracer.compile_gradient(inputs=x,
                                          outputs=out3,
                                          wrt=y,
                                          reduction=theano.tensor.sum)

        assert fn_grad  # to stop flake error

        self.assertTrue(np.allclose(new_fn(x, y, np.ones(10)), f3(f2(f1(x)))))

    def test_class(self):
        class Test(object):
            def f(self, x):
                return x + 100.0

            @classmethod
            def g(cls, x):
                return x + 100.0

            @staticmethod
            def h(x):
                return x + 100.0

        t = Test()
        s = Symbolic()
        x = 1.0
        o = s.trace(t.f, x)
        f = s.compile_function(x, o)
        assert(f(2.0) == 102.0)

        o = s.trace(t.g, x)
        f = s.compile_function(x, o)
        assert(f(2.0) == 102.0)

        o = s.trace(t.h, x)
        f = s.compile_function(x, o)
        assert(f(2.0) == 102.0)
