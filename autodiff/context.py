"""
Example of how to use byte-code execution technique to trace accesses to numpy
arrays.

This file demonstrates two applications of this technique:
* optimize numpy computations for repeated calling
* provide automatic differentiation of procedural code

"""

import __builtin__
import ctypes
import inspect
import logging
import opcode
#import os
import sys
#import trace
import traceback
import types

import numpy as np
import theano

import autodiff
from autodiff.utils import itercode, orderedcallargs, flat_from_doc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# XXX FIXME This will not do - seed must be exposed.
global_randomstreams = RandomStreams(seed=123)


# Opcode help: http://docs.python.org/library/dis.html

# -- cellget returns the contents of a cell
cellget = ctypes.pythonapi.PyCell_Get
cellget.restype = ctypes.py_object
cellget.argtypes = (ctypes.py_object,)

# -- cellmake creates a cell pointer
cellmake = ctypes.pythonapi.PyCell_New
cellmake.restype = ctypes.py_object
cellmake.argtypes = (ctypes.py_object,)


def istensor(x):
    tensortypes = (theano.tensor.TensorConstant,
                   theano.tensor.TensorVariable)
    return isinstance(x, tensortypes)


class Unassigned(object):
    """Unassigned value"""


class LoadUnassigned(Exception):
    """Access to Unassigned value"""


class FrameVM(object):
    """
    A Class for evaluating a code block of CPython bytecode,
    and tracking accesses to numpy arrays.

    """
    def __init__(self, watcher, func):
        logger.debug('FrameVM: {0}'.format(func))
        self.watcher = watcher
        if isinstance(func, autodiff.symbolic.Function):
            func = func.pyfn
        self.func = func
        self.stack = []
        self._locals = None
        self._myglobals = None
        self.code_iter = None
        self.print_ops = False
        self.print_stack = False

    def push(self, item):
        if item is Unassigned:
            raise LoadUnassigned()
        self.stack.append(item)

    def pop(self):
        return self.stack.pop(-1)

    def pushN(self, items):
        for item in items:
            if item is Unassigned:
                raise LoadUnassigned()
        self.stack.extend(items)

    def popN(self, N):
        rval = self.stack[-N:]
        self.stack[-N:] = []
        return rval

    def add_shadow(self, x):
        if id(x) in self.watcher.constants:
            return
        # -- We cannot safely set up shadow variables that are aliased to
        #    memory that is visible to the running program, unless that
        #    program can guarantee that all views of that memory are
        #    immutable. CPython caches small ints (-5 <= i <= 256), so
        #    we wrap them in a non-cached _int() instance.
        if isinstance(x, int):
            if type(x) is int and -5 <= x <= 256:
                x = np.int_(x)
            s_x = self.watcher.shared(np.asarray(x))
        elif isinstance(x, float):
            s_x = self.watcher.shared(np.asarray(x))
        elif getattr(x, 'dtype', None) == bool:
            print >> sys.stderr, ('Warning: Theano has no bool, '
                                  'upgrading to int8')
            s_x = self.watcher.shared(x.astype('int8'))
        elif isinstance(x, (np.ndarray, np.number)):
            s_x = self.watcher.shared(x)
        else:
            return
        self.watcher.shadow(x, s_x)

    def ensure_shadow(self, x):
        # small ints can not be shadowed due to CPython memory caching, so we
        # wrap them in non-cached _ints.
        if type(x) is int and -5 <= x <= 256:
            x = np.int_(x)
        if id(x) not in self.watcher:
            self.add_shadow(x)
        return self.watcher.getvar(x)

    def call(self, args, kwargs):
        if not isinstance(args, tuple):
            raise TypeError('vm.call: args must be tuple', args)
        if not isinstance(kwargs, dict):
            raise TypeError('vm.call: kwargs must be dict', kwargs)

        func = self.func

        if isinstance(func, type) and issubclass(func, BaseException):
            # XXX not shadowing exception creation, because exceptions
            # do not have func_code. Is this OK? can we do better?
            return func(*args, **kwargs)

        func_code = self.func.func_code

        self._myglobals = {}
        self._locals = []

        for name in func_code.co_names:
            #print 'name', name
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                try:
                    self._myglobals[name] = __builtin__.__getattribute__(name)
                except AttributeError:
                    #print 'WARNING: name lookup failed', name
                    pass

        # get function arguments
        argspec = inspect.getargspec(func)

        # match function arguments to passed parameters
        callargs = orderedcallargs(func, *args, **kwargs)

        # named args => locals
        self._locals.extend(callargs[arg] for arg in argspec.args)

        # *args => locals
        if argspec.varargs:
            self._locals.append(callargs[argspec.varargs])

        # **kwargs => locals
        if argspec.keywords:
            self._locals.append(callargs[argspec.keywords])

        # other vars => locals
        no_unbound_args = len(func_code.co_varnames) - len(self._locals)
        self._locals.extend([Unassigned] * no_unbound_args)

        # shadow arguments
        for val in flat_from_doc(callargs):
            if id(val) not in self.watcher:
                self.add_shadow(val)

        self.code_iter = itercode(func_code.co_code)
        jmp = None
        while not hasattr(self, 'rval'):
            try:
                i, op, arg = self.code_iter.send(jmp)
            except StopIteration:
                break
            name = opcode.opname[op]
            # method names can't have '+' in them
            name = {'SLICE+0': 'SLICE_PLUS_0',
                    'SLICE+1': 'SLICE_PLUS_1',
                    'SLICE+2': 'SLICE_PLUS_2',
                    'SLICE+3': 'SLICE_PLUS_3',
                    'STORE_SLICE+0': 'STORE_SLICE_PLUS_0',
                    'STORE_SLICE+1': 'STORE_SLICE_PLUS_1',
                    'STORE_SLICE+2': 'STORE_SLICE_PLUS_2',
                    'STORE_SLICE+3': 'STORE_SLICE_PLUS_3',
                    }.get(name, name)
            if self.print_ops:
                print 'OP: ', i, name
            if self.print_stack:
                print self.stack
            try:
                op_method = getattr(self, 'op_' + name)
            except AttributeError:
                raise AttributeError('FrameVM does not have a method defined '
                                     'for \'op_{0}\''.format(name))
            except:
                raise
            jmp = op_method(i, op, arg)

        return self.rval

    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        # No Theano vars allowed on the stack
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 + arg2
        self.push(r)
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 + s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 + s2)
            #print 'added sym'

    def op_BINARY_DIVIDE(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 / arg2
        self.push(r)
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 / s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 / s2)

    def op_BINARY_FLOOR_DIVIDE(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 // arg2
        self.push(r)
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 // s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 // s2)

    def op_BINARY_SUBTRACT(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        r = arg1 - arg2
        self.push(r)
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 - s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 - s2)

    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        r = arg1 * arg2
        self.push(r)
        assert not hasattr(arg1, 'type')
        assert not hasattr(arg2, 'type')
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 * s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 * s2)
            #print 'mul sym', id(r)

    def op_BINARY_POWER(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        r = arg1 ** arg2
        self.push(r)
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2).astype(s1.dtype)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 ** s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 ** s2)
            #print 'mul sym', id(r)

    def op_BINARY_MODULO(self, i, op, arg):
        arg2 = self.pop()
        arg1 = self.pop()
        r = arg1 % arg2
        self.push(r)
        if (id(arg1) in self.watcher or id(arg2) in self.watcher):
            s1 = self.ensure_shadow(arg1)
            s2 = self.ensure_shadow(arg2)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s1 % s2).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s1 % s2)

    def op_BINARY_SUBSCR(self, i, op, arg):
        # Implements TOS = TOS1[TOS].
        tos1, tos = self.popN(2)
        #print 'tos', tos
        #print 'tos1', tos1
        rval = tos1[tos]
        self.push(rval)

        if id(tos) in self.watcher:
            s_tos = self.ensure_shadow(tos)
        else:
            s_tos = tos

        if id(tos1) in self.watcher:
            s_tos1 = self.ensure_shadow(tos1)
        else:
            s_tos1 = tos1

        if isinstance(tos, np.ndarray) and tos.dtype == bool:
            s_rval = s_tos1[s_tos.nonzero()]
        else:
            s_rval = s_tos1[s_tos]

        if id(tos) in self.watcher or id(tos1) in self.watcher:
            self.watcher.shadow(rval, s_rval)

    def op_BUILD_MAP(self, i, op, arg):
        self.push({})

    def op_BUILD_SLICE(self, i, op, arg):
        if arg == 2:
            tos1, tos = self.popN(2)
            self.push(slice(tos1, tos))
        elif arg == 3:
            tos2, tos1, tos = self.popN(3)
            self.push(slice(tos2, tos1, tos))
        else:
            raise NotImplementedError()

    def op_BUILD_TUPLE(self, i, op, arg):
        if arg:
            self.push(tuple(self.popN(arg)))
        else:
            self.push(())

    def op_BUILD_LIST(self, i, op, arg):
        if arg:
            self.push(list(self.popN(arg)))
        else:
            self.push([])

    def op_CALL_FUNCTION(self, i, op, arg, call_vargs=None, call_kwargs=None):
        if call_vargs is None:
            # -- these are the things passed with *foo syntax
            call_vargs = ()

        if call_kwargs is None:
            # -- these are the things passed with **foo syntax
            call_kwargs = {}

        n_args = arg & 0xFF
        n_kwargs = (arg & 0xFF00) >> 8
        #print 'N_ARGS', n_args, n_kwargs, call_vargs
        assert not (arg >> 16)  # what would this stuff up here mean?
        kwargs = dict([(self.stack[-2 * ii], self.stack[-2 * ii + 1])
                       for ii in range(n_kwargs, 0, -1)])
        args = [self.stack[-ii - 2 * n_kwargs] for ii in range(n_args, 0, -1)]
        assert all(Unassigned is not ai for ai in args)
        # -- pop all args off the stack
        if arg:
            self.stack = self.stack[:- n_args - 2 * n_kwargs]
        # -- pop the function itself off the stack
        func = self.pop()

        args = args + list(call_vargs)
        orig_kwargs_size = len(kwargs)
        kwargs.update(call_kwargs)
        assert len(kwargs) == orig_kwargs_size + len(call_kwargs)
        #print dir(func)
        #print func.__self__
        all_args = args + kwargs.values()

        # -- get symbolic args
        if len(call_vargs) > 0:
            s_args = [self.watcher.getvar(a) for a in args[:-len(call_vargs)]]
            s_args.extend(self.watcher.getvar(a) for a in call_vargs)
            s_args = tuple(s_args)
        else:
            s_args = tuple(self.watcher.getvar(a) for a in args)
        s_kwargs = dict([(kw, self.watcher.getvar(val))
                         for kw, val in kwargs.items()])

        if hasattr(func, '__theano_op__'):
            # XXX: document that we are assuming func is pure -
            #      if rval depends on globals or closure this Context is not
            #      going to know that.
            # -- hand control back to Python for duration of func
            rval = func(*args, **kwargs)
            if any(id(a) in self.watcher for a in all_args):
                s_rval = func.__theano_op__(*s_args, **s_kwargs)
                self.watcher.shadow(rval, s_rval)

        # ================ NumPy and builtin functions
        elif ((getattr(func, '__module__', None)
                and func.__module__.startswith('numpy'))
              or isinstance(func, np.ufunc)
              or str(func) == '<built-in function abs>'
              or str(func) == '<built-in function max>'
              or str(func) == '<built-in function min>'
              or str(func) == '<built-in function sum>'):

            rval = func(*args, **kwargs)
            if any(id(a) in self.watcher for a in all_args):
                if func.__name__ == 'sum':
                    if type(rval) == int:
                        rval = np.int_(rval)
                    s_rval = theano.tensor.sum(*s_args, **s_kwargs)
                    self.watcher.shadow(rval, s_rval)
                elif func.__name__ in ('abs', 'absolute'):
                    self.watcher.shadow(rval, abs(*s_args))
                elif func.__name__ == 'max':
                    assert str(func) == '<built-in function max>'
                    s_rval = theano.tensor.maximum(*s_args, **s_kwargs)
                    assert s_rval.ndim == 0  # builtin max can't make vector
                    self.watcher.shadow(rval, s_rval)
                elif func.__name__ == 'min':
                    assert str(func) == '<built-in function min>'
                    s_rval = theano.tensor.minimum(*s_args, **s_kwargs)
                    assert s_rval.ndim == 0  # builtin min can't make vector
                    self.watcher.shadow(rval, s_rval)
                elif func.__name__ == 'reshape':
                    self.watcher.shadow(
                        rval, theano.tensor.reshape(*s_args, **s_kwargs))
                elif func.__name__ == 'arange':
                    # tensor.arange takes the dtype of its input but
                    # numpy.arange does not. Since we are compiling the Theano
                    # graph, recast the numpy value to match the symbolic dtype
                    sval = theano.tensor.arange(*s_args, **s_kwargs)
                    rval = rval.astype(sval.dtype)
                elif func.__name__ in theano.tensor.basic._cast_mapping.keys():
                    # handle cast functions
                    rval = func(*args, **kwargs)
                    sval = theano.tensor.cast(*s_args, dtype=func.__name__)
                    self.watcher.shadow(rval, sval)
                elif func.__name__ in ['bool', 'bool_', 'bool8']:
                    # Theano has no bool type, cast to int8 instead
                    sval = theano.tensor.cast(*s_args, dtype='int8')
                elif func.__name__ in ['ones', 'zeros']:
                    s_fn = getattr(theano.tensor, func.__name__)
                    sval = s_fn(*s_args, **s_kwargs).astype(str(rval.dtype))
                    self.watcher.shadow(rval, sval)
                elif func.__name__ == 'identity':
                    # theano has no identity function, only 'eye'
                    dtype = s_kwargs.get('dtype', None)
                    if not dtype and len(s_args) > 1:
                        dtype = s_args[1]
                    sval = theano.tensor.eye(s_args[0], dtype=dtype)
                    self.watcher.shadow(rval, sval)
                else:
                    try:
                        theano_fn = getattr(theano.tensor, func.__name__)
                    except:
                        raise NotImplementedError(func)

                    # XXX should we do this? since it is not obvious that
                    # reductions don't take symbolic args, this could lead to
                    # users compiling functions that are supposed to have axis
                    # arguments but silently ignore them. Leaving this
                    # functionality out for now -- Users must call Constant()
                    # explicitly.

                    # many Theano reductions do not support symbolic axes
                    # by checking for it here we don't have to wrap the
                    # argument in a Constant()
                    # argspec = orderedargspec(theano_fn, *s_args, **s_kwargs)
                    # if (istensor(argspec.get('axis', None)) and
                        # func.__name__ not in ['concatenate']):
                        # if 'axis' in s_kwargs:
                            # s_kwargs['axis'] = kwargs['axis']
                        # else:
                            # r_axis = args[argspec.args.index('axis')]
                            # s_args[argspec.args.index('axis')] = r_axis
                    self.watcher.shadow(rval, theano_fn(*s_args, **s_kwargs))
            else:
                # no argument was shadowed (e.g. zeros())
                self.add_shadow(rval)

        # ================ Array methods

        elif isinstance(getattr(func, '__self__', None),
                        (np.ndarray, np.number)):
            assert id(func.__self__) in self.watcher
            s_self = self.watcher.svars[id(func.__self__)]

            if 0:
                pass
            elif func.__name__ == 'copy':
                assert not args
                assert not kwargs
                rval = func()
                self.watcher.shadow(rval, s_self.copy())
            elif func.__name__ == 'reshape':
                rval = func(*args, **kwargs)
                # Theano requires shape to be a tuple
                if not isinstance(s_args[0], (list, tuple)):
                    s_args = (s_args,)
                self.watcher.shadow(rval, s_self.reshape(*s_args, **s_kwargs))
            elif func.__name__ == 'swapaxes':
                rval = func(*args, **kwargs)
                axis1, axis2 = args
                s_dims = range(s_self.ndim)
                s_dims[axis1], s_dims[axis2] = s_dims[axis2], s_dims[axis1]
                self.watcher.shadow(rval, s_self.dimshuffle(*s_dims))
            elif func.__name__ == 'astype':
                rval = func(*args, **kwargs)
                if 'dtype' in kwargs:
                    dtype = kwargs['dtype']
                else:
                    dtype = args[0]
                if not isinstance(dtype, str):
                    # catch numpy dtype objects like np.float32
                    try:
                        dtype = dtype.__name__
                    except:
                        raise NotImplementedError
                if dtype == 'bool':
                    dtype == 'int8'
                self.watcher.shadow(rval, s_self.astype(dtype))
            elif func.__name__ == 'sort':
                # sort is an inplace method
                rval = func()  # returns None
                # shadow the original array; it has been updated inplace
                self.watcher.shadow(func.__self__, s_self.sort())
            else:
                try:
                    theano_fn = getattr(s_self, func.__name__)
                except:
                    raise NotImplementedError(func)
                rval = func(*args, **kwargs)
                self.watcher.shadow(rval, theano_fn(*s_args, **s_kwargs))

        # ================ built-ins

        elif 'built-in' in str(func):
            if len(args) == len(kwargs) == 0:
                rval = func()
            # -- built-in ndarray methods should be caught above, not here.
            elif func.__name__ in ('setdefault',):
                rval = func(*args, **kwargs)
            elif func.__name__ in ('enumerate', 'range', 'xrange', 'zip'):
                rval = func(*args, **kwargs)
            elif 'method rand of mtrand.RandomState' in str(func):
                # build Theano random uniform numbers
                rval = func(*args, **kwargs)
                self.watcher.shadow(
                    rval,
                    global_randomstreams.uniform(
                        low=0,
                        high=1,
                        size=tuple(args),
                        dtype=str(np.asarray(rval).dtype)))
            elif ('method random of mtrand.RandomState' in str(func)
                  or 'method random_sample of mtrand.RandomState'
                  in str(func)):
                # build Theano random uniform numbers
                rval = func(*args, **kwargs)
                self.watcher.shadow(
                    rval,
                    global_randomstreams.uniform(
                        low=0,
                        high=1,
                        size=autodiff.utils.as_seq(args[0], tuple),
                        dtype=str(np.asarray(rval).dtype)))
            elif 'method uniform of mtrand.RandomState' in str(func):
                # build Theano random normal numbers
                rval = func(*args, **kwargs)
                self.watcher.shadow(
                    rval,
                    global_randomstreams.uniform(
                        *args,
                        dtype=str(np.asarray(rval).dtype),
                        **kwargs))
            else:
                raise NotImplementedError(func)

        # ================ Types

        elif type(func) == type:
            rval = func(*args, **kwargs)

        # ================ AutoDiff Functions

        elif func is autodiff.functions.constant:
            # make sure the rval will have a vaild id, then add it to the
            # Context's constants set (so it can be ignored)
            rval = func(*args, **kwargs)
            if isinstance(rval, int):
                rval = np.int_(rval)
            elif isinstance(rval, float):
                rval = np.float_(rval)
            elif isinstance(rval, bool):
                rval = np.bool_(rval)
            else:
                rval = np.asarray(rval)
            self.watcher.constants.add(id(rval))

        elif func is autodiff.functions.tag:
            # make sure the rval is shadowed, then add a new svar with the
            # appropriate tag
            rval = func(*args, **kwargs)
            tag = kwargs.pop('tag', args[1])
            sval = self.ensure_shadow(rval)
            self.watcher.svars[tag] = sval

        # ================ Everything Else

        else:
            logger.debug('stepping into %s' % str(func))
            vm = FrameVM(self.watcher, func)
            rval = vm.call(tuple(args), kwargs)
        self.push(rval)

    def op_CALL_FUNCTION_VAR(self, i, op, arg):
        call_vargs = self.pop()
        return self.op_CALL_FUNCTION(i, op, arg, call_vargs=call_vargs)

    def op_CALL_FUNCTION_VAR_KW(self, i, op, arg):
        call_vargs, call_kwargs = self.popN(2)
        rval = self.op_CALL_FUNCTION(i,
                                     op,
                                     arg,
                                     call_vargs=call_vargs,
                                     call_kwargs=call_kwargs)
        return rval

    def op_COMPARE_OP(self, i, op, arg):
        opname = opcode.cmp_op[arg]
        right = self.pop()
        left = self.pop()
        if 0:
            pass
        elif opname == '==':
            self.push(left == right)
        elif opname == '!=':
            self.push(left != right)
        elif opname == '>':
            self.push(left > right)
        elif opname == '<':
            self.push(left < right)
        elif opname == '>=':
            self.push(left >= right)
        elif opname == '<=':
            self.push(left <= right)
        elif opname == 'is':
            self.push(left is right)
        elif opname == 'in':
            self.push(left in right)
        else:
            raise NotImplementedError('comparison: %s' % opname)

        if any(id(a) in self.watcher for a in [left, right]):
            sargs = [self.watcher.getvar(ai) for ai in [left, right]]
            tos = self.stack[-1]
            if 0:
                pass
            elif opname == '==':
                self.watcher.shadow(tos, theano.tensor.eq(*sargs))
            elif opname == '!=':
                self.watcher.shadow(tos, theano.tensor.neq(*sargs))
            elif opname == '<':
                self.watcher.shadow(tos, theano.tensor.lt(*sargs))
            elif opname == '>':
                self.watcher.shadow(tos, theano.tensor.gt(*sargs))
            elif opname == '<=':
                self.watcher.shadow(tos, theano.tensor.le(*sargs))
            elif opname == '>=':
                self.watcher.shadow(tos, theano.tensor.ge(*sargs))
            elif opname == 'is':
                pass
            else:
                raise NotImplementedError('Comparison on watched args',
                                          opname)

    def op_DUP_TOP(self, i, op, arg):
        self.stack.append(self.stack[-1])

    def op_DUP_TOPX(self, i, op, arg):
        assert arg > 0
        self.stack.extend(self.stack[-arg:])

    def op_FOR_ITER(self, i, op, arg):
        # either push tos.next()
        # or pop tos and send (arg)
        tos = self.stack[-1]
        try:
            next = tos.next()
            # print 'next', next
            self.push(next)
        except StopIteration:
            self.pop()
            return ('rel', arg)

    def op_INPLACE_ADD(self, i, op, arg):
        tos = self.pop()
        tos1 = self.pop()

        r = tos1
        r += tos
        self.push(r)
        if (id(tos) in self.watcher or id(tos1) in self.watcher):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s_tos + s_tos1).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s_tos + s_tos1)

    def op_INPLACE_DIVIDE(self, i, op, arg):
        tos = self.pop()
        tos1 = self.pop()

        r = tos1
        r /= tos
        self.push(r)
        if (id(tos) in self.watcher or id(tos1) in self.watcher):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s_tos / s_tos1).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s_tos / s_tos1)

    def op_INPLACE_MULTIPLY(self, i, op, arg):
        tos = self.pop()
        tos1 = self.pop()

        r = tos1
        r *= tos
        self.push(r)
        if (id(tos) in self.watcher or id(tos1) in self.watcher):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s_tos * s_tos1).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s_tos * s_tos1)

    def op_INPLACE_SUBTRACT(self, i, op, arg):
        tos1, tos = self.popN(2)

        r = tos1
        r -= tos
        self.push(r)
        if (id(tos) in self.watcher or id(tos1) in self.watcher):
            s_tos = self.ensure_shadow(tos)
            s_tos1 = self.ensure_shadow(tos1)
            if isinstance(r, np.ndarray):
                self.watcher.shadow(r, (s_tos - s_tos1).astype(str(r.dtype)))
            else:
                self.watcher.shadow(r, s_tos - s_tos1)

    def op_JUMP_ABSOLUTE(self, i, op, arg):
        # print 'sending', arg
        return ('abs', arg)

    def op_JUMP_FORWARD(self, i, op, arg):
        return ('rel', arg)

    def op_JUMP_IF_TRUE(self, i, op, arg):
        tos = self.stack[-1]
        if tos:
            return ('rel', arg)

    def op_GET_ITER(self, i, op, arg):
        # replace tos -> iter(tos)
        tos = self.stack[-1]
        if id(tos) in self.watcher:
            raise NotImplementedError('iterator of watched value')
        self.stack[-1] = iter(tos)

    def op_LOAD_GLOBAL(self, i, op, arg):
        # print 'LOAD_GLOBAL', self.names[arg]
        tos = self._myglobals[self.func.func_code.co_names[arg]]
        if type(tos) is int and -5 <= tos <= 256:
            tos = np.int_(tos)
        self.push(tos)
        if id(tos) not in self.watcher:
            self.add_shadow(self.stack[-1])

    def op_LOAD_ATTR(self, i, op, arg):
        # print 'LOAD_ATTR', self.names[arg]
        attr = self.func.func_code.co_names[arg]
        #
        # we would like to do
        #    self.stack[-1] = getattr(TOS, attr)
        #
        # *EXCEPT* if attr is a property, then it actually represents a
        # function call
        tos = self.pop()

        if isinstance(tos, np.ndarray):
            if id(tos) not in self.watcher:
                raise NotImplementedError(
                    'how did this var get here?', (id(tos), tos))

        if id(tos) in self.watcher:
            s_tos = self.watcher.svars[id(tos)]

            if attr == 'shape':
                rval = tos.shape
                # note this old comment... what does it mean?
                # XXX: NOT TRACKING SHAPE CHANGES BECAUSE
                #      BAD INTERACTION WITH fbncc.__theano_op__
                self.watcher.shadow(rval, s_tos.shape)
            elif attr == 'T':
                rval = tos.T
                self.watcher.shadow(rval, s_tos.T)
            elif attr == 'imag':
                rval = tos.imag
                self.watcher.shadow(rval, s_tos.imag)
            else:
                try:
                    rval = getattr(tos, attr)
                except:
                    raise NotImplementedError('ndarray attribute %s' % attr)
            self.push(rval)
        else:
            logger.debug('attribute access %s' % attr)
            rval = getattr(tos, attr)
            self.push(rval)
            # if (isinstance(rval, np.ndarray)
                # and id(rval) not in self.watcher):
                # self.add_shadow(rval)
            if id(rval) not in self.watcher:
                self.add_shadow(rval)

    def op_LOAD_CONST(self, i, op, arg):
        tos = self.func.func_code.co_consts[arg]
        if type(tos) is int and -5 <= tos <= 256:
            tos = np.int_(tos)
        self.push(tos)
        # if isinstance(tos, float):
            # if id(tos) not in self.watcher:
                # var = theano.tensor.as_tensor_variable(tos)
                # self.watcher.svars[id(tos)] = var
        if (isinstance(tos, np.ndarray) and id(tos) not in self.watcher):
            raise NotImplementedError()

    def op_LOAD_CLOSURE(self, i, op, arg):
        co_cellvars = self.func.func_code.co_cellvars
        co_freevars = self.func.func_code.co_freevars
        co_varnames = self.func.func_code.co_varnames
        if arg < len(co_cellvars):
            name = co_cellvars[arg]
        else:
            name = co_freevars[arg - len(co_cellvars)]
        thing = self._locals[co_varnames.index(name)]
        cell = cellmake(thing)
        self.push(cell)

    def op_LOAD_DEREF(self, i, op, arg):
        # -- this is called to access a variable that appears in multiple
        #    scopes.

        # -- vars *referenced* by nested scopes
        co_cellvars = self.func.func_code.co_cellvars

        # -- vars read from enclosing scopes
        co_freevars = self.func.func_code.co_freevars

        # -- all varnames
        co_varnames = self.func.func_code.co_varnames

        if arg < len(co_cellvars):
            # -- normal case
            name = co_cellvars[arg]
            # -- XXX: Is this really the right thing to do??
            thing = self._locals[co_varnames.index(name)]
        else:
            name = co_freevars[arg - len(co_cellvars)]
            closure = self.func.func_closure
            assert len(co_freevars) == len(closure)
            # print 'LOAD_DEREF (%s:%s)' % (self.func, name)
            cell = closure[arg - len(co_cellvars)]
            thing = cellget(cell)
        self.push(thing)
        # if (isinstance(thing, np.ndarray) and id(thing) not in self.watcher):
            # self.add_shadow(thing)
        if id(thing) not in self.watcher:
            self.add_shadow(thing)

    def op_LOAD_FAST(self, i, op, arg):
        tos = self._locals[arg]

        try:
            self.push(tos)
        except LoadUnassigned:
            raise LoadUnassigned(self.func.func_code.co_varnames[arg])
        if not isinstance(tos, (int, float)):
            if id(tos) not in self.watcher:
                self.add_shadow(tos)

    def op_MAKE_CLOSURE(self, i, op, arg):
        return self.op_MAKE_FUNCTION(i, op, arg, w_closure=True)

    def op_MAKE_FUNCTION(self, i, op, arg, w_closure=False):
        func_code = self.pop()
        if w_closure:
            cells = self.pop()
        if arg:
            argdefs = tuple(self.stack[-arg:])
            self.stack[-arg:] = []
        else:
            argdefs = ()
        if w_closure:
            fn = types.FunctionType(func_code,
                                    self.func.func_globals,
                                    argdefs=argdefs,
                                    closure=cells,)
        else:
            fn = types.FunctionType(func_code,
                                    self.func.func_globals,
                                    argdefs=argdefs)

        self.push(fn)

    def op_POP_BLOCK(self, i, op, arg):
        logger.debug('POP_BLOCK, what to do?')
        pass

    def op_POP_JUMP_IF_FALSE(self, i, op, arg):
        #tos = self.stack[-1]
        tos = self.pop()
        if not tos:
            return ('abs', arg)

    def op_POP_JUMP_IF_TRUE(self, i, op, arg):
        #tos = self.stack[-1]
        tos = self.pop()
        if tos:
            return ('abs', arg)

    def op_POP_TOP(self, i, op, arg):
        self.pop()

    def op_PRINT_ITEM(self, i, op, arg):
        thing = self.pop()
        if str(thing) == 'PRINT_OPS:True':
            self.print_ops = True
        if str(thing) == 'PRINT_STACK:True':
            self.print_stack = True
        print thing,

    def op_PRINT_NEWLINE(self, i, op, arg):
        print ''

    def op_SETUP_LOOP(self, i, op, arg):
        logger.debug('SETUP_LOOP, what to do?')
        pass

    def op_SLICE_PLUS_0(self, i, op, arg):
        #Implements TOS = TOS[:].
        TOS = self.pop()
        new_tos = TOS[:]
        self.push(new_tos)

        if id(TOS) in self.watcher:
            s = self.watcher.getvar(TOS)
            s_rval = s[:]
            self.watcher.shadow(new_tos, s_rval)

    def op_SLICE_PLUS_1(self, i, op, arg):
        # TOS = TOS1[TOS:]
        TOS1, TOS = self.popN(2)
        new_tos = TOS1[TOS:]
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1]):
            s = self.watcher.getvar(TOS)
            s1 = self.watcher.getvar(TOS1)
            s_rval = s1[s:]
            self.watcher.shadow(new_tos, s_rval)

    def op_SLICE_PLUS_2(self, i, op, arg):
        # TOS = TOS1[:TOS]
        TOS1, TOS = self.popN(2)
        new_tos = TOS1[:TOS]
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1]):
            s = self.watcher.getvar(TOS)
            s1 = self.watcher.getvar(TOS1)
            s_rval = s1[:s]
            self.watcher.shadow(new_tos, s_rval)

    def op_SLICE_PLUS_3(self, i, op, arg):
        # Implements TOS = TOS2[TOS1:TOS]
        TOS2, TOS1, TOS = self.popN(3)
        new_tos = TOS2[TOS1:TOS]
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1, TOS2]):
            s = self.watcher.getvar(TOS)
            s1 = self.watcher.getvar(TOS1)
            s2 = self.watcher.getvar(TOS2)
            s_rval = s2[s1:s]
            self.watcher.shadow(new_tos, s_rval)

    def op_STORE_ATTR(self, i, op, arg):
        # implements TOS.name = TOS1
        TOS1, TOS = self.popN(2)
        if TOS in self.watcher:
            raise NotImplementedError()
        name = self.func.func_code.co_names[arg]
        setattr(TOS, name, TOS1)

    def op_STORE_SLICE_PLUS_0(self, i, op, arg):
        #Implements TOS[:] = TOS1
        TOS1, TOS = self.popN(2)
        new_tos = TOS
        new_tos[:] = TOS1
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1]):
            s_tos = self.watcher.getvar(TOS)
            s_tos1 = self.watcher.getvar(TOS1)
            s_rval = theano.tensor.set_subtensor(s_tos[:], s_tos1)
            self.watcher.shadow(new_tos, s_rval)

    def op_STORE_SLICE_PLUS_1(self, i, op, arg):
        TOS2, TOS1, TOS = self.popN(3)
        new_tos = TOS1
        new_tos[TOS:] = TOS2
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1, TOS2]):
            s_tos = self.watcher.getvar(TOS)
            s_tos1 = self.watcher.getvar(TOS1)
            s_tos2 = self.watcher.getvar(TOS2)
            s_rval = theano.tensor.set_subtensor(s_tos1[s_tos:], s_tos2)
            self.watcher.shadow(new_tos, s_rval)

    def op_STORE_SLICE_PLUS_2(self, i, op, arg):
        # TOS1[:TOS] = TOS2
        TOS2, TOS1, TOS = self.popN(3)
        new_tos = TOS1
        new_tos[:TOS] = TOS2
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1, TOS2]):
            s_tos = self.watcher.getvar(TOS)
            s_tos1 = self.watcher.getvar(TOS1)
            s_tos2 = self.watcher.getvar(TOS2)
            s_rval = theano.tensor.set_subtensor(s_tos1[:s_tos], s_tos2)
            self.watcher.shadow(new_tos, s_rval)

    def op_STORE_SLICE_PLUS_3(self, i, op, arg):
        # Implements TOS2[TOS1:TOS] = TOS3
        TOS3, TOS2, TOS1, TOS = self.popN(4)
        new_tos = TOS2
        new_tos[TOS1:TOS] = TOS3
        self.push(new_tos)

        if any(id(t) in self.watcher for t in [TOS, TOS1, TOS2, TOS3]):
            s_tos = self.watcher.getvar(TOS)
            s_tos1 = self.watcher.getvar(TOS1)
            s_tos2 = self.watcher.getvar(TOS2)
            s_tos3 = self.watcher.getvar(TOS3)
            s_rval = theano.tensor.set_subtensor(s_tos2[s_tos1:s_tos], s_tos3)
            self.watcher.shadow(new_tos, s_rval)

    def op_STORE_FAST(self, i, op, arg):
        self._locals[arg] = self.pop()

    def op_STORE_MAP(self, i, op, arg):
        key = self.pop()
        val = self.pop()
        dct = self.stack[-1]
        dct[key] = val

    def op_STORE_SUBSCR(self, i, op, arg):
        # Implements TOS1[TOS] = TOS2.
        tos = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()

        tos1[tos] = tos2

        # tos can't be real-valued so there's no gradient through it
        if id(tos1) in self.watcher or id(tos2) in self.watcher:
            s_tos1 = self.ensure_shadow(tos1)
            s_tos2 = self.ensure_shadow(tos2)

            new_s_tos1 = theano.tensor.set_subtensor(s_tos1[tos], s_tos2)
            self.watcher.svars[id(tos1)] = new_s_tos1

    def op_RAISE_VARARGS(self, i, op, arg):
        print >> sys.stderr, "Exception in autodiff.Context:"
        if 1 <= arg:
            exc = self.pop()
        else:
            exc = None
        if 2 <= arg:
            param = self.pop()
        else:
            param = None
        if 3 <= arg:
            tb = self.pop()
            traceback.print_tb(tb, file=sys.stderr)
        else:
            print >> sys.stderr, "No traceback info available"
        if param is not None:
            raise param
        elif exc is not None:
            raise exc()
        else:
            raise Exception('Completely mysterious exception')

    def op_RETURN_VALUE(self, i, op, arg):
        self.rval = self.pop()
        if id(self.rval) not in self.watcher:
            self.add_shadow(self.rval)

    def op_ROT_TWO(self, i, op, arg):
        a = self.stack[-1]
        b = self.stack[-2]
        self.stack[-1] = b
        self.stack[-2] = a

    def op_ROT_THREE(self, i, op, arg):
        a = self.stack[-1]
        b = self.stack[-2]
        c = self.stack[-3]
        self.stack[-1] = b
        self.stack[-2] = c
        self.stack[-3] = a

    def op_ROT_FOUR(self, i, op, arg):
        a = self.stack[-1]
        b = self.stack[-2]
        c = self.stack[-3]
        d = self.stack[-4]
        self.stack[-1] = b
        self.stack[-2] = c
        self.stack[-3] = d
        self.stack[-4] = a

    def op_UNARY_NEGATIVE(self, i, op, arg):
        arg1 = self.pop()
        assert not hasattr(arg1, 'type')
        r = -arg1
        self.push(r)
        if id(arg1) in self.watcher:
            s1 = self.ensure_shadow(arg1)
            self.watcher.shadow(r,  -s1)

    def op_UNPACK_SEQUENCE(self, i, op, arg):
        tos = self.pop()
        self.stack.extend(tos[::-1])


class Context(object):
    def __init__(self, device=None, borrowable=(), force_floatX=False):
        """
        borrowable : tuple of objects
            If an object in this tuple is encountered while tracing the
            function, then its symbolic representation will alias that object's
            memory location. This means that *inplace* operations on the Python
            (likely NumPy) object will affect the symbolic function.

        force_floatX : bool
            If True, floats and float NumPy ndarrays will be cast to the dtype
            specified at theano.config.floatX when forming symbolic shared
            variables, if they do not have it already. Objects in `borrowable`
            are never cast.

        """
        self.svars = {}
        self.nogc = []  # ids that must not be reused
        # XXX: rethink to avoid actually holding on to all these intermediates.
        self.device = device
        self.borrowable_ids = [id(b) for b in borrowable]
        self.force_floatX = force_floatX
        self.constants = set()

    def __iter__(self):
        return self.svars.__iter__()

    def shadow(self, rval, sval, force=True):
        assert hasattr(sval, 'type')  # assert sval is Theano variable
        if force:
            self.svars[id(rval)] = sval
        else:
            self.svars.setdefault(id(rval), sval)

        # -- shadow vars have to match dtype and ndim
        if isinstance(rval, np.ndarray):
            if str(rval.dtype) == 'bool':
                assert sval.dtype == 'int8', (rval.dtype, sval.dtype)
            elif not self.force_floatX:
                assert str(rval.dtype) == sval.dtype, (rval, sval)
            assert rval.ndim == sval.ndim, (rval, sval)

        # -- assert postcondition
        assert sval is self.getvar(rval)
        self.nogc.append(rval)

    def call(self, fn, args=(), kwargs={}):
        vm = FrameVM(self, fn)
        return vm.call(args, kwargs)

    def shared(self, obj, name=None, borrow=None):
        if borrow is None:
            borrow = (id(obj) in self.borrowable_ids)
        if self.force_floatX and not borrow:
            if (isinstance(obj, np.ndarray)
               and 'float' in str(obj.dtype)
               and str(obj.dtype) != theano.config.floatX):
                obj = obj.astype(theano.config.floatX)

        # not all objects have shared constructors with a borrow keyword
        # for example theano.shared(np.float32(1)) works but
        # theano.shared(np.float32(1), borrow=[False|True]) fails
        if self.device == 'cpu':
            try:
                return theano.tensor._shared(obj, borrow=borrow)
            except:
                return theano.tensor._shared(obj)
        else:
            try:
                return theano.shared(obj, borrow=borrow)
            except:
                return theano.shared(obj)

    def getvar(self, var):
        return self.svars.get(id(var), var)

    def reset(self):
        self.constants.clear()
