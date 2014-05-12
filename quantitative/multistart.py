import multiprocessing
import numpy as np
import scipy.optimize as opt
from functools import partial
from parallel import BetterPool
from scipy.linalg import LinAlgError


def minimize_success(res):
    # Condition to check if opt.minimize result is successful
    if res.success:
        return res.fun, res
    return res.message


class ParMap(object):
    # http://stackoverflow.com/a/16071616
    def __init__(self, n_procs=multiprocessing.cpu_count()):
        self._processes = n_procs
        self.q_in = multiprocessing.Queue(1)  # why is max size 1? immediately consumed?
        self.q_out = multiprocessing.Queue()

    def _spawn(self, func):
        def fun():
            while True:
                i, x = self.q_in.get()
                if i is None:
                    break
                self.q_out.put((i, func(**x)))
        return fun

    def map(self, f, iterable):
        q_in = self.q_in
        q_out = self.q_out
        nprocs = self._processes

        proc = [multiprocessing.Process(target=self._spawn(f)) for _ in range(nprocs)]
        for p in proc:
            p.daemon = True
            p.start()

        sent = [q_in.put((i, x)) for i, x in enumerate(iterable)]
        [q_in.put((None, None)) for _ in range(nprocs)]
        res = [q_out.get() for _ in range(len(sent))]

        [p.join() for p in proc]
        return [x for i, x in sorted(res)]


class MultiStart(object):
    """
    Uniformly random sample n_start_points in the x0_range box.  Additional args and kwargs will partial on func.
    Calls func with keyword x0=each_start_point using parallel_pool threads.
    Calls with other args and keywords also partial func (on top of the existing partial) and can be used to
    override x0 or previous arguments.
    """
    def __init__(self, n_start_points, x0_range, func=opt.minimize, verbose=True, *args, **kwargs):
        self.n_start_points = n_start_points
        self.x0_range = x0_range
        self.func = func
        self.partial_func = partial(func, *args, **kwargs)
        self.verbose = verbose

    def _print(self, s):
        if self.verbose:
            print s

    def _solve(self, partial_func, label='', i_start_point='', success=minimize_success, **fkwargs):
        """
        success is a function that parses the solver output into a tuple of (the cost, anything else to be kept in self.candidates).
            It should return an error string on failure.
        """
        try:
            res = partial_func(**fkwargs)
        except (LinAlgError, ValueError) as e:
            self._print('%s at Start %s: Exception: %s' % (label, i_start_point, e))
            return None
        s = success(res)
        if isinstance(s, basestring):
            self._print('%s at Start %s: Failure: %s' % (label, i_start_point, s))
            return None
        else:
            self._print('%s at Start %s: Success!' % (label, i_start_point))
            #self._print(res)
            return s

    def __call__(self, partial_func, *fargs, **fkwargs):
        # Workaround for multiprocessing compatibility due to unpicklable instancemethods
        return self._solve(partial_func, *fargs, **fkwargs)

    def solve(self, parallel_pool=None, start_points=None, *fargs, **fkwargs):
        """
        start_points is None generates a list of random x0 points from self.x0_range.  Otherwise it should be a list of kwargs to give the solver function.
        """
        picklable_partial_func = partial(self, self.partial_func, *fargs, **fkwargs)
        # TODO is it confusing to have double partialing in init and solve?  does prove useful e.g. with opt.leastsq needing func argument
        # TODO don't hardcode 'x0'
        # TODO choose sample points that are far from each other?  something more like GlobalSearch
        # TODO try BetterPool instead of ParMap, I think there's a reason why I was forced to use ParMap
        if start_points is None:
            start_points = [{'x0': np.random.uniform(*zip(*self.x0_range))} for _ in xrange(self.n_start_points)]
        for i, start_point in enumerate(start_points):
            start_point['i_start_point'] = i
        self.start_points = start_points
        if parallel_pool is None:
            self._print('Running in serial:')
            candidates = [picklable_partial_func(**el) for el in start_points]
        else:
            if isinstance(parallel_pool, int):
                #parallel_pool = BetterPool(parallel_pool)
                if parallel_pool < 1:
                    parallel_pool = ParMap()
                else:
                    parallel_pool = ParMap(parallel_pool)
            self._print('Running in parallel with %d workers:' % parallel_pool._processes)
            candidates = parallel_pool.map(picklable_partial_func, start_points)
        self.candidates = candidates = sorted(filter(lambda e: e is not None, candidates), key=lambda e: e[0])
        self._print('== %s Success Rate: %.2f' % (fkwargs.get('label', ''), 100.*len(candidates)/float(len(start_points))))
        try:
            return candidates[0][1]
        except IndexError:
            return None


def global_optimize(n_start_points, x0_range, *args, **kwargs):
    candidates = []
    for i in range(n_start_points):
        x0 = np.random.uniform(*zip(*x0_range))
        try:
            res = opt.minimize(x0=x0, *args, **kwargs)
        except (LinAlgError, ValueError):
            continue
        if res.success:
            print 'Success!'
            candidates.append((res.fun, res))
        else:
            print res
    candidates = sorted(candidates, key=lambda e: e[0])
    try:
        return candidates[0][1], candidates
    except IndexError:
        return None


if __name__ == '__main__':
    def cost_func(x, a=0.0):
        return (x-a)**2

    M = MultiStart(10, [(-10., 10.)], method='SLSQP')
    soln = M.solve(4, fun=cost_func)
    print soln
    print M.candidates
