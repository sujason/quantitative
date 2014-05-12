import multiprocessing
import os

def spawn(f):
    def fun(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i == None:
                break
            q_out.put((i,f(x)))
    return fun

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

class ParMap(object):
    def __init__(self, n_procs=multiprocessing.cpu_count()):
        self._processes = n_procs
        self.q_in = multiprocessing.Queue(1) # why is max size 1? immediately consumed?
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

if __name__ == '__main__':
    def f(i):
        os.system('sleep %s' % 3)
        return i*2
    #print parmap(f, range(8))
    print ParMap().map(f, [{'i': e} for e in range(10)])
