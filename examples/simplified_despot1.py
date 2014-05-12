"""
Solve for the optimal flip angles and acquisition time fractions to measure T1 using SPGR images.
"""


import shelve
import numpy as np
import scipy.linalg as spla
from functools import partial
from collections import OrderedDict
from numpy import newaxis
from quantitative import HigherAD, calc_crlb, MultiOptimizationHelper, MultiStart
from quantitative.costtools import sort_jacobians, remove_small_nex
from despot import spgr


SPGR = HigherAD(spgr)
tr = 5.0


def despot1_cost_function(
        combine=np.sum,
        cov=True,
        L=np.atleast_1d([1.0]),
        spgr=None,
        t1=np.array([1.0]),
        m0=np.array([1.0]),
        wrt_in=('t1', 'm0'),
):
    # Use default values if not specified
    if spgr is not None:
        spgr_theta = spgr.get('theta', np.atleast_1d([np.pi/4]))
        spgr_nex = spgr.get('nex', np.atleast_1d([1.0]))
        spgr_tr = spgr.get('tr', np.atleast_1d(tr))
        # Remove values with small NEX, these tend to cause inversion errors as they are essentially a row of 0s
        # Can be omitted if regularization in calc_crlb is used instead.
        spgr_nex, (spgr_theta, spgr_tr) = remove_small_nex(spgr_nex, (spgr_theta, spgr_tr))

    # Estimating T1 and M0, calculate Jacobian for each sample tissue
    Js = sort_jacobians(SPGR.jacobian(
        spgr_theta,
        spgr_tr,
        t1[:, newaxis],
        m0[:, newaxis],
        wrt_in=wrt_in,
    ))
    noise_variance = 1.0/spgr_nex
    try:
        crlb = np.array([calc_crlb(J, noise_variance) for J in Js])
    except spla.LinAlgError:
        print 'CRLB Error'
        print Js.shape
        print Js[0].shape
        raise
    if combine is None:
        # Bypass to return all the lower bound variances
        return crlb

    true_values = np.broadcast_arrays(*[locals()[tissue_param] for tissue_param in wrt_in])
    true_values = np.vstack(true_values).T
    if cov:
        # Minimize coefficient of variation
        return combine((L*crlb/true_values).sum(axis=0))
    else:
        # Maximize T1NR = minimize the negative
        return combine((-L*true_values/crlb).sum(axis=0))


def construct_despot1_cost_function(cost_func, n_images, params, start_range, constraints):
    """
    Arguments:
        cost_func --
        params -- the parameters that are free to vary for each sequence
        start_range -- 
        constraints -- 
    """
    DESPOT1_Cost_Function = MultiOptimizationHelper(cost_func, n_images, params=params, start_range=start_range)
    indices = DESPOT1_Cost_Function.parameter_indices
    # TODO name the constants
    for seq in n_images:
        # nex must be > 0.01 to avoid singular F
        DESPOT1_Cost_Function.add_affine_constraint((seq, 'nex'), 'ineq', 1.0, -0.01)
    if 'spgr' in indices and 'theta' in indices['spgr']:
        DESPOT1_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', 1.0, -1./32767.)
        # theta must be < 80 for spgr
        DESPOT1_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', -1.0, np.pi*80./180.)
        # sort theta ascending
        for prev, next in zip(indices['spgr']['theta'][:-1], indices['spgr']['theta'][1:]):
            DESPOT1_Cost_Function.constraints.append({
                'type': 'ineq',
                'fun': lambda x, prev=prev, next=next: x[next] - x[prev]
            })   
    # TODO: modify this to account for differences in tr between seq.
    # Fixed time constraint
    DESPOT1_Cost_Function.constraints.append({
        'type': 'eq',
        'fun': lambda x, spgr_nex_idx=indices['spgr']['nex']: sum(x[spgr_nex_idx]) - 10.
    })
    return DESPOT1_Cost_Function


def optimize_despot1_protocol(params, start_range, t1, m0, L, wrt_in, protocol_frameworks, cost_types, **kwargs):
    """
    Optimize the DESPOT1 protocol of collecting SPGR images to estimate a subset of (t1, t2, m0, off_resonance_phase).
    
    Arguments:
        params -- the parameters that are free to vary for each sequence, sequences may be removed to fit with the protocol framework
        start_range -- a list of tuples containing the solver's initial point start range for each of the params.
        L -- the lambda weights in the cost function, in corresponding order with "wrt_in"
        wrt_in -- the parameters being estimated, the strings must match the actual signal equation function input names exactly
        protocol_frameworks -- a list of dictionaries for example [{'spgr': 2, 'ssfp': 2}], note that params and start_range will be properly filtered if the sequence is missing from a framework
        cost_types -- a dictionary containing a 'combine' function for the 
    """
    store_solutions = OrderedDict()
    for n_images in protocol_frameworks:
        print '\n\n========== Solving %s ==========' % (n_images, )
        p = {k: v for k, v in params.items() if k in n_images}
        sr = {k: v for k, v in start_range.items() if k in n_images}
        DESPOT1_Cost_Function = construct_despot1_cost_function(n_images=n_images, params=p, start_range=sr, **kwargs)   
        print 'Constraints:', len(DESPOT1_Cost_Function.constraints), DESPOT1_Cost_Function.constraints
        partial_cost_func = partial(
            DESPOT1_Cost_Function,
            t1=t1,
            m0=m0,
            L=L,
            wrt_in=wrt_in,
        )
        # Call this first with arbitrary input to cache the compiled function and avoid MultiStart compiling many times.
        print 'Compile Theano for floats'
        try:
            partial_cost_func(np.random.random(len(DESPOT1_Cost_Function.start_range)))
        except spla.LinAlgError:
            pass

        # Only SLSQP can handle equality and inequality constraints.
        M = MultiStart(
            100,
            DESPOT1_Cost_Function.start_range,
            constraints=DESPOT1_Cost_Function.constraints,
            method='SLSQP',
        )

        for i, (cost_type_name, cost_type) in enumerate(cost_types.iteritems()):
            print 'Cost Type', cost_type_name, cost_type
            res = M.solve(
                parallel_pool=0,
                fun=partial(partial_cost_func, **cost_type),
                label=str(n_images)
            )
            if res:
                print '  Top Solution: %s\n' % DESPOT1_Cost_Function._parameter_values(res.x)
            store_solutions['%s %s' % (n_images, cost_type_name)] = M.candidates
    return store_solutions


cost_types = OrderedDict([
    ('sum 1/cov', {'combine': np.sum, 'cov': False}),
    ('sum cov', {'combine': np.sum, 'cov': True}),
    # {'combine': np.max, 'cov': True},
    # {'combine': np.max, 'cov': False},
])

# 3T GM T1=1645, T2=85 (Stanisz 2005)
t1 = np.array([1645.])
t2 = np.array([85.])
m0 = np.array([1.0])
off_resonance_phase = np.array([0.])
# t1, t2, off_resonance_phase = flatten_space(t1, t2, off_resonance_phase)

problems = OrderedDict([
    ('DESPOT1', {
        'cost_func': despot1_cost_function,
        'L': np.array([1., 0.]),
        'wrt_in': ('t1', 'm0'),
        'protocol_frameworks': [{'spgr': el} for el in range(2, 5)],
        'params': {'spgr': ['theta', 'nex']},
        'start_range': {'spgr': [(np.pi/180., np.pi*70./180.), (0.1, 5.)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
])

if __name__ == '__main__':
    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    
    database = shelve.open('db_despot1')
    for name, problem in problems.iteritems():
        print '\n\n_________________________________________'
        print name
        print '_________________________________________'
        print 'Lambda: %s' % zip(problem['wrt_in'], problem['L'])
        print 'Problem:'
        for el in problem.iteritems():
            print '\t%s: %s' % el
        database[name] = optimize_despot1_protocol(t1=t1, m0=m0, **problem)
    database.close()
