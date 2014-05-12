"""
Solve for the optimal b-values to measure D in the exponential diffusion model.
"""

import shelve
import numpy as np
import scipy.linalg as spla
from functools import partial
from collections import OrderedDict
from numpy import newaxis
from quantitative import HigherAD, calc_crlb, MultiOptimizationHelper, MultiStart
from quantitative.costtools import sort_jacobians, remove_small_nex


def diffusion(b, D, m0):
    return m0*np.exp(-b*D)


Diffusion = HigherAD(diffusion)


def diffusion_cost_function(
        combine=np.sum,
        cov=True,
        L=np.atleast_1d([1.0]),
        diffusion=None,
        D=np.array([1.0]),
        m0=np.array([1.0]),
        wrt_in=('D', 'm0'),
        regularization=1e-6,
):
    # Use default values if not specified
    if diffusion is not None:
        b = diffusion.get('b', np.atleast_1d([np.pi/4]))
        nex = diffusion.get('nex', np.atleast_1d([1.0]))
        # Remove values with small NEX, these tend to cause inversion errors as they are essentially a row of 0s
        nex, (b,) = remove_small_nex(nex, (b,))

    # Estimating T1 and M0, calculate Jacobian for each sample tissue
    Js = sort_jacobians(Diffusion.jacobian(
        b,
        D[:, newaxis],
        m0[:, newaxis],
        wrt_in=wrt_in,
    ))
    noise_variance = 1.0/nex
    try:
        crlb = np.array([calc_crlb(J, noise_variance, regularization=regularization) for J in Js])
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
        return combine((L*crlb/true_values).sum(axis=0))
    else:
        return combine((-L*true_values/crlb).sum(axis=0))


def construct_diffusion_cost_function(cost_func, n_images, params, start_range, constraints):
    """
    Arguments:
        cost_func --
        params -- the parameters that are free to vary for each sequence
        start_range -- 
        constraints -- 
    """
    Cost_Function = MultiOptimizationHelper(cost_func, n_images, params=params, start_range=start_range)
    indices = Cost_Function.parameter_indices
    # TODO name the constants
    # nex must be > 0.01 to avoid singular F
    Cost_Function.add_affine_constraint(('diffusion', 'nex'), 'ineq', 1.0, -0.01)
    Cost_Function.add_affine_constraint(('diffusion', 'b'), 'ineq')
    for prev, next in zip(indices['diffusion']['b'][:-1], indices['diffusion']['b'][1:]):
        Cost_Function.constraints.append({
            'type': 'ineq',
            'fun': lambda x, prev=prev, next=next: x[next] - x[prev]
        })
    # Fixed time constraint
    # TODO: modify this to account for difference in min_tr bet seq.
    Cost_Function.constraints.append({
        'type': 'eq',
        'fun': lambda x, nex_idx=indices['diffusion']['nex']: sum(x[nex_idx]) - 1.
    })
    return Cost_Function


def optimize_diffusion_protocol(params, start_range, D, m0, L, wrt_in, protocol_frameworks, cost_types, **kwargs):
    """
    Optimize the diffusion protocol of collecting images at different b-values to estimate a m0 and D.
    
    Arguments:
        params -- the parameters that are free to vary for each sequence, sequences may be removed to fit with the protocol framework
        start_range -- 
        L -- 
        wrt_in -- 
        wrt_out_ssfp -- 
        protocol_frameworks -- a list of dictionaries for example [{'spgr': 2, 'ssfp': 2}], note that params and start_range will be properly filtered if the sequence is missing from a framework
        cost_types -- 
    """
    store_solutions = OrderedDict()
    for n_images in protocol_frameworks:
        print '\n\n========== Solving %s ==========' % (n_images, )
        p = {k: v for k, v in params.items() if k in n_images}
        sr = {k: v for k, v in start_range.items() if k in n_images}
        Cost_Function = construct_diffusion_cost_function(n_images=n_images, params=p, start_range=sr, **kwargs)   
        print 'Constraints:', len(Cost_Function.constraints), Cost_Function.constraints
        partial_cost_func = partial(
            Cost_Function,
            D=D,
            m0=m0,
            L=L,
            wrt_in=wrt_in,
        )
        print 'Compile Theano for floats'
        try:
            partial_cost_func(np.random.random(len(Cost_Function.start_range)))
        except spla.LinAlgError:
            pass

        M = MultiStart(
            100,
            Cost_Function.start_range,
            constraints=Cost_Function.constraints,
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
                top_solution = Cost_Function._parameter_values(res.x)
                top_solution['diffusion']['b'] = top_solution['diffusion']['b']
                top_solution['diffusion']['nex'] = top_solution['diffusion']['nex']
                print '  Top Solution: %s %s\n' % (res.fun, top_solution)
            store_solutions['%s %s' % (n_images, cost_type_name)] = M.candidates
    return store_solutions


cost_types = OrderedDict([
    # ('sum 1/cov', {'combine': np.sum, 'cov': False}),
    ('sum cov', {'combine': np.sum, 'cov': True}),
    # {'combine': np.max, 'cov': True},
    # {'combine': np.max, 'cov': False},
])

D = np.array([1.0])
m0 = np.array([1.0])

problems = OrderedDict([
    ('Diffusion', {
        'cost_func': diffusion_cost_function,
        'L': np.array([1., 0.]),
        'wrt_in': ('D', 'm0'),
        'protocol_frameworks': [{'diffusion': el} for el in range(2, 5)],
        'params': {'diffusion': ['b', 'nex']},
        'start_range': {'diffusion': [(0.1, 10.), (0.1, 1.)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
])

if __name__ == '__main__':
    print 'D', D
    print 'm0', m0
    
    database = shelve.open('db_diffusion')
    for name, problem in problems.iteritems():
        print '\n\n_________________________________________'
        print name
        print '_________________________________________'
        print 'Lambda: %s' % zip(problem['wrt_in'], problem['L'])
        print 'Problem:'
        for el in problem.iteritems():
            print '\t%s: %s' % el
        database[name] = optimize_diffusion_protocol(D=D, m0=m0, **problem)
    database.close()
