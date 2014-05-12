import scipy.optimize as opt
from functools import wraps
from itertools import chain
from collections import OrderedDict
from despot import despot1_optimal_angles, despot2_optimal_angles
from permute import *
from variable_transforms import *


def match_length(match_to, vec):
    # Cyclically replicates vec to match the length of match_to.
    return np.tile(vec, (match_to.size/vec.size + 1))[:match_to.size]


@wraps(despot2_cost_function)
def plain_despot2_cost_function(*args, **kwargs):
    # If no theta specified in SPGR, use the optimal pair of DESPOT1 angles for the mean T1.
    if 'spgr' in kwargs and 'theta' not in kwargs['spgr']:
        spgr = kwargs['spgr']
        spgr_tr = spgr.get('tr', np.atleast_1d(min_tr))
        spgr['theta'] = match_length(spgr['nex'], despot1_optimal_angles(spgr_tr, t1.mean()))
    # If no theta specified in SSFP, use the optimal pair of DESPOT2 angles for the mean T1 and T2.
    if 'ssfp' in kwargs and 'theta' not in kwargs['ssfp']:
        ssfp = kwargs['ssfp']
        ssfp_tr = ssfp.get('tr', np.atleast_1d(min_tr))
        ssfp['theta'] = match_length(ssfp['nex'], despot2_optimal_angles(ssfp_tr, t1.mean(), t2.mean()))
    # Fix phase_rf to pi if not specified.
    if 'ssfp' in kwargs and 'phase_rf' not in kwargs['ssfp']:
        kwargs['ssfp']['phase_rf'] = np.atleast_1d(np.pi)
    return despot2_cost_function(*args, **kwargs)


def construct_despot2_cost_function(cost_func, n_images, params, start_range, constraints, transforms=None):
    """
    Arguments:
        cost_func --
        params -- the parameters that are free to vary for each sequence
        start_range -- 
        constraints -- 
    """
    DESPOT2_Cost_Function = MultiOptimizationHelper(cost_func, n_images, params=params, start_range=start_range, transforms=transforms)
    indices = DESPOT2_Cost_Function.parameter_indices
    # TODO name the constants
    for seq in n_images:
        # nex must be > 0.01 to avoid singular F
        DESPOT2_Cost_Function.add_affine_constraint((seq, 'nex'), 'ineq', 1.0, -0.001)
    if 'spgr' in indices and 'theta' in indices['spgr']:
        DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', 1.0, -1./32767.)
        # theta must be < 80 for spgr
        DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', -1.0, np.pi*80./180.)
        # sort theta ascending
        for prev, next in zip(indices['spgr']['theta'][:-1], indices['spgr']['theta'][1:]):
            DESPOT2_Cost_Function.constraints.append({
                'type': 'ineq',
                'fun': lambda x, prev=prev, next=next: x[next] - x[prev]
            })
    if 'ssfp' in indices:
        if 'theta' in indices['ssfp']:
            # theta must be > 0
            #DESPOT2_Cost_Function.add_affine_constraint((seq, 'theta'), 'ineq')
            DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'theta'), 'ineq', 1.0, -1./32767.)
            # theta must be < 70 for ssfp due to SAR
            DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'theta'), 'ineq', -1.0, np.pi*70./180.)
            # sort theta ascending
            for prev, next in zip(indices['ssfp']['theta'][:-1], indices['ssfp']['theta'][1:]):
                DESPOT2_Cost_Function.constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, prev=prev, next=next: x[next] - x[prev]
                })
        if True: #'phase_rf' in indices['ssfp']:
            # -pi < phase_rf < pi
            DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq', 1.0, np.pi)
            #DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq') # lower bound of 0, symmetry makes pi redundant?
            DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq', -1.0, np.pi)

    #print 'Indices:', indices
    if 'SPGR equal nex' in constraints and 'spgr' in indices:
        # Enforce SPGR equal nex assumption as in Deoni 2003
        for prev, next in zip(indices['spgr']['nex'][:-1], indices['spgr']['nex'][1:]):
            DESPOT2_Cost_Function.constraints.append({
                'type': 'eq',
                'fun': lambda x, prev=prev, next=next: x[next] - x[prev]
            })
    if 'SSFP equal nex' in constraints and 'ssfp' in indices:
        # Enforce SSFP equal nex assumption as in Deoni 2003
        for prev, next in zip(indices['ssfp']['nex'][:-1], indices['ssfp']['nex'][1:]):
            DESPOT2_Cost_Function.constraints.append({
                'type': 'eq',
                'fun': lambda x, prev=prev, next=next: x[next] - x[prev]
            })
    # Fixed time constraint
    # TODO: modify this to account for difference in min_tr bet seq.
    if 'spgr' in n_images and 'ssfp' in n_images:
        DESPOT2_Cost_Function.constraints.append({
            'type': 'eq',
            'fun': lambda x, spgr_nex_idx=indices['spgr']['nex'], ssfp_nex_idx=indices['ssfp']['nex']: sum(x[spgr_nex_idx]) + sum(x[ssfp_nex_idx]) - 1.
        })
    elif 'ssfp' in n_images:
        DESPOT2_Cost_Function.constraints.append({
            'type': 'eq',
            'fun': lambda x, ssfp_nex_idx=indices['ssfp']['nex']: sum(x[ssfp_nex_idx]) - 1.
        })
    return DESPOT2_Cost_Function


def optimize_despot2_protocol(params, start_range, t1, t2, m0, off_resonance_phase, L, wrt_in, wrt_out_ssfp, protocol_frameworks, cost_types, **kwargs):
    """
    Optimize the DESPOT2 protocol of collecting SPGR and SSFP images to estimate a subset of (t1, t2, m0, off_resonance_phase).
    
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
        DESPOT2_Cost_Function = construct_despot2_cost_function(n_images=n_images, params=p, start_range=sr, **kwargs)   
        print 'Constraints:', len(DESPOT2_Cost_Function.constraints), DESPOT2_Cost_Function.constraints
        partial_cost_func = partial(
            DESPOT2_Cost_Function,
            t1=t1,
            t2=t2,
            m0=m0,
            off_resonance_phase=off_resonance_phase,
            L=L,
            wrt_in=wrt_in,
            wrt_out_ssfp=wrt_out_ssfp,
            regularization=0.,
        )
        print 'Compile Theano for floats'
        try:
            partial_cost_func(np.random.random(len(DESPOT2_Cost_Function.start_range)))
        except spla.LinAlgError:
            pass

        M = MultiStart(
            100,
            DESPOT2_Cost_Function.start_range,
            constraints=DESPOT2_Cost_Function.constraints,
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
                top_solution = DESPOT2_Cost_Function._parameter_values(res.x)
                print '  Top Solution: %s %s\n' % (res.fun, top_solution)
            store_solutions['%s %s' % (n_images, cost_type_name)] = M.candidates
    return store_solutions


# n_images = {
#     'spgr': 2,
#     'ssfp': 2,
# }
# params = {
#     'spgr': ['nex'],
#     'ssfp': ['theta', 'nex'],
# }
# start_range = {
#     'spgr': [(0.1, 5.)],
#     'ssfp': [(0., np.pi*70./180.), (0.1, 5.)],
# }

cost_types = OrderedDict([
    # ('sum 1/cov', {'combine': np.sum, 'cov': False}),
    ('sum cov', {'combine': np.sum, 'cov': True}),
    # {'combine': np.max, 'cov': True},
    # {'combine': np.max, 'cov': False},
])

# 3T WM T1=1100, T2=60 (Stanisz 2005)
# t1 = np.array([1100.])
# t2 = np.array([60.])
# m0 = np.array([1.0])

# 3T GM T1=1645, T2=85 (Stanisz 2005)
t1 = np.array([1645.])
t2 = np.array([85.])
m0 = np.array([1.0])
off_resonance_phase = np.array([0.])
# t1, t2, off_resonance_phase = flatten_space(t1, t2, off_resonance_phase)


plain_problems = OrderedDict([
    # ('Deoni 2003-SSFP', {
    #     'cost_func': plain_despot2_cost_function,
    #     'L': np.array([1., 0.]),
    #     'wrt_in': ('t2', 'm0'),
    #     'wrt_out_ssfp': ('magnitude',),
    #     'protocol_frameworks': [{'ssfp': el} for el in range(1, 4)],
    #     'params': {'ssfp': ['theta', 'nex']},
    #     # 'start_range': {'ssfp': [(np.pi/180., np.pi*70./180.), (0.1, 5.)]},
    #     'constraints': ['SPGR equal nex'],
    #     'cost_types': cost_types,
    # }),
    ('Deoni 2003-Equal NEX', {
        'cost_func': plain_despot2_cost_function,
        'L': np.array([1., 1., 0.]),
        'wrt_in': ('t1', 't2', 'm0'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': [{'spgr': el, 'ssfp': el} for el in [2]],
        'params': {'spgr': ['nex'], 'ssfp': ['nex']},
        # 'start_range': {'spgr': [(0.1, 5.)], 'ssfp': [(0.1, 5.)]},
        'constraints': ['SPGR equal nex', 'SSFP equal nex'],
        'cost_types': cost_types,
    }),
    ('Deoni 2003-Free NEX', {
        'cost_func': plain_despot2_cost_function,
        'L': np.array([1., 1., 0.]),
        'wrt_in': ('t1', 't2', 'm0'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': [{'spgr': el, 'ssfp': el} for el in [2]],
        'params': {'spgr': ['nex'], 'ssfp': ['nex']},
        # 'start_range': {'spgr': [(0.1, 5.)], 'ssfp': [(0.1, 5.)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
])


# Create protocols from all combinations of SPGR and SSFP totalling 3-6 distinct images and keep only those with at least 1 SSFP
joint_combinations = chain.from_iterable([combinations_with_replacement(('spgr', 'ssfp'), total_n) for total_n in range(4, 7)])
joint_combinations = map(Counter, joint_combinations)
joint_combinations = [el for el in joint_combinations if 'ssfp' in el]
joint_problems = OrderedDict([
    # ('Joint, free phase_rf, magn', {
    #     'cost_func': despot2_cost_function,
    #     'L': np.array([1., 1., 0.]),
    #     'wrt_in': ('t1', 't2', 'm0'),
    #     'wrt_out_ssfp': ('magnitude',),
    #     'protocol_frameworks': [{'ssfp': el} for el in range(4, 7)],
    #     'params': {'spgr': ['theta', 'nex'], 'ssfp': ['theta', 'nex', 'phase_rf']},
    #     # 'start_range': {'spgr': [(np.pi/180., np.pi*70./180.), (0.1, 5.)], 'ssfp': [(np.pi/180., np.pi*70./180.), (0.1, 5.), (-np.pi, np.pi)]},
    #     'constraints': [],
    #     'cost_types': cost_types,
    # }),
])

problems = plain_problems.copy()
problems.update(joint_problems)

start_ranges = {
    'theta': (np.pi/180., np.pi*70./180.),
    'nex': (0.1, 5.),
    'phase_rf': (-np.pi, np.pi),
}

transforms = {
    # 'theta': partial(bounded_box_sigmoid_transform, lower=0., upper=np.pi/2.),
    # 'nex': normalize_transform,
    # 'phase_rf': partial(bounded_box_sigmoid_transform, lower=-np.pi, upper=np.pi),
    'theta': None,
    'nex': None,
    'phase_rf': None,
}

if __name__ == '__main__':
    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    print joint_combinations
    
    database = shelve.open('db_constrained_despot2')
    for name, problem in problems.iteritems():
        problem['start_range'] = {seq: [start_ranges[param] for param in params] for seq, params in problem['params'].items()}
        # problem['transforms'] = {seq: [transforms[param] for param in params] for seq, params in problem['params'].items()}
        print '\n\n_________________________________________'
        print name
        print '_________________________________________'
        print 'Lambda: %s' % zip(problem['wrt_in'], problem['L'])
        print 'Problem:'
        for el in problem.iteritems():
            print '\t%s: %s' % el
        database[name] = optimize_despot2_protocol(t1=t1, t2=t2, m0=m0, off_resonance_phase=off_resonance_phase, **problem)
    database.close()
