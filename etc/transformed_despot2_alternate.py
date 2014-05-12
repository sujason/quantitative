from constrained_despot2 import *
from variable_transforms import *


@wraps(despot2_cost_function)
def xfm_despot2_cost_function(general=None, *args, **kwargs):
    kwargs['spgr'] = spgr = {}
    spgr['nex'] = general['nex'][:2]
    spgr['tr'] = spgr_tr = np.atleast_1d(min_tr)
    spgr['theta'] = despot1_optimal_angles(spgr_tr, t1.mean())

    kwargs['ssfp'] = ssfp = {}
    ssfp['nex'] = general['nex'][2:]
    ssfp['tr'] = ssfp_tr = np.atleast_1d(min_tr)
    ssfp['theta'] = despot2_optimal_angles(ssfp_tr, t1.mean(), t2.mean())
    ssfp['phase_rf'] = np.atleast_1d(np.pi)
    return despot2_cost_function(*args, **kwargs)


def construct_despot2_cost_function(cost_func, n_images, params, start_range, transforms, constraints):
    """
    Arguments:
        cost_func --
        params -- the parameters that are free to vary for each sequence
        start_range -- 
        constraints -- 
    """
    DESPOT2_Cost_Function = MultiOptimizationHelper(cost_func, n_images, params=params, start_range=start_range, transforms=transforms)
    indices = DESPOT2_Cost_Function.parameter_indices
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
        def normalize_nex_all(x, spgr_nex_idx=indices['spgr']['nex'], ssfp_nex_idx=indices['ssfp']['nex']):
            i = np.concatenate((spgr_nex_idx, ssfp_nex_idx))
            return (i, normalize_transform(x[i]))
        DESPOT2_Cost_Function.pre_array_transforms = [normalize_nex_all]
    elif 'ssfp' in n_images:
        def normalize_nex_ssfp(x, ssfp_nex_idx=indices['ssfp']['nex']):
            i = ssfp_nex_idx
            return (i, normalize_transform(x[i]))
        DESPOT2_Cost_Function.pre_array_transforms = [normalize_nex_ssfp]
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
            # constraints=DESPOT2_Cost_Function.constraints,
            method='L-BFGS-B',
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


start_ranges = {
    'theta': (-1., 1.),
    'nex': (0.1, 1.),
    'phase_rf': (-1., 1.),
}

transforms = {
    'theta': partial(bounded_box_sigmoid_transform, lower=0., upper=np.pi/2.),
    'nex': normalize_transform,
    'phase_rf': partial(bounded_box_sigmoid_transform, lower=-np.pi, upper=np.pi),
}

problems = OrderedDict([
    # ('Deoni 2003-Equal NEX-Mixed', {
    #     'cost_func': plain_despot2_cost_function,
    #     'L': np.array([1., 1., 0.]),
    #     'wrt_in': ('t1', 't2', 'm0'),
    #     'wrt_out_ssfp': ('magnitude',),
    #     'protocol_frameworks': [{'spgr': el, 'ssfp': el} for el in [2]],
    #     'params': {'spgr': ['nex'], 'ssfp': ['nex']},
    #     # 'start_range': {'spgr': [(0.1, 5.)], 'ssfp': [(0.1, 5.)]},
    #     'constraints': ['SPGR equal nex', 'SSFP equal nex'],
    #     'cost_types': cost_types,
    # }),
    ('Deoni 2003-Free NEX-Transforms Only', {
        'cost_func': xfm_despot2_cost_function,
        'L': np.array([1., 1., 0.]),
        'wrt_in': ('t1', 't2', 'm0'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': [{'general': 4}],
        'params': {'general': ['nex']},
        # 'start_range': {'spgr': [(0.1, 5.)], 'ssfp': [(0.1, 5.)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
])

if __name__ == '__main__':
    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    print joint_combinations
    
    database = shelve.open('db_transformed_despot2_alternate')
    for name, problem in problems.iteritems():
        problem['start_range'] = {seq: [start_ranges[param] for param in params] for seq, params in problem['params'].items()}
        problem['transforms'] = {seq: [transforms[param] for param in params] for seq, params in problem['params'].items()}
        print '\n\n_________________________________________'
        print name
        print '_________________________________________'
        print 'Lambda: %s' % zip(problem['wrt_in'], problem['L'])
        print 'Problem:'
        for el in problem.iteritems():
            print '\t%s: %s' % el
        database[name] = optimize_despot2_protocol(t1=t1, t2=t2, m0=m0, off_resonance_phase=off_resonance_phase, **problem)
    database.close()
