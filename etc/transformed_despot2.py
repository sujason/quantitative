from constrained_despot2 import *
from variable_transforms import *


def construct_despot2_cost_function(cost_func, n_images, params, start_range, transforms, constraints):
    """
    Arguments:
        cost_func --
        params -- the parameters that are free to vary for each sequence
        start_range -- 
        constraints -- 
    """
    DESPOT2_Cost_Function = MultiOptimizationHelper(cost_func, n_images, params=params, start_range=start_range, transforms=transforms)
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
    'theta': partial(ascending_bounded_box_transform, lower=0., upper=np.pi/2.),
    'nex': partial(normalize_transform, total=10.),
    'phase_rf': partial(bounded_box_sigmoid_transform, lower=-np.pi, upper=np.pi),
}

if __name__ == '__main__':
    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    print joint_combinations
    
    database = shelve.open('db_transformed_despot2')
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
