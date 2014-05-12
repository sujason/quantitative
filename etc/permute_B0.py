from plain_despot2_permute import *

cost_types = OrderedDict([
    ('mean 1/cov', {'combine': np.mean, 'cov': False}),
    ('mean cov', {'combine': np.mean, 'cov': True}),
    ('max 1/cov', {'combine': np.max, 'cov': False}),
    ('max cov', {'combine': np.max, 'cov': True}),
])

# 3T WM T1=1100, T2=60 (Stanisz 2005)
# t1 = np.array([1100.])
# t2 = np.array([60.])
# m0 = np.array([1.0])
# 3T GM T1=1645, T2=85 (Stanisz 2005)
t1 = np.array([1645.])
t2 = np.array([85.])
m0 = np.array([1.0])
# off_resonance_phase = np.array([0.])
off_resonance_phase = np.linspace(0., 2*np.pi, 9)[:-1]
_, t1, t2, off_resonance_phase = flatten_space(t1, t2, off_resonance_phase)


plain_problems = OrderedDict([
    ('DESPOT2-SSFP', {
        'cost_func': plain_despot2_cost_function,
        'L': np.array([1., 0.]),
        'wrt_in': ('t2', 'm0'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': [{'ssfp': el} for el in range(3, 17)],
        'params': {'ssfp': ['theta', 'nex', 'phase_rf']},
        # 'start_range': {'ssfp': [(np.pi/180., np.pi*70./180.), (0.1, 5.)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
    ('B0, DESPOT2-SSFP', {
        'cost_func': plain_despot2_cost_function,
        'L': np.array([1., 0., 0.]),
        'wrt_in': ('t2', 'm0', 'off_resonance_phase'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': [{'ssfp': el} for el in range(3, 17)],
        'params': {'ssfp': ['theta', 'nex', 'phase_rf']},
        # 'start_range': {'ssfp': [(np.pi/180., np.pi*70./180.), (0.1, 5.)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
    # ('Deoni 2003-Equal NEX', {
    #     'cost_func': plain_despot2_cost_function,
    #     'L': np.array([1., 1., 0.]),
    #     'wrt_in': ('t1', 't2', 'm0'),
    #     'wrt_out_ssfp': ('magnitude',),
    #     'protocol_frameworks': [{'spgr': el, 'ssfp': el} for el in range(2, 4)],
    #     'params': {'spgr': ['nex'], 'ssfp': ['nex']},
    #     # 'start_range': {'spgr': [(0.1, 5.)], 'ssfp': [(0.1, 5.)]},
    #     'constraints': ['SPGR equal nex', 'SSFP equal nex'],
    #     'cost_types': cost_types,
    # }),
])


# Create protocols from all combinations of SPGR and SSFP totalling 3-6 distinct images and keep only those with at least 1 SSFP
# joint_combinations = chain.from_iterable([combinations_with_replacement(('spgr', 'ssfp'), total_n) for total_n in range(4, 11)])
# joint_combinations = map(Counter, joint_combinations)
# joint_combinations = [el for el in joint_combinations if 'ssfp' in el]
joint_combinations = [{'ssfp': el} for el in range(4, 17)]
joint_problems = OrderedDict([
    ('Joint, free phase_rf, magn', {
        'cost_func': despot2_cost_function,
        'L': np.array([1., 1., 0.]),
        'wrt_in': ('t1', 't2', 'm0'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': joint_combinations,
        'params': {'spgr': ['theta', 'nex'], 'ssfp': ['theta', 'nex', 'phase_rf']},
        # 'start_range': {'spgr': [(np.pi/180., np.pi*70./180.), (0.1, 5.)], 'ssfp': [(np.pi/180., np.pi*70./180.), (0.1, 5.), (-np.pi, np.pi)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
    ('B0, Joint, free phase_rf, magn', {
        'cost_func': despot2_cost_function,
        'L': np.array([1., 1., 0., 0.]),
        'wrt_in': ('t1', 't2', 'm0', 'off_resonance_phase'),
        'wrt_out_ssfp': ('magnitude',),
        'protocol_frameworks': joint_combinations,
        'params': {'spgr': ['theta', 'nex'], 'ssfp': ['theta', 'nex', 'phase_rf']},
        # 'start_range': {'spgr': [(np.pi/180., np.pi*70./180.), (0.1, 5.)], 'ssfp': [(np.pi/180., np.pi*70./180.), (0.1, 5.), (-np.pi, np.pi)]},
        'constraints': [],
        'cost_types': cost_types,
    }),
])

problems = plain_problems.copy()
problems.update(joint_problems)


if __name__ == '__main__':
    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    print joint_combinations
    
    database = shelve.open('db_permute_B0_to16')
    for name, problem in problems.iteritems():
        problem['start_range'] = {seq: [start_ranges[param] for param in params] for seq, params in problem['params'].items()}
        print '\n\n_________________________________________'
        print name
        print '_________________________________________'
        print 'Lambda: %s' % zip(problem['wrt_in'], problem['L'])
        print 'Problem:'
        for el in problem.iteritems():
            print '\t%s: %s' % el
        database[name] = optimize_despot2_protocol(t1=t1, t2=t2, m0=m0, off_resonance_phase=off_resonance_phase, **problem)
    database.close()
