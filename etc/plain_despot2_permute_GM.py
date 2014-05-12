from plain_despot2_permute import *

# 3T GM T1=1645, T2=85 (Stanisz 2005)
t1 = np.array([1645.])
t2 = np.array([85.])
m0 = np.array([1.0])
off_resonance_phase = np.array([0.])
# t1, t2, off_resonance_phase = flatten_space(t1, t2, off_resonance_phase)

if __name__ == '__main__':
    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    print joint_combinations
    
    database = shelve.open('db_plain_permute_GM')
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
