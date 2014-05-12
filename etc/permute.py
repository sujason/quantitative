import sys
#import dumbdbm
import shelve
import numpy as np
import scipy.linalg as spla
from functools import partial
from itertools import combinations_with_replacement
from collections import Counter
from numpy import newaxis
from higherad import HigherAD
from opt_helper import MultiOptimizationHelper
from multistart import MultiStart
from despot import spgr, ssfp_after
from crlb import calc_crlb
from costtools import *


FLOAT_MAX = np.finfo(np.float).max


SPGR = HigherAD(spgr)
SSFP_After = HigherAD(ssfp_after)
min_tr = 5.0


def despot2_cost_function(
        combine=np.sum,
        cov=True,
        L=np.atleast_1d([1.0]),
        spgr=None,
        ssfp=None,
        default_tr=min_tr,
        t1=np.array([1.0]),
        t2=np.array([1.0]),
        m0=np.array([1.0]),
        off_resonance_phase=np.array([0.0]),
        wrt_in = ('t1', 't2', 'm0', 'off_resonance_phase'),
        wrt_out_ssfp=('magnitude', 'phase'),
        regularization=0.
):
    if spgr is not None:
        spgr_theta = spgr.get('theta', np.atleast_1d([np.pi/4]))
        spgr_nex_orig = spgr_nex = spgr.get('nex', np.atleast_1d([1.0]))
        spgr_tr = spgr.get('tr', np.atleast_1d(default_tr))
        if regularization == 0:
            spgr_nex, (spgr_theta, spgr_tr) = remove_small_nex(spgr_nex, (spgr_theta, spgr_tr))
    if ssfp is not None:
        ssfp_theta = ssfp.get('theta', np.atleast_1d([np.pi/4]))
        ssfp_nex_orig = ssfp_nex = ssfp.get('nex', np.atleast_1d([1.0]))
        ssfp_tr = ssfp.get('tr', np.atleast_1d(default_tr))
        ssfp_phase_rf = ssfp.get('phase_rf', np.atleast_1d([np.pi]))
        if regularization == 0:
            ssfp_nex, (ssfp_theta, ssfp_tr, ssfp_phase_rf) = remove_small_nex(ssfp_nex, (ssfp_theta, ssfp_tr, ssfp_phase_rf))

    # Estimating T1, T2, and M0, calculate Jacobian for each sample tissue
    Js_seq = []
    nex = []
    if spgr is not None:
        #if spgr_nex.size:
        Js_spgr = sort_jacobians(SPGR.jacobian(
            spgr_theta,
            spgr_tr,
            t1[:, newaxis],
            m0[:, newaxis],
            wrt_in=wrt_in,
        ))
        Js_spgr = np.nan_to_num(Js_spgr)
        Js_seq.append(Js_spgr)
        nex.append(spgr_nex)
    if ssfp is not None:
        #if ssfp_nex.size:
        Js_ssfp = sort_jacobians(SSFP_After.jacobian(
            ssfp_theta,
            ssfp_phase_rf,
            ssfp_tr,
            t1[:, newaxis],
            t2[:, newaxis],
            m0[:, newaxis],
            off_resonance_phase[:, newaxis],
            wrt_in=wrt_in,
            wrt_out=wrt_out_ssfp,
        ))
        Js_ssfp = np.nan_to_num(Js_ssfp)
        Js_seq.append(Js_ssfp)
        nex.extend((ssfp_nex, )*len(wrt_out_ssfp))
    # Concatenate along protocol outputs dimension
    Js = np.concatenate(Js_seq, axis=-2)
    # Merge tissue dimensions
    # TODO use this instead of flatten_space
    #Js = Js.reshape((-1,)+Js.shape[-2:])
    noise_variance = 1.0/np.concatenate(nex)
    try:
        crlb = np.array([calc_crlb(J, noise_variance, regularization=regularization) for J in Js])
    except spla.LinAlgError:
        print 'CRLB Error'
        print Js.shape
        print Js[0].shape
        raise
        #return 1e12*np.ones(len(wrt_in))
    if combine is None:
        return crlb
    # Minimize integrated CoV
    #return np.sum(crlb*L/(np.array([locals()[el] for el in wrt_in]).T)
    #return np.sum(crlb[:, 0]/t1 + L*crlb[:, 1]/m0, axis=0)
    #return -np.sum(t1/crlb[:, 0] + L*m0/crlb[:, 1], axis=0) #t1nr
    #return area_underneath(crlb[:, 0]/t1 + L*crlb[:, 1]/m0, t1)
    #return np.sum(crlb[:, 0] + L*crlb[:, 1], axis=0)
    #return area_underneath(crlb[:, 0] + L*crlb[:, 1], t1)

    # Use absolute units for off_resonance_phase precision because can easily blow up at on-resonance.
    true_values = np.broadcast_arrays(*[np.ones_like(off_resonance_phase, dtype=np.float) if tissue_param == 'off_resonance_phase' else locals()[tissue_param] for tissue_param in wrt_in])
    true_values = np.vstack(true_values).T
    if cov:
        return combine((L*crlb/true_values).sum(axis=0))
    else:
        return combine((-L*true_values/crlb).sum(axis=0))
    # if cov:
    #     return combine(L_t1*crlb[:, 0]/t1 + L_t2*crlb[:, 1]/t2 + L_m0*crlb[:, 2]/m0 + L_off_resonance_phase*crlb[:, 3])
    # else:
    #     return combine(-L_t1*t1/crlb[:, 0] - L_t2*t2/crlb[:, 1] - L_m0*m0/crlb[:, 2] - L_off_resonance_phase*crlb[:, 3])


if __name__ == '__main__':
    # n_images = {
    #     'spgr': 2,
    #     'ssfp': 2,
    # }
    params = {
        'spgr': ['theta', 'nex'],
        'ssfp': ['theta', 'nex', 'phase_rf'],
    }
    start_range = {
        'spgr': [(0., np.pi*90./180.), (0.1, 5.)],
        'ssfp': [(0., np.pi*70./180.), (0.1, 5.), (-np.pi, np.pi)],
    }

    #t1 = np.linspace(500., 5000., 20)
    t1 = np.linspace(500., 3000., 3)
    t2 = np.linspace(20., 200., 3)
    m0 = np.array([1.0])
    # Ignore the last point because phase is cyclic.
    off_resonance_phase = np.linspace(-np.pi, np.pi, 4)[:-1]
    _, t1, t2, off_resonance_phase = flatten_space(t1, t2, off_resonance_phase)
    L_t1 = 1.0
    L_t2 = 1.0
    L_m0 = 0.0
    L_off_resonance_phase = 0.0
    L = np.array([L_t1, L_t2, L_m0, L_off_resonance_phase])

    print 't1', t1
    print 't2', t2
    print 'off_resonance_phase', off_resonance_phase
    print 'L_t1, L_t2, L_m0, L_off_resonance_phase', L_t1, L_t2, L_m0, L_off_resonance_phase

    cost_types = [
        {'combine': np.sum, 'cov': True},
        # {'combine': np.max, 'cov': True},
        # {'combine': np.sum, 'cov': False},
        # {'combine': np.max, 'cov': False},
    ]

    #dumbdbm.open('permutedb')
    store_solutions = shelve.open('permutedb')
    for total_n_images in range(3, 11):  # need at least 3 for M0, T1, T2
        combinations = map(Counter, combinations_with_replacement(params, total_n_images))
        combinations = [el for el in combinations if 'ssfp' in el]
        for n_images in combinations:
            print '\n\n========== SOLVING %s ==========' % (n_images, )
            p = {k: v for k, v in params.items() if k in n_images}
            sr = {k: v for k, v in start_range.items() if k in n_images}
            DESPOT2_Cost_Function = MultiOptimizationHelper(despot2_cost_function, n_images, params=p, start_range=sr)
            for seq in n_images:
                # theta must be > 0
                # TODO name the constants
                DESPOT2_Cost_Function.add_affine_constraint((seq, 'theta'), 'ineq')
                DESPOT2_Cost_Function.add_affine_constraint((seq, 'theta'), 'ineq', 1.0, -1./32767.)
                # nex must be > 1 to avoid singular F
                DESPOT2_Cost_Function.add_affine_constraint((seq, 'nex'), 'ineq', 1.0, -0.1)
            if 'spgr' in n_images:
                # theta must be < 90 for spgr
                DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', -1.0, np.pi*80./180.)
            if 'ssfp' in n_images:
                # theta must be < 70 for ssfp due to SAR
                DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'theta'), 'ineq', -1.0, np.pi*70./180.)
                # -pi < phase_rf < pi
                DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq', 1.0, np.pi)
                #DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq') # lower bound of 0, symmetry makes pi redundant?
                DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq', -1.0, np.pi)

            # Fixed time constraint
            # TODO: modify this to account for difference in min_tr bet seq.
            indices = DESPOT2_Cost_Function.parameter_indices
            print 'Indices:', indices
            if 'spgr' in n_images and 'ssfp' in n_images:
                DESPOT2_Cost_Function.constraints.append({
                    'type': 'eq',
                    'fun': lambda x, spgr_nex_idx=indices['spgr']['nex'], ssfp_nex_idx=indices['ssfp']['nex']: sum(x[spgr_nex_idx]) + sum(x[ssfp_nex_idx]) - 10.
                    #'jac': constant_nex_jac,
                })
            elif 'ssfp' in n_images:
                DESPOT2_Cost_Function.constraints.append({
                    'type': 'eq',
                    'fun': lambda x, ssfp_nex_idx=indices['ssfp']['nex']: sum(x[ssfp_nex_idx]) - 10.
                    #'jac': constant_nex_jac,
                })
            print 'Constraints:', DESPOT2_Cost_Function.constraints

            cost_func = partial(DESPOT2_Cost_Function, t1=t1, t2=t2, m0=m0, L=L)
            print 'Compile Theano for floats'
            _ = cost_func(np.arange(len(DESPOT2_Cost_Function.start_range), dtype=np.float))

            M = MultiStart(
                100,
                DESPOT2_Cost_Function.start_range,
                constraints=DESPOT2_Cost_Function.constraints,
                method='SLSQP',
            )

            for i, cost_type in enumerate(cost_types):
                res = M.solve(
                    parallel_pool=0,
                    fun=partial(cost_func, **cost_type),
                    label=str(n_images)
                )
                store_solutions['%s %s' % (n_images, cost_type)] = M.candidates
    store_solutions.close()
