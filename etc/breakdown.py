import sys
import numpy as np
import matplotlib as mpl
import scipy.linalg as spla
from functools import partial
from numpy import newaxis
from higherad import HigherAD
from opt_helper import MultiOptimizationHelper
from multistart import MultiStart
from despot import spgr, ssfp_after
from parallel import BetterPool


FLOAT_MAX = np.finfo(np.float).max


def flatten_space(*args):
    expanded_args = [e.reshape([len(e)]+[1]*i) for i, e in enumerate(args)]
    expanded_args = np.broadcast_arrays(*expanded_args)
    return [e.flatten() for e in expanded_args]


def sort_jacobians(Js):
    # Input comes in with dimensions:
    #  (n_outputs_func, n_inputs_func, param_1, ..., param_m, n_out_protocol)
    # where:
    #  n_outputs_func = the number of ++outputs of the function (rows in J)
    #  n_inputs_func = the number of inputs of the function (columns of J)
    #  param_1 ... param_m = vectors of test cases where J is evaluated
    #  n_out_protocol = the number of outputs for a given "protocol",
    #    i.e. the group of function evaluations that are measured
    # Output combines the function outputs with protocol outputs:
    #  (param1, ..., param_m, n_out_protocol*n_outputs_func, n_inputs_func)

    # (n_out_protocol, n_outputs_func, n_inputs_func, param_1, ..., param_m)
    out = np.rollaxis(Js, Js.ndim - 1)
    s = out.shape
    # (n_out_protocol*n_outputs_func, n_inputs_func, param_1, ..., param_m)
    # Grouped so that protocol ouputs are the inner loop/tighter grouping
    out = out.reshape((s[0]*s[1],)+s[2:], order='F')
    dim = range(out.ndim)
    # (param1, ..., param_m, n_out_protocol*n_outputs_func, n_inputs_func)
    return out.transpose(dim[2:] + dim[:2])


def area_underneath(y, x=None):
    # Compute the integral of a linearly interpolated y.
    if x is None or np.atleast_1d(x).size == 1:
        diff_x = 1.0
    else:
        diff_x = np.diff(x)
    integrand = np.convolve(np.atleast_1d(y), np.array([0.5, 0.5]), 'valid')*diff_x
    return integrand.sum()


def test_sort_jacobians():
    pass


def calc_crlb(J, Sigma=1.0):
    # CRLB for an unbiased estimator.  Sigma is noise covariance matrix.
    Sigma, _ = np.broadcast_arrays(Sigma, np.zeros(J.shape[0]))
    if Sigma.ndim == 1:
        Sigma = np.diag(Sigma)
    #F = (J.T).dot(inv(Sigma)).dot(J)
    try:
        F = (J.T).dot(spla.solve(Sigma, J, sym_pos=True, check_finite=True))
        #print F
    except ValueError:
        print 'Sigma=', Sigma
        print 'J=', J
        raise
    except spla.LinAlgError:
        print 'WARNING: Sigma is a singular matrix.'
        print 'Sigma=', np.diag(Sigma)
        #return np.zeros(J.shape[1])*np.nan
        #return np.zeros(J.shape[1])+FLOAT_MAX
        raise
    try:
        # would rather it die for rank-deficient F than succeed with pseudoinverse
        #F_inv =  spla.pinvh(F, check_finite=True)
        F_inv = spla.inv(F, check_finite=True)
    except spla.LinAlgError:
        print 'WARNING: F is a singular matrix.'
        print 'J=', J
        print 'F=', F
        #return np.zeros(J.shape[1])*np.nan
        #return np.zeros(J.shape[1])+FLOAT_MAX
        raise
    return np.sqrt(np.diag(F_inv))


def calc_crlb_direct(J, Sigma=1.0):
    # this is incorrect
    # Direct calculation of F^(-1), requires J
    Sigma, _ = np.broadcast_arrays(Sigma, np.zeros(J.shape[0]))
    if Sigma.ndim == 1:
        Sigma = np.diag(Sigma)
    try:
        F_inv = spla.solve(J, Sigma).dot(spla.pinv(J.T))
    except spla.LinAlgError:
        print 'WARNING: Singular matrix.'
        return np.sqrt(np.zeros(J.shape[1])+FLOAT_MAX)
    return np.sqrt(np.diag(F_inv))


def remove_small_nex(nex, data, threshold=1e-6):
    nex, _ = np.broadcast_arrays(nex, data[0])
    not_small = nex > threshold
    def try_filter(x):
        try:
            return x[not_small]
        except ValueError:
            # when x doesn't have enough entries
            return x
        except TypeError:
            print x
            raise
    nex = nex[not_small]
    data_out = [try_filter(el) for el in data]
    return nex, data_out


SPGR = HigherAD(spgr)
SSFP_After = HigherAD(ssfp_after)
min_tr = 5.0


def despot2_cost_function(
        combine=np.sum,
        cov=True,
        L_t2=np.array([1.0]),
        L_m0=np.array([1.0]),
        L_off_resonance_phase=np.array([1.0]),
        spgr=None,
        ssfp=None,
        t1=np.array([1.0]),
        t2=np.array([1.0]),
        m0=np.array([1.0]),
        off_resonance_phase=np.array([0.0]),
):
    spgr_theta = spgr.get('theta', np.atleast_1d([np.pi/4]))
    spgr_nex = spgr.get('nex', np.atleast_1d([1.0]))
    spgr_tr = spgr.get('tr', np.atleast_1d(min_tr))
    
    ssfp_theta = ssfp.get('theta', np.atleast_1d([np.pi/4]))
    ssfp_nex = ssfp.get('nex', np.atleast_1d([1.0]))
    ssfp_tr = ssfp.get('tr', np.atleast_1d(min_tr))
    ssfp_phase_rf = ssfp.get('phase_rf', np.atleast_1d([np.pi]))
    
    #print 'DESPOT2 diagnostic:'
    #print 'SPGR', spgr_nex, spgr_theta
    #print 'SSFP', ssfp_nex, ssfp_theta
    spgr_nex, (spgr_theta, spgr_tr) = remove_small_nex(spgr_nex, (spgr_theta, spgr_tr))
    ssfp_nex, (ssfp_theta, ssfp_tr, ssfp_phase_rf) = remove_small_nex(ssfp_nex, (ssfp_theta, ssfp_tr, ssfp_phase_rf))
    #print 'SPGR', spgr_nex, spgr_theta
    #print 'SSFP', ssfp_nex, ssfp_theta
    wrt_in = ('t1', 't2', 'm0', 'off_resonance_phase')
    # Estimating T1, T2, and M0, calculate Jacobian for each sample tissue
    Js_seq = []
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
        wrt_out=('magnitude', 'phase')
    ))
    Js_ssfp = np.nan_to_num(Js_ssfp)
    Js_seq.append(Js_ssfp)
    # TODO: Problematic if different number of images for diff seq but we've limited ourselves to that with this scheme and OptHelper
    #Js_spgr, Js_ssfp = np.broadcast_arrays(Js_spgr, Js_ssfp)
    # Concatenate along protocol outputs dimension
    Js = np.concatenate(Js_seq, axis=-2)
    # Merge tissue dimensions
    #Js = Js.reshape((-1,)+Js.shape[-2:])
    noise_variance = 1.0/np.concatenate((spgr_nex, ssfp_nex, ssfp_nex))
    try:
        crlb = np.array([calc_crlb(J, noise_variance) for J in Js])
    except spla.LinAlgError:
        print Js.shape
        print Js[0].shape
        raise
        #return 1e12*np.ones(len(wrt_in))
    
    # Minimize integrated CoV
    #return np.sum(crlb*L/(np.array([locals()[el] for el in wrt_in]).T)
    #return np.sum(crlb[:, 0]/t1 + L*crlb[:, 1]/m0, axis=0)
    #return -np.sum(t1/crlb[:, 0] + L*m0/crlb[:, 1], axis=0) #t1nr
    #return area_underneath(crlb[:, 0]/t1 + L*crlb[:, 1]/m0, t1)
    #return np.sum(crlb[:, 0] + L*crlb[:, 1], axis=0)
    #return area_underneath(crlb[:, 0] + L*crlb[:, 1], t1)
    if cov:
        return combine(crlb[:, 0]/t1 + L_t2*crlb[:, 1]/t2 + L_m0*crlb[:, 2]/m0 + L_off_resonance_phase*crlb[:, 3])
    else:
        return combine(-t1/crlb[:, 0] - L_t2*t2/crlb[:, 1] - L_m0*m0/crlb[:, 2] - L_off_resonance_phase*crlb[:, 3])
    # Use absolute units for off_resonance_phase precision because can easily blow up at on-resonance.
    

# <codecell>


n_images = {
    'spgr': 2,
    'ssfp': 2,
}
params = {
    'spgr': ['theta', 'nex'],
    'ssfp': ['theta', 'nex', 'phase_rf'],
}
start_range = {
    'spgr': [(0., np.pi*90./180.), (0.1, 5.)],
    'ssfp': [(0., np.pi*70./180.), (0.1, 5.), (-np.pi, np.pi)],
}

#from itertools import combinations_with_replacement
#from collections import Counter

#total_n_images = 4
#map(Counter, combinations_with_replacement(params, total_n_images))

DESPOT2_Cost_Function = MultiOptimizationHelper(despot2_cost_function, n_images, params, start_range)
# theta must be > 0
# TODO name the constants
DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq')
DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'theta'), 'ineq')
DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', 1.0, -1./32767.)
DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'theta'), 'ineq', 1.0, -1./32767.)
# nex must be > 1 to avoid singular F
DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'nex'), 'ineq', 1.0, -0.1)
DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'nex'), 'ineq', 1.0, -0.1)

# theta must be < 90 for spgr and 70 for ssfp due to SAR
DESPOT2_Cost_Function.add_affine_constraint(('spgr', 'theta'), 'ineq', -1.0, np.pi*80./180.)
DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'theta'), 'ineq', -1.0, np.pi*70./180.)

# -pi < phase_rf < pi
DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq', 1.0, np.pi)
#DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq') # lower bound of 0, symmetry makes pi redundant?
DESPOT2_Cost_Function.add_affine_constraint(('ssfp', 'phase_rf'), 'ineq', -1.0, np.pi)

# Fixed time constraint
# TODO: modify this to account for difference in min_tr bet seq.
indices = DESPOT2_Cost_Function.parameter_indices
print indices
DESPOT2_Cost_Function.constraints.append({
    'type': 'eq',
    'fun': lambda x, spgr_nex_idx=indices['spgr']['nex'], ssfp_nex_idx=indices['ssfp']['nex']: sum(x[spgr_nex_idx]) + sum(x[ssfp_nex_idx]) - 10.
    #'jac': constant_nex_jac,
})
print DESPOT2_Cost_Function.constraints

# <codecell>
#t1 = np.linspace(500., 5000., 20)
t1 = np.linspace(500., 3000., 3)
t2 = np.linspace(20., 200., 3)
m0 = np.array([1.0])
# Ignore the last point because phase is cyclic.
off_resonance_phase = np.linspace(-np.pi, np.pi, 4)[:-1]
t1, t2, off_resonance_phase = flatten_space(t1, t2, off_resonance_phase)
L_t2 = np.array([1.0])
L_m0 = np.array([0.0])
L_off_resonance_phase = np.array([0.0])

print 't1', t1
print 't2', t2
print 'off_resonance_phase', off_resonance_phase
print 'L_t2, L_m0, L_off_resonance_phase', L_t2, L_m0, L_off_resonance_phase

cost_func = partial(DESPOT2_Cost_Function, t1=t1, t2=t2, m0=m0, L_t2=L_t2, L_m0=L_m0, L_off_resonance_phase=L_off_resonance_phase)
print 'Compile Theano for floats'
print cost_func(np.arange(len(DESPOT2_Cost_Function.start_range), dtype=np.float))
# <codecell>

print DESPOT2_Cost_Function.start_range

M = MultiStart(
    100,
    DESPOT2_Cost_Function.start_range,
    constraints=DESPOT2_Cost_Function.constraints,
    method='SLSQP',
)

cost_types = [
    {'combine': np.sum, 'cov': True},
#    {'combine': np.max, 'cov': True},
#    {'combine': np.sum, 'cov': False},
#    {'combine': np.max, 'cov': False},
]
solns = []
for i, cost_type in enumerate(cost_types):
    res = M.solve(
        parallel_pool=4,
        # start_points=[{'x0': np.array([ 0.6417117 ,  4.21054067,  0.59139877,  3.12048133,  0.77893495,
        #     3.32466537, -1.91358885,  0.66419318,  0.932947  ,  2.02531311])}],
        fun=partial(cost_func, **cost_type),
        label=str(n_images)
    )
    solns.append(res)

# <codecell>

print '=============== Solutions ==============='
print solns
print M.candidates

import pickle
pickle.dump(M.candidates, open( "save.p", "wb" ))
# <codecell>
