import numpy as np
import scipy.optimize as opt
from numpy import sqrt, sin, cos, exp


def magnitude_phase(Mx, My):
    return {
        'x': Mx,
        'y': My,
        'magnitude': sqrt(Mx**2 + My**2),
        'phase': np.arctan2(My, Mx)
    }


def spgr(alpha, tr, t1, m0=1.0):
    e1 = exp(-tr/t1)
    return m0*(1.0-e1)*sin(alpha)/(1.0-e1*cos(alpha));


def ernst_spgr(tr, t1, m0=1.0):
    alpha = np.arccos(exp(-tr/t1))
    return {'alpha': alpha, 'spgr': spgr(alpha, tr, t1, m0)}


def ssfp_after(alpha, phase_rf, tr, t1, t2, m0=1.0, off_resonance_phase=0.0):
    """
    After RF excitation.  Lauzon 2009, Freeman 1971.
    """
    e1 = exp(-tr/t1)
    e2 = exp(-tr/t2)
    a = alpha
    b = phase_rf + off_resonance_phase
    My = m0*(1.0-e1)*sin(a)*(1.0-e2*cos(b))/((1.0-e1*cos(a))*(1.0-e2*cos(b))-e2*(e1-cos(a))*(e2-cos(b)));
    Mx = m0*(1.0-e1)*e2*sin(a)*sin(b)/((1.0-e1*cos(a))*(1.0-e2*cos(b))-e2*(e1-cos(a))*(e2-cos(b)));
    return magnitude_phase(Mx, My)


def ssfp_before(alpha, phase_rf, tr, t1, t2, m0=1.0, off_resonance_phase=0.0):
    """
    Before RF excitation.  Deoni 2009, Freeman 1971.
    """
    e1 = exp(-tr/t1)
    e2 = exp(-tr/t2)
    a = alpha
    b = phase_rf + off_resonance_phase
    My = m0*(1.0-e1)*e2*sin(a)*(cos(b)-e2)/((1.0-e1*cos(a))*(1.0-e2*cos(b))-e2*(e1-cos(a))*(e2-cos(b)));
    Mx = m0*(1.0-e1)*e2*sin(a)*sin(b)/((1.0-e1*cos(a))*(1.0-e2*cos(b))-e2*(e1-cos(a))*(e2-cos(b)));
    return magnitude_phase(Mx, My)


def ernst_ssfp(tr, t1, t2, m0=1.0):
    """
    The analytic expression for the maximum of the bSSFP curve (which is invariant to phase-cycling).
    Derived symbolically in MATLAB, Jul 2013.
    """
    phase_rf = np.pi
    e1 = exp(-tr/t1)
    e2 = exp(-tr/t2)
    alpha = np.arccos((e1-e2)/(1-e1*e2))
    signal_after = ssfp_after(alpha, phase_rf, tr, t1, t2, m0)
    signal_before = ssfp_before(alpha, phase_rf, tr, t1, t2, m0)
    return {'alpha': alpha, 'ssfp_after': signal_after, 'ssfp_before': signal_before}


def despot1_optimal_angles(tr, t1, f=2.**(-0.5)):
    """
    Solve for the optimal DESPOT1/VFA angles with f, the fraction of the max signal, =1/sqrt(2) by default.
    ref. Deoni et al. MRM 2003
    """
    e1 = exp(-tr/t1)
    f2 = f**2.
    num = f2*e1 + (1. - e1**2.) * sqrt(1. - f2) * np.array([1., -1.])
    den = 1. - e1**2. * (1. - f2)
    return np.arccos(num/den)


def despot2_optimal_angles(tr, t1, t2, f=2.**(-0.5)):
    """
    Solve for the optimal DESPOT2 angles with f, the fraction of the max signal, =1/sqrt(2) by default.
    The underlying sequence is assumed to be bSSFP with pi phase cycling
    ref. Deoni et al. MRM 2003
    """
    e1 = exp(-tr/t1)
    e2 = exp(-tr/t2)
    f2 = f**2.
    psi = (e1 - e2) / (1. - e1*e2)
    A = -f2 * (1.-psi**2.) * (e1-e2)**2. - (1. - e1*e2 - (e1-e2)*psi)**2.
    B = 2.*f2*(1.-psi**2.)*(e1-e2) - 2.*f2*(1.-psi**2.)*e1*e2*(e1-e2)
    C = (1. - e1*e2 - (e1-e2)*psi)**2. - f2*(1.-psi**2.)*(1. - 2.*e1*e2 + e1**2.*e2**2.)
    return np.arccos((-B - sqrt(B**2. - 4*A*C) * np.array([1., -1.])) / (2.*A))


def despot2_optimal_angles_numerical(phase_rf, tr, t1, t2, f=2.**(-0.5)):
    """
    Numerically solve for the optimal DESPOT2 angles with f, the fraction of the max signal, =1/sqrt(2) by default.
    ref. Deoni et al. MRM 2003
    """
    # The max signal is invariant with phase cycle.
    max_signal = ernst_ssfp(tr, t1, t2)
    offset = f*max_signal['ssfp_after']['magnitude']

    def offset_ssfp(alpha, offset=offset):
        return ssfp_after(alpha, phase_rf, tr, t1, t2)['magnitude'] - offset

    if phase_rf == np.pi:
        alpha_max = max_signal['alpha']
    else:
        alpha_max = opt.minimize(lambda alpha: -offset_ssfp(alpha), np.pi/4.)
        # Sometimes it finds the negative alpha solution, but it's symmetric
        alpha_max = np.abs(alpha_max.x)
    a = opt.brentq(offset_ssfp, 0., alpha_max)
    b = opt.brentq(offset_ssfp, alpha_max, np.pi)
    return np.array([a, b])
