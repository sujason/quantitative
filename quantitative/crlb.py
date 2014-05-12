import numpy as np
import scipy.linalg as spla


def fisher_information(J, Sigma):
    #F = (J.T).dot(inv(Sigma)).dot(J)
    F = (J.T).dot(spla.solve(Sigma, J, sym_pos=True, check_finite=True))
    return np.atleast_2d(F)


def calc_crlb(J, Sigma=1.0, regularization=0.0):
    """
    CRLB for an unbiased estimator.
    Sigma is the noise covariance matrix.
    regularization is an epsilion*I to add to F to stabilize the inversion.
    """
    Sigma, _ = np.broadcast_arrays(Sigma, np.zeros(J.shape[0]))
    if Sigma.ndim == 1:
        Sigma = np.diag(Sigma)
    try:
        F = fisher_information(J, Sigma)
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
        F_inv = spla.inv(F + regularization*np.eye(*F.shape), check_finite=True)
    except spla.LinAlgError:
        print 'WARNING: F is a singular matrix.'
        print 'J=', J
        print 'F=', F
        #return np.zeros(J.shape[1])*np.nan
        #return np.zeros(J.shape[1])+FLOAT_MAX
        raise
    return np.sqrt(np.diag(F_inv))
