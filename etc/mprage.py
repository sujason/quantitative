import numpy as np
from numpy import exp, cos, sin


def ir_t1(ti, a, b, t1):
    return a-b*exp(-ti/t1)


def mp1rage(n_read, i_read, theta, tr, ta, tb, t1, m0=1.0, inv_eff=1.0, kappa=1.0, e2=1.0):
    """
    b1- is incoporated into m0
    i_read is the index of the pulse in the train that reads the center of k-space, indexing begins at 0
        i.e. centric i_read = 0; linear i_read = n_read/2 -1
    ta is same as TI
    tb is the same as TD
    """
    tr_total = ta + (n_read-1.0)*tr + tb
    theta_exc = kappa*theta
    c = cos(theta_exc)
    e1 = exp(-tr/t1)
    rho = e1*c
    rage = (1.0-rho**(n_read-1.0))/(1.0-rho)
    ea = exp(-ta/t1)
    eb = exp(-tb/t1)
    m_1 = 1.0 - inv_eff*ea*eb*(rage*(1.0-e1)*c-1.0) - (1.0+inv_eff)*ea
    m_1 /= 1.0 + inv_eff * c**n_read * exp(-tr_total/t1)
    
    rage_i = (1.0-rho**(i_read))/(1.0-rho)
    m = m0*((1.0-e1)*rage_i + rho**i_read*m_1)
    return sin(theta_exc)*e2*m


def mp1rage_combined(ta_min, ta_med, ta_max, t1):
    return (exp((ta_max-ta_min)/t1)-1), (exp((ta_max-ta_med)/t1) - 1)


def test_mp1rage():
    pass


def mp2rage(n_read, i_read, theta_1, theta_2, tr, ta, tb, tc, t1, m0=1.0, inv_eff=1.0, kappa=1.0, e2=1.0):
    """
    b1- is incorporated into m0
    i_read is the index of the pulse in the train that reads the center of k-space, indexing begins at 0
        i.e. centric i_read = 0; linear i_read = n_read/2 -1
    """
    tr_total = ta + (n_read-1.0)*tr + tb + (n_read-1.0)*tr + tc
    theta_exc_1, theta_exc_2 = kappa*np.array([theta_1, theta_2])
    c1, c2 = cos(theta_exc_1), cos(theta_exc_2)
    e1 = exp(-tr/t1)
    ea, eb, ec = exp(-np.array([ta, tb, tc], dtype=np.float)/t1)
    rho_1, rho_2 = e1*np.array([c1, c2])

    mz_ss = ((((1.0-ea)*rho_1**n_read + (1.0-e1)*(1.0-rho_1**n_read)/(1.0-rho_1))*eb + 
        (1.0-eb))*rho_2**n_read + (1.0-e1)*(1.0-rho_2**n_read)/(1.0-rho_2))*ec + (1.0-ec)
    mz_ss /= 1.0 + inv_eff * (c1*c2)**n_read * exp(-tr_total/t1)

    gre_1 = m0*e2*sin(theta_exc_1) * ((-inv_eff*mz_ss*ea + (1-ea))*rho_1**i_read +
        (1-e1)*(1-rho_1**i_read)/(1-rho_1))
    gre_2 = m0*e2*sin(theta_exc_2) * ((mz_ss - (1-ec))/(ec*rho_2**i_read) - (1-e1)*(rho_2**i_read-1)/(1-rho_2))
    #gre_2 = m0*e2*sin(theta_exc_2) * ((mz_ss - (1-ec))/(ec*rho_2**(i_read+1)) - (1-e1)*(rho_2**-(i_read+1)-1)/(1-rho_2))
    return [gre_1, gre_2]


def test_mp2rage():
    pass


def mp3rage(theta, tr, td, t1):
    pass


def test_mp3rage():
    pass


def mpnrage(theta, tr, td, t1):
    pass


if __name__ == "__main__":
    from higherad import HigherAD
    test_mp1rage()
    test_mp2rage()
    test_mp3rage()
    tr_total = 2180.
    n_read = 16.
    i_read = 0.
    theta = np.pi*9./180.
    tr = 9.
    ta = 60.
    tb = tr_total - ta - (n_read-1)*tr

    t1 = 1000.
    print mp1rage(n_read=n_read, i_read=0, theta=theta, tr=tr, ta=ta, tb=tb, t1=t1)
    MP1RAGE = HigherAD(mp1rage)
    wrt_in = ['t1', 'm0']
    print MP1RAGE.jacobian(n_read=n_read, i_read=0, theta=theta, tr=tr, ta=ta, tb=tb, t1=t1, wrt_in=wrt_in)

    theta_1 = theta
    theta_2 = 0.
    ta2 = ta
    tb2 = 0.0
    tc2 = tr_total - ta - 2*(n_read-1)*tr - tb2
    print mp2rage(n_read=n_read, i_read=i_read, theta_1=theta_1, theta_2=theta_2, tr=tr, ta=ta2, tb=tb2, tc=tc2, t1=t1)
