"""
Run a mechanism.
"""
# import warnings
# from copy import deepcopy as copy
import numpy as np
import pandas as pd
from scipy import integrate

__all__ = (
    "run_exp",
    "ppbv_to_molec_cm3",
    "molec_cm3_to_ppbv",
)


def n_a_m3(p=1000, T=298):
    """Air number density (per m^3) at temperature `T` (K) and pressure `p` (mb/hPa)."""
    from scipy.constants import k as k_B

    p_Pa = p * 100  # hPa -> Pa

    return p_Pa / (k_B * T)


def ppbv_to_molec_cm3(x, p=1000, T=298):
    """Convert `x` in ppbv units to molec cm^-3 units.

    p : float
        units: hPa
    T : float
        units: K
    """
    n_a_cm3 = n_a_m3(p, T) / 1e6  # m^-3 -> cm^-3

    return x * 1e-9 * n_a_cm3


def molec_cm3_to_ppbv(x, p=1000, T=298):
    """Convert `x` in molec cm^-3 units to ppbv units.

    p : float
        units: hPa
    T : float
        units: K
    """
    n_a_cm3 = n_a_m3(p, T) / 1e6  # m^-3 -> cm^-3

    return x / n_a_cm3 * 1e9


c0_default_ppbv = {
    # background non-trace species
    "N2": 0.7808 * 1e9,
    "O2": 0.2095 * 1e9,
    "H2O": 0.01 * 1e9,
}


def run_exp(
    mech,
    *,
    c0_ppbv=None,
    c0_molec_cm3=None,
    fixed=None,
    sources=None,
    p=1000,
    T=298,
    t_tot=10800,
    dt=60,
    method="BDF",
):
    """Run an experiment.

    Parameters
    ----------
    mech : EqnSet
        Mechanism to use.
    c0_ppbv : dict
        Initial concentrations (ppbv) for species. Species not included will be
        assumed to start at 0.
    c0_molec_cm3 : dict
        Like `c0_ppbv` but in molec cm^-3 units.
    fixed : dict
        Fixed rates (dcdt) for species, in molec cm^-3 s^-1 units.
    sources : dict
        Additional constant source/sink terms for species, in molec cm^-3 s^-1 units.
    p, T : float
        Pressure (hPa) and temperature (K). Required in order to convert molec cm^-3 to ppbv.
    t_tot : float or int
        Total run time **in seconds**. Default: 3 hours.
    dt : float
        Time step for output **in seconds**.
        Used to construct ``t_eval`` for the SciPy integrator.
        Default: 1 min.
    method : str
        `scipy.integrate.solve_ivp` ``method`` option. Should be Radau, BDF, or LSODA,
        since those are suited for stiff problems.

    Returns
    -------
    pd.DataFrame
        Concentration time series in molec cm^-3.
    """
    all_spc = mech.spcs
    n_spc = mech.n_spc
    assert n_spc == len(all_spc)

    if c0_ppbv is None:
        c0_ppbv = {}
    if c0_molec_cm3 is None:
        c0_molec_cm3 = {}

    if any(spc in c0_molec_cm3 for spc in c0_ppbv):
        raise Exception(
            "Species initial conc. must be either in c0_ppbv or c0_molec_cm3, not both."
        )

    # Construct initial condition vector
    c0 = np.zeros((n_spc,))  # y0 must be 1-d for solve_ivp
    for spc, c_ppbv in c0_ppbv.items():
        c0[all_spc.index(spc)] = ppbv_to_molec_cm3(c_ppbv, p=p, T=T)
    for spc, c_molec in c0_molec_cm3.items():
        c0[all_spc.index(spc)] = c_molec

    # Construct output times vector
    t = np.arange(0, t_tot + dt, dt)
    t_span = (t[0], t[-1])
    t_td = [pd.Timedelta(x, unit="s") for x in t]

    # Construct function to feed to `solve_ivp`
    # The first two args must be `t, y`
    # currently dcdt doesn't use t directly, just c(t)
    # squeeze to go from col vector to true 1-d
    f = lambda t, y: mech.dcdt_fn(y, fixed=fixed, sources=sources).squeeze()  # noqa: E731

    # Run
    ret = integrate.solve_ivp(f, t_span, c0, t_eval=t, method=method)

    # Save data in DataFrame
    data = ret.y.T
    df = pd.DataFrame(data=data, columns=all_spc, index=t_td)
    # df["t_td"] = t_td

    # Negative check
    assess_neg(df)

    return df


def assess_neg(df):
    """Look for negative concentrations. Silent if none found, otherwise prints."""
    neg = (df < 0).any()
    negs = neg[neg]

    if negs.size:
        print(f"\n{negs.size} species have negative concentrations:")

        n = df.index.size
        for spc in negs.index:
            pct_neg = (df[spc] < 0).sum() / n * 100
            print(f"{spc}: {pct_neg:.3g}%")
