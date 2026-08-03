"""
Gas and magnetic-field profile implementations.

All electron-number-density arrays are returned in units of 1/Mpc^3.
Magnetic-field callables return B in eV=1 natural units (multiply Gauss by
1.95e-20 to get GeV; multiply by 1e-6*1.95e-20 to go from microGauss).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict

import numpy as np


# Unit conversions used throughout.
M_P_MSUN = 8.411856872862986e-58    # proton mass in Msun
GAUSS_TO_GEV = 1.95e-20             # 1 Gauss -> GeV in natural units


# ---------------------------------------------------------------------------
# Gas profile base class
# ---------------------------------------------------------------------------

class GasProfile:
    """Abstract base: provide ``ne(r, ...)`` and ``dne_dr(r, ...)`` in 1/Mpc^3."""

    def ne(self, r, M200c, R200c, c200c, z, rho_critz):
        raise NotImplementedError

    def dne_dr(self, r, M200c, R200c, c200c, z, rho_critz):
        raise NotImplementedError


class BattagliaGas(GasProfile):
    """Battaglia et al. (2012) gas-density profile.

    Parameters follow the fits of Battaglia+12 Eq. (10).  A fixed override
    ``param_dict`` can be passed for sensitivity studies.
    """

    def __init__(self, fb: float, *, param_dict: Optional[Dict[str, float]] = None):
        self.fb = fb
        self.param_dict = param_dict

    def _params(self, M200c, z):
        if self.param_dict is not None:
            p = self.param_dict
            return p['rho0'], p['alpha'], p['beta'], p['gamma']
        rho0  = 4000. * (M200c / 1e14) ** 0.29    * (1. + z) ** (-0.66)
        alpha = 0.88  * (M200c / 1e14) ** (-0.03) * (1. + z) **   0.19
        beta  = 3.83  * (M200c / 1e14) ** 0.04    * (1. + z) ** (-0.025)
        gamma = -0.2
        return rho0, alpha, beta, gamma

    def ne(self, r, M200c, R200c, c200c, z, rho_critz):
        rho0, alpha, beta, gamma = self._params(M200c, z)
        x = 2.0 * r / R200c
        rho_gas = (self.fb * rho_critz * rho0
                   * (x ** gamma)
                   * ((1.0 + x ** alpha) ** (-(beta + gamma) / alpha)))
        return (1 + 0.76) / (2 * M_P_MSUN) * rho_gas

    def dne_dr(self, r, M200c, R200c, c200c, z, rho_critz):
        rho0, alpha, beta, gamma = self._params(M200c, z)
        x = 2.0 * r / R200c
        drho_dx = (self.fb * rho_critz * rho0 * (
            gamma * x ** (gamma - 1) * (1.0 + x ** alpha) ** (-(beta + gamma) / alpha)
            + x ** gamma * (-(beta + gamma) / alpha)
              * (1.0 + x ** alpha) ** (-(beta + gamma) / alpha - 1)
              * alpha * x ** (alpha - 1)
        ))
        drho_dr = (2.0 / R200c) * drho_dx
        return (1 + 0.76) / (2 * M_P_MSUN) * drho_dr


class NFWGas(GasProfile):
    """NFW gas profile (for cross-checks; not the fiducial)."""

    def __init__(self, fb: float):
        self.fb = fb

    def _rho_s(self, M200c, R200c, c200c):
        rs = R200c / c200c
        fNFW = (np.log(1 + c200c) - c200c / (1 + c200c)) ** -1
        return (M200c / (4 * np.pi * rs ** 3)) * fNFW, rs

    def ne(self, r, M200c, R200c, c200c, z, rho_critz):
        rho_s, rs = self._rho_s(M200c, R200c, c200c)
        r_rs = r / rs
        rho_gas = self.fb * rho_s / (r_rs * (1 + r_rs) ** 2)
        return (1 + 0.76) / (2 * M_P_MSUN) * rho_gas

    def dne_dr(self, r, M200c, R200c, c200c, z, rho_critz):
        rho_s, rs = self._rho_s(M200c, R200c, c200c)
        r_rs = r / rs
        drho_gas_dr = (-self.fb * rho_s / rs
                       * r_rs ** -1 * (1 + r_rs) ** -2
                       * (1.0 / r_rs + 2.0 / (1 + r_rs)))
        return (1 + 0.76) / (2 * M_P_MSUN) * drho_gas_dr


# ---------------------------------------------------------------------------
# Magnetic-field profile
# ---------------------------------------------------------------------------

class MagneticField:
    """Callable returning B at resonance in GeV units (eV=1).

    Subclasses implement ``__call__(z, Rres, M200c, R200c, rho_critz)``.  By
    convention, returning 0.0 switches off conversion in that cell.
    """

    def __call__(self, z, Rres, M200c, R200c, rho_critz):
        raise NotImplementedError


@dataclass
class ConstantB(MagneticField):
    """Constant magnetic field inside an optional [r_lo, r_hi] * R200c band."""
    B_muG: float = 1.0
    r_lo: Optional[float] = 0.225
    r_hi: Optional[float] = 1.175

    def __call__(self, z, Rres, M200c, R200c, rho_critz):
        x = Rres / R200c
        if self.r_lo is not None and x < self.r_lo:
            return 0.0
        if self.r_hi is not None and x > self.r_hi:
            return 0.0
        return self.B_muG * 1e-6 * GAUSS_TO_GEV


class CallableB(MagneticField):
    """Thin adapter: wrap any plain callable ``fn(z, Rres, M200c, R200c, rho_critz)``."""
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, z, Rres, M200c, R200c, rho_critz):
        return self.fn(z, Rres, M200c, R200c, rho_critz)
