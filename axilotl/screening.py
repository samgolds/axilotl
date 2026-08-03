"""
Photon → dark-state conversion physics: resonance finder, conversion
probability, anisotropic screening optical depth τ_ℓ on the halo grid.

Supports dark_comp='axion' out of the box.  dark_comp='dark_photon' is
scaffolded; its conversion probability needs user verification against the
reference formulae before use (see compute_P_conversion).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import eval_legendre
try:
    from scipy.integrate import simps
except:
    from scipy.integrate import simpson as simps
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from .backend import HaloModelBackend
from .profiles import GasProfile, MagneticField


# Natural-unit conversions
NE_MPC3_PER_GEV2 = 2.937998946096347e+73 / 1.376743653116386e-39

#   n_e (Mpc^-3) = m_a_GeV^2 * NE_MPC3_PER_GEV2 at plasma-mass resonance.
MPC_INV4_TO_GEV3 = 2.9988201748857558e-151
#   d n_e / d r   in (Mpc^-4) * this factor = (GeV^3)  for dm_γ²/dr.
HBAR_EVS = 6.58121959825954e-16


# ---------------------------------------------------------------------------
# Resonance finding
# ---------------------------------------------------------------------------

def find_Rres(gas: GasProfile, ne_res, M200c, R200c, c200c, z, rho_critz,
              r_lo=1e-5, r_hi=50.0, rtol=1e-6):
    """
    Return R where ne(R) = ne_res, by bracketed root-finding on
    the **signed** function f(r) = ne(r) - ne_res.

    Assumes ne(r) is monotonically decreasing.  If the target density is not
    bracketed (e.g. too large for any r in the halo), returns ``np.nan``.
    """
    f_lo = gas.ne(r_lo, M200c, R200c, c200c, z, rho_critz) - ne_res
    f_hi = gas.ne(r_hi, M200c, R200c, c200c, z, rho_critz) - ne_res
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(
        lambda r: gas.ne(r, M200c, R200c, c200c, z, rho_critz) - ne_res,
        r_lo, r_hi, rtol=rtol,
    )


def N_res(Rvir, Rres):
    """
    Estimate number of resonances.
    """
    return 2.0 * np.heaviside(Rvir - Rres, 0.5)


# ---------------------------------------------------------------------------
# Screening model
# ---------------------------------------------------------------------------

class ScreeningModel:
    """
    Compute anisotropic screening τ_ℓ on the (z, M, ell) grid.

    Parameters
    ----------
    backend
        :class:`HaloModelBackend` with precomputed halo quantities.
    gas
        :class:`GasProfile` — sets ne(r) profile and dne/dr.
    magnetic
        :class:`MagneticField` — returns B at resonance (natural units).
        Unused for dark-photon conversion.
    dark_comp
        'axion' or 'dark_photon'.
    nu_GHz
        Reference frequency for τ (axion case only).
    """

    def __init__(self, backend: HaloModelBackend, gas: GasProfile,
                 magnetic: Optional[MagneticField] = None, *,
                 dark_comp: str = 'axion', nu_GHz: float = 353.0):
        if dark_comp not in ('axion', 'dark_photon'):
            raise ValueError(f"dark_comp must be 'axion' or 'dark_photon' (got {dark_comp!r})")
        if dark_comp == 'axion' and magnetic is None:
            raise ValueError("axion screening requires a MagneticField instance")
        self.backend = backend
        self.gas = gas
        self.magnetic = magnetic
        self.dark_comp = dark_comp
        self.nu_GHz = nu_GHz
        self.omega = 2.0 * np.pi * nu_GHz * HBAR_EVS           # angular freq in GeV (eV=1)

        # Caches
        self._I_theta_spl_list = None

    # ------------------------------------------------------------------
    # Resonance condition
    # ------------------------------------------------------------------
    @staticmethod
    def ne_resonance(m_dark_eV: float) -> float:
        """Electron density (1/Mpc^3) at which m_γ(n_e) = m_dark."""
        m_GeV = m_dark_eV / 1e9
        return m_GeV ** 2 * NE_MPC3_PER_GEV2

    def dmgamma_sq_dr(self, r, M200c, R200c, c200c, z, rho_critz):
        """Derivative of plasma mass² with respect to r in GeV^3 (eV=1)."""
        return self.gas.dne_dr(r, M200c, R200c, c200c, z, rho_critz) * MPC_INV4_TO_GEV3

    def compute_Rres_grid(self, m_dark_eV: float) -> np.ndarray:
        """Return R_res in Mpc, shape (Nz, NM).  NaN where no resonance in halo."""
        be = self.backend
        ne_res = self.ne_resonance(m_dark_eV)
        R_res = np.empty((be.Nz, be.NM))
        for i in range(be.Nz):
            z = be.z_bins[i]; rc = be.rhocritz_bins[i]
            for j in range(be.NM):
                R_res[i, j] = find_Rres(
                    self.gas, ne_res,
                    be.M200c_bins[i, j], be.R200c_bins[i, j], be.c200c_bins[i, j],
                    z, rc,
                )
        return R_res

    # ------------------------------------------------------------------
    # Conversion probability
    # ------------------------------------------------------------------
    def compute_P_conversion(self, coupling: float, m_dark_eV: float,
                             R_res, B_res, M200c, R200c, c200c, z, rho_critz):
        """Per-halo (axion↔γ or A'↔γ) resonant conversion probability.

        Axion: P = π · ω · (1+z) · g² · B² / |dm_γ²/dr|_res.

        Dark photon: TODO.  The Landau–Zener form is
        ``P_{γ→A'} = π · m_{A'}^2 · ε^2 · (1+z) / |dm_γ²/dr|_res``
        at a longitudinal resonance, but factors of 2, 1/3 (orientation
        averaging vs not), and ω-dependence differ between references.  We
        refuse to guess until the convention is pinned down.
        """
        if np.isnan(R_res):
            return 0.0

        dmg_sq_dr = self.dmgamma_sq_dr(R_res, M200c, R200c, c200c, z, rho_critz)
        if dmg_sq_dr == 0.0:
            return 0.0

        if self.dark_comp == 'axion':
            g_a = coupling
            return (np.pi * self.omega * (1 + z) * g_a ** 2 * B_res ** 2
                    / np.abs(dmg_sq_dr))

        # dark photon
        raise NotImplementedError(
            "Dark-photon conversion probability not implemented.  Verify the "
            "Landau–Zener formula (factors of orientation averaging + ω "
            "dependence) before enabling.  See screening.compute_P_conversion "
            "docstring for the canonical scaffold."
        )

    # ------------------------------------------------------------------
    # Angular integral I_ℓ(θ_max)
    # ------------------------------------------------------------------
    def _build_I_theta_interpolators(self, Ntheta_per_ell=None, v_min=5e-6):
        """Pre-tabulate I_ℓ(θ_max) for every ell in backend.ell_range."""
        be = self.backend

        # Fast route: Pell interpolator shared across θ_max calls
        theta_interp = np.append([0], np.logspace(-5, np.log10(np.pi), 10000))
        cos_theta    = np.cos(theta_interp)
        Pell_splines = [CubicSpline(theta_interp, eval_legendre(ell, cos_theta))
                        for ell in be.ell_range]

        I_splines = []
        for i, ell in enumerate(be.ell_range):
            Ntheta = Ntheta_per_ell if Ntheta_per_ell is not None else (10000 if ell >= 1000 else 2000)
            theta_max_grid = np.logspace(-4, np.log10(np.pi), Ntheta)
            I_vals = np.asarray([
                _I_theta_max(th, Pell_splines[i], v_min=v_min)
                for th in theta_max_grid
            ])
            I_splines.append(CubicSpline(theta_max_grid, I_vals))
        self._I_theta_spl_list = I_splines

    # ------------------------------------------------------------------
    # τ_ℓ matrix
    # ------------------------------------------------------------------
    def compute_tau_ell_mat(self, coupling: float, m_dark_eV: float,
                            *, progress=True) -> np.ndarray:
        """Return τ_ℓ on the (Nz, NM, Nell) grid for the given (g_a, m_a)."""
        be = self.backend

        if self._I_theta_spl_list is None:
            self._build_I_theta_interpolators()

        # 1. Resonance radii and B at resonance
        R_res = self.compute_Rres_grid(m_dark_eV)
        B_res = np.zeros_like(R_res)
        if self.dark_comp == 'axion':
            for i, z in enumerate(be.z_bins):
                for j in range(be.NM):
                    if np.isnan(R_res[i, j]):
                        continue
                    B_res[i, j] = self.magnetic(z, R_res[i, j],
                                                be.M200c_bins[i, j],
                                                be.R200c_bins[i, j],
                                                be.rhocritz_bins[i])

        # 2. Number of resonances within virial radius
        N_res_mat = np.where(np.isnan(R_res), 0.0, N_res(be.Rvir_bins, R_res))

        # 3. Per-halo conversion probability
        Pdark = np.zeros_like(R_res)
        for i in range(be.Nz):
            z = be.z_bins[i]; rc = be.rhocritz_bins[i]
            for j in range(be.NM):
                if np.isnan(R_res[i, j]) or N_res_mat[i, j] == 0.0:
                    continue
                Pdark[i, j] = self.compute_P_conversion(
                    coupling, m_dark_eV, R_res[i, j], B_res[i, j],
                    be.M200c_bins[i, j], be.R200c_bins[i, j],
                    be.c200c_bins[i, j], z, rc,
                )

        # 4. Angular integral u_ℓ0(θ_max) per cell
        tau = np.zeros((be.Nz, be.NM, be.N_ell))
        it = tqdm(range(be.Nz), disable=not progress)
        for i in it:
            z = be.z_bins[i]; chi = be.chi_bins[i]
            for j in range(be.NM):
                if Pdark[i, j] == 0.0:
                    continue
                theta_max = R_res[i, j] * (1 + z) / chi
                u_ell0 = np.asarray([spl(theta_max) for spl in self._I_theta_spl_list])
                tau[i, j] = (1.0 / 3.0) * N_res_mat[i, j] * Pdark[i, j] * u_ell0

        # Expose intermediates for diagnostics
        self.R_res_mat = R_res
        self.B_res_mat = B_res
        self.P_conv_mat = Pdark
        self.N_res_mat = N_res_mat

        return tau


# ---------------------------------------------------------------------------
# Standalone angular integral helper
# ---------------------------------------------------------------------------

def _I_theta_max(theta_max, Pell_spline, *, Ntheta=80, v_min=5e-6):
    """∫_0^{θ_max} P_ℓ(cos θ) sin θ (1−(θ/θ_max)²)^(−1/2) dθ · 2π."""
    theta = theta_max * np.append(np.flip(1. - np.geomspace(v_min, 1., Ntheta)), [1])
    sin_t = np.sin(theta)
    Pell  = Pell_spline(theta)
    kernel = (1.0 - (theta / theta_max) ** 2) ** -0.5
    integrand = Pell * sin_t * kernel
    integrand[theta == theta_max] = 0.0
    return 2 * np.pi * simps(integrand, theta)
