"""
HOD models, tracer kernels, and the generic Limber Cl integrator.

Design
------
Each :class:`Kernel` exposes two quantities evaluated on the backend's
(Nz, NM, Nell) grid:

* ``u_1h[z, M, ell]``  — per-halo projected field amplitude that enters the
  1-halo integrand as ``∫dM n(M) K_a·K_b``.
* ``b_proj[z, ell]``   — halo-bias-weighted projection
  ``∫dM n(M) b(M) K(z, M, ell)`` that enters the 2-halo integrand.

Any (kernel_a, kernel_b) pair is combined by :func:`limber_Cl` with the
standard ``∫dz χ²/H`` volume element.  For the screening kernel an extra
``(1−e^{−x})/x · T_CMB`` factor is applied post-hoc so τ_ℓ stays unit-clean.
"""
from typing import Optional, Sequence

import numpy as np
import astropy.constants as const
import astropy.units as u
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from scipy.special import erf, sici

from .backend import HaloModelBackend
from .screening import ScreeningModel


# ---------------------------------------------------------------------------
# HOD models
# ---------------------------------------------------------------------------

class HODModel:
    """Standard HOD as used in Kusiak+22, Kusiak+23, and class_sz.

    Occupation:
        N_c(M) = ½[1 + erf((log₁₀M − log₁₀M_min)/σ_log10M)]
        N_s(M) = N_c(M) · ((M − M_0) / M'_1)^α_s        for M > M_0  else 0

    Galaxy kernel (Kusiak+22 Eq. 11):
        u_ℓ^g(M, z) = (H(z)/χ²(z)) · dN/dz · n̄_g⁻¹ · [N_c + N_s·u_ℓ^NFW(M, z)]

    1-halo galaxy auto moment (Kusiak+22 Eq. 16, identical in K23):
        ⟨|u_ℓ^g|²⟩ = (H(z)/χ²(z))² · (dN/dz)² · n̄_g⁻² · [2 N_s u + N_s² u²]

    Following the K22/K23 fiducial choice we set ``M_0 = 0`` by default;
    pass ``params['log10_m0'] = log₁₀(M_0/M⊙)`` to override.

    The HOD parameters can be calibrated either against virial mass or M₂₀₀c.
    Set ``mass_def='200c'`` (default) or ``'vir'`` accordingly — the integration
    measure ``dM_vir·n(M_vir)`` is unchanged; only the mass argument fed to the
    HOD step changes (and so N_c, N_s become z-dependent for ``'200c'``).
    """

    def __init__(self, backend: HaloModelBackend, params: dict, dN_dz_interp,
                 *, mass_def: str = '200c'):
        if mass_def not in ('vir', '200c'):
            raise ValueError(f"mass_def must be 'vir' or '200c' (got {mass_def!r})")
        self.backend = backend
        self.params = params
        self.dN_dz_interp = dN_dz_interp
        self.mass_def = mass_def
        self._build()

    def _build(self):
        be = self.backend

        # Mass argument fed to the HOD step.  Always shape (Nz, NM) so Nc/Ns
        # broadcast cleanly with n_halo_bins.
        if self.mass_def == 'vir':
            M_HOD = np.tile(be.Mvir_bins, (be.Nz, 1))
        else:
            M_HOD = be.M200c_bins                                          # (Nz, NM)

        self.Nc   = self._Nc(M_HOD)                                       # (Nz, NM)
        self.Ns   = self._Ns(M_HOD)                                       # (Nz, NM)
        self.nbar = simps((self.Nc + self.Ns) * be.n_halo_bins,
                          be.Mvir_bins, axis=-1)                          # (Nz,)

        # NFW projection kernel u_ell^NFW(z, M)
        self.u_NFW = self._u_NFW()                                        # (Nell, Nz, NM)

        Ns_mat = np.broadcast_to(self.Ns, (be.N_ell, be.Nz, be.NM))
        Nc_mat = np.broadcast_to(self.Nc, (be.N_ell, be.Nz, be.NM))

        # 1-halo galaxy auto moment ⟨|u_ℓ^g|²⟩ ∝ 2·N_s·u + N_s²·u²
        # (Kusiak+22 Eq. 16; identical in Kusiak+23 and in class_sz).
        self.u_eff_sq = 2 * Ns_mat * self.u_NFW + Ns_mat ** 2 * self.u_NFW ** 2

        # Linear HOD kernel for cross spectra and 2-halo (Kusiak+22 Eq. 11).
        self.u_eff = Nc_mat + Ns_mat * self.u_NFW                         # (Nell, Nz, NM)

        # Galaxy kernel z-dependence:  u_ell^g = H/χ² · dN/dz · nbar^{-1} · u_eff
        self._u_ell_g_zprefactor = (
            be.Hz_bins / be.chi_bins ** 2 * self.dN_dz_interp(be.z_bins) / self.nbar
        )  # (Nz,)

    # --- occupation (shared across conventions) ---------------------------
    def _Nc(self, M):
        p = self.params
        arg = (np.log10(M) - p['log10_m_min']) / p['sigma_log_m']
        return 0.5 + 0.5 * erf(arg)

    def _Ns(self, M):
        p = self.params
        M0 = 10 ** p['log10_m0'] if 'log10_m0' in p else 0.0
        M1p = 10 ** p['log10_m_star']     # historical key name; this is M'_1
        Nc = self._Nc(M)
        # Avoid negative bases: only halos above M_0 host satellites.
        Ns = np.zeros_like(Nc)
        mask = M > M0
        Ns[mask] = (Nc[mask]
                    * ((M[mask] - M0) / M1p) ** p['alpha_s'])
        return Ns

    # --- NFW projected profile ------------------------------------------
    def _u_NFW(self):
        be = self.backend
        lam = self.params['lambda']
        c_mat = np.tile(be.c200c_bins, (be.N_ell, 1, 1))
        R200_mat = np.tile(be.R200c_bins, (be.N_ell, 1, 1))
        z_mat = np.tile(be.z_bins[None, :, None], (be.N_ell, 1, be.NM))
        k_lim = be.k_lim_mat_ell_z_m

        x = lam * c_mat
        f_NFW = (np.log(1 + x) - x / (1 + x)) ** -1
        q = (k_lim * R200_mat / c_mat) * (1 + z_mat)
        q_bar = (1 + x) * q
        si_q, ci_q = sici(q)
        si_qb, ci_qb = sici(q_bar)
        return f_NFW * (
            np.cos(q) * (ci_qb - ci_q)
            + np.sin(q) * (si_qb - si_q)
            - np.sin(q_bar - q) / q_bar
        )

    # --- assembled projected galaxy field per halo ----------------------
    def u_ell_g(self):
        """u_ell^g on (Nell, Nz, NM) — exposes the paper's full S10 expression."""
        return self._u_ell_g_zprefactor[None, :, None] * self.u_eff




# ---------------------------------------------------------------------------
# Tracer kernels
# ---------------------------------------------------------------------------

class Kernel:
    """Base: exposes (u_1h, b_proj) on the backend grid."""

    backend: HaloModelBackend
    z_prefactor: np.ndarray  # (Nz,) — multiplied into the z-integrand once in limber_Cl

    def u_1h(self) -> np.ndarray:
        raise NotImplementedError

    def b_proj(self) -> np.ndarray:
        raise NotImplementedError

    def post_Cl(self, Cl: np.ndarray) -> np.ndarray:
        """Optional unit-conversion / post-multiplication applied once at the end."""
        return Cl


class GalaxyKernel(Kernel):
    """Projected galaxy-number-density tracer."""
    def __init__(self, hod: HODModel):
        self.hod = hod
        self.backend = hod.backend
        self.z_prefactor = hod._u_ell_g_zprefactor.copy()

    def u_1h(self):
        # Linear in HOD moment — caller composes with partner via axis=-1 (M) integral
        return self.hod.u_eff                          # (Nell, Nz, NM)

    def b_proj(self):
        be = self.backend
        # (∫dM n·b·u_eff) · z_prefactor
        inner = simps(be.n_halo_bins[None, :, :] * be.b1_bins[None, :, :] * self.hod.u_eff,
                      be.Mvir_bins, axis=-1)           # (Nell, Nz)
        return inner * self.z_prefactor[None, :]


class ScreeningKernel(Kernel):
    """Anisotropic screening tracer.  τ_ℓ built from a ScreeningModel + coupling."""
    def __init__(self, screening: ScreeningModel, coupling: float, m_dark_eV: float,
                 *, progress=True):
        self.screening = screening
        self.backend = screening.backend
        self.coupling = coupling
        self.m_dark_eV = m_dark_eV
        self.tau = screening.compute_tau_ell_mat(coupling, m_dark_eV, progress=progress)
        # Shape convention: store (Nell, Nz, NM) to match other kernels.
        self.tau_ell_z_m = np.transpose(self.tau, (2, 0, 1))
        # Screening contributes a flat z-prefactor; the tracer absorbs the τ→T
        # unit-conversion in post_Cl instead.
        self.z_prefactor = np.ones(self.backend.Nz)

    def u_1h(self):
        return self.tau_ell_z_m                        # (Nell, Nz, NM)

    def b_proj(self):
        be = self.backend
        return simps(be.n_halo_bins[None, :, :] * be.b1_bins[None, :, :] * self.tau_ell_z_m,
                     be.Mvir_bins, axis=-1)            # (Nell, Nz)

    def post_Cl(self, Cl):
        # Convert τ_ℓ² → μK² or τ_ℓ → μK as appropriate; handled in limber_Cl
        # for symmetric (a=b) and cross combinations.  Base implementation is
        # a no-op; the tau→T factor is applied by limber_Cl because it needs
        # to know how many screening kernels are in the pair.
        return Cl


class KappaKernel(Kernel):
    """CMB-lensing convergence kernel.  Scaffold only — halo-model P_mκ is TODO."""
    def __init__(self, backend: HaloModelBackend, z_star: float = 1089.0):
        self.backend = backend
        self.z_star = z_star
        chi_star = backend._hmvec.comoving_radial_distance(np.array([z_star]))[0]
        z, chi = backend.z_bins, backend.chi_bins
        H0_inv_Mpc = backend.cosmo.h / 2997.92458                          # 1/Mpc
        # W_κ(z) = (3/2) Ω_m H0² (1+z) χ (χ* − χ)/χ*    (c = 1, Mpc units)
        self.W_kappa = (1.5 * backend.cosmo.Omega_m * H0_inv_Mpc ** 2
                        * (1 + z) * chi * (chi_star - chi) / chi_star)
        # Kernel placeholder: full halo-model Cl_Tκ requires κ_per_halo profile,
        # implemented on the backlog.
        self.z_prefactor = self.W_kappa

    def u_1h(self):
        raise NotImplementedError(
            "κ × screening 1-halo requires κ_per_halo(ell, z, M) — implement via "
            "(M / ρ̄_m) · u_NFW_ell before enabling."
        )

    def b_proj(self):
        raise NotImplementedError(
            "κ × screening 2-halo requires a (possibly consistency-relation) "
            "matter bias at the κ side."
        )


# ---------------------------------------------------------------------------
# Generic Limber integrator
# ---------------------------------------------------------------------------

def limber_Cl(kernel_a: Kernel, kernel_b: Kernel, *,
              include_1halo=True, include_2halo=True):
    """Generic 1h+2h Cl for a pair of tracer kernels sharing a backend.

    Returns ``(Cl_1h, Cl_2h)`` with shape ``(Nell,)``.
    """
    be = kernel_a.backend
    assert kernel_b.backend is be, "kernels must share a backend"

    ell = be.ell_range.size
    volume = be.chi_bins ** 2 / be.Hz_bins              # (Nz,)

    # 1-halo: ∫dz [χ²/H · u_1h product inside ∫dM n] · z_prefactors
    Cl_1h = np.zeros(ell)
    if include_1halo:
        u_a = kernel_a.u_1h(); u_b = kernel_b.u_1h()      # (Nell, Nz, NM)
        inner = simps(be.n_halo_bins[None, :, :] * u_a * u_b, be.Mvir_bins, axis=-1)  # (Nell, Nz)
        integrand = inner * (kernel_a.z_prefactor * kernel_b.z_prefactor
                             * volume)[None, :]
        Cl_1h = simps(integrand, be.z_bins, axis=-1)

    # 2-halo: ∫dz [χ²/H · b_proj_a · b_proj_b · P_lin]
    Cl_2h = np.zeros(ell)
    if include_2halo:
        ba = kernel_a.b_proj(); bb = kernel_b.b_proj()   # (Nell, Nz) each
        integrand = (ba * bb * np.diagonal(be.P_lin_lim_mat, axis1=1, axis2=2)
                     * volume[None, :])
        Cl_2h = simps(integrand, be.z_bins, axis=-1)

    return kernel_b.post_Cl(kernel_a.post_Cl(Cl_1h)), kernel_b.post_Cl(kernel_a.post_Cl(Cl_2h))


# ---------------------------------------------------------------------------
# tau → T conversion (applied explicitly by named wrappers)
# ---------------------------------------------------------------------------

def tau_to_T_factor(nu_GHz: float, T_CMB_K: float) -> float:
    x = ((const.h * nu_GHz * u.GHz) / (const.k_B * T_CMB_K * u.K)).to(u.dimensionless_unscaled).value
    return (1 - np.exp(-x)) / x * T_CMB_K                # K


# ---------------------------------------------------------------------------
# Named convenience wrappers
# ---------------------------------------------------------------------------

def Cl_Tg(screening_kernel: ScreeningKernel, galaxy_kernel: GalaxyKernel):
    """Cross-spectrum of screening T × galaxies, in μK (1h, 2h tuple)."""
    Cl_1h, Cl_2h = limber_Cl(screening_kernel, galaxy_kernel)
    fac = tau_to_T_factor(screening_kernel.screening.nu_GHz,
                          screening_kernel.screening.backend.cosmo.T_CMB)
    # (K) × dimensionless = K; convert to μK
    conv = (fac * u.K).to(u.microKelvin).value
    return Cl_1h * conv, Cl_2h * conv


def Cl_TT(screening_kernel: ScreeningKernel):
    """Auto-spectrum of screening T, in μK² (1h, 2h tuple)."""
    Cl_1h, Cl_2h = limber_Cl(screening_kernel, screening_kernel)
    fac = tau_to_T_factor(screening_kernel.screening.nu_GHz,
                          screening_kernel.screening.backend.cosmo.T_CMB)
    conv = (fac ** 2 * u.K ** 2).to(u.microKelvin ** 2).value
    return Cl_1h * conv, Cl_2h * conv


def Cl_gg(galaxy_kernel: GalaxyKernel):
    """Auto-spectrum of galaxies (dimensionless).  Uses HOD ⟨N(N−1)⟩ convention."""
    be = galaxy_kernel.backend
    # 1h uses u_eff_sq instead of u_eff · u_eff — convention-aware.
    hod = galaxy_kernel.hod
    inner = simps(be.n_halo_bins[None, :, :] * hod.u_eff_sq, be.Mvir_bins, axis=-1)
    volume = be.chi_bins ** 2 / be.Hz_bins
    z_pref = galaxy_kernel.z_prefactor
    Cl_1h = simps(inner * (z_pref ** 2 * volume)[None, :], be.z_bins, axis=-1)

    # 2h identical to generic
    _, Cl_2h = limber_Cl(galaxy_kernel, galaxy_kernel, include_1halo=False)
    return Cl_1h, Cl_2h
