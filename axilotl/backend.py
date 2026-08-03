"""
Halo-model backend.  Wraps hmvec, precomputes everything that depends only on
cosmology + (z, M, k, ell) binning.  Consumed by profile / screening / spectra
modules downstream.

Units throughout:
  * masses   Msun
  * lengths  Mpc (comoving)
  * k        1/Mpc
  * densities Msun/Mpc^3
  * H(z)     1/Mpc  (i.e. c = 1)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import hmvec


@dataclass
class Cosmology:
    """Minimal cosmology container; forwarded to hmvec for background quantities."""
    H0: float
    ombh2: float
    omch2: float
    As: float = 2.100583e-09
    ns: float = 0.9661
    tau: float = 0.0543
    T_CMB: float = 2.7255  # K

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def Omega_b(self) -> float:
        return self.ombh2 / self.h ** 2

    @property
    def Omega_cdm(self) -> float:
        return self.omch2 / self.h ** 2

    @property
    def Omega_m(self) -> float:
        return self.Omega_b + self.Omega_cdm

    @property
    def fb(self) -> float:
        """Baryon fraction Ω_b / Ω_m — the physically correct Battaglia normalisation."""
        return self.Omega_b / self.Omega_m

    def to_hmvec_dict(self) -> Dict[str, float]:
        return {'H0': self.H0, 'ombh2': self.ombh2, 'omch2': self.omch2,
                'As': self.As, 'ns': self.ns, 'tau': self.tau}


def _duffy08_concentration(M_Msun, z, h, mdef='200c'):
    """Duffy+08 concentration–mass relation.  M in Msun; h is dimensionless."""
    if mdef == '200c':
        A, B, C = 5.71, -0.084, -0.47
    elif mdef == 'vir':
        A, B, C = 7.85, -0.081, -0.71
    else:
        raise ValueError(f"Unknown mass definition: {mdef!r}")
    return A * (M_Msun / (2e12 / h)) ** B * (1.0 + z) ** C


class HaloModelBackend:
    """Holds precomputed halo / cosmology matrices for Limber integrals.

    All arrays with shape (Nz, NM) are indexed [z, M]; arrays of shape
    (Nell, Nz, NM) are indexed [ell, z, M].
    """

    def __init__(self, cosmo: Cosmology, ell_range, *,
                 z_min=0.005, z_max=1.9, Nz=50,
                 Mvir_min=1e9, Mvir_max=5e15, NM=100,
                 kmin=1e-4, kmax=1e3, Nk=1000,
                 hmf_def='tinker', use_eh_Pk=True,
                 c200c_from_duffy=True, verbose=False):

        self.cosmo = cosmo
        self.verbose = verbose

        self.ell_range = np.asarray(ell_range)
        self.N_ell = self.ell_range.size

        self.z_bins = np.linspace(z_min, z_max, Nz)
        self.Nz = Nz

        self.Mvir_bins = np.geomspace(Mvir_min, Mvir_max, NM)
        self.NM = NM

        self.k_bins = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

        # hmvec does the heavy lifting for background + halo model quantities
        self._hmvec = hmvec.HaloModel(self.z_bins, self.k_bins,
                                      ms=self.Mvir_bins,
                                      mass_function=hmf_def, mdef='vir',
                                      params=cosmo.to_hmvec_dict())

        # Background
        self.chi_bins       = self._hmvec.comoving_radial_distance(self.z_bins)  # Mpc
        self.Hz_bins        = self._hmvec.h_of_z(self.z_bins)                    # 1/Mpc
        self.rhocritz_bins  = self._hmvec.rho_critical_z(self.z_bins)            # Msun/Mpc^3

        # Halo properties
        self.b1_bins        = self._hmvec.get_bh()                  # (Nz, NM)
        self.Rvir_bins      = np.asarray([self._hmvec.rvir(self.Mvir_bins, z) for z in self.z_bins])
        self.delta_vir_bins = self._hmvec.deltav(self.z_bins)       # (Nz,)
        self.cvir_bins      = self._hmvec.concentration()           # (Nz, NM)
        self.n_halo_bins    = self._hmvec.get_nzm()                 # (Nz, NM)

        self.M200c_bins = hmvec.mdelta_from_mdelta(
            self.Mvir_bins, self.cvir_bins,
            self.delta_vir_bins * self.rhocritz_bins,
            200 * self.rhocritz_bins,
        )
        self.R200c_bins = hmvec.R_from_M(
            self.M200c_bins, self.rhocritz_bins[:, None], delta=200,
        )

        # Concentration at the 200c definition — needed when we plug into
        # Duffy-style fits that assume M200c input.  Original code exposed
        # `c200c_new`; we make it the default and keep the legacy switch.
        z_mat = np.tile(self.z_bins[:, None], (1, NM))
        if c200c_from_duffy:
            self.c200c_bins = _duffy08_concentration(
                self.M200c_bins, z_mat, cosmo.h, '200c',
            )
        else:
            self.c200c_bins = self.cvir_bins

        # Limber P_lin on the diagonal k = (ell + 1/2) / chi(z)
        self._use_eh_Pk = use_eh_Pk
        self.P_lin_lim_mat = self._build_Plin_limber()              # (Nell, Nz, Nz) — diagonal

        # Useful broadcast helpers
        self.z_mat_z_m    = z_mat                                   # (Nz, NM)
        self.rhocritz_z_m = np.tile(self.rhocritz_bins[:, None], (1, NM))

    def _build_Plin_limber(self):
        """Precompute P_lin at k_lim = (ell+0.5)/chi(z) for every ell, z."""
        Plin_fn = (self._hmvec.P_lin_approx if self._use_eh_Pk
                   else self._hmvec.P_lin)
        out = np.empty((self.N_ell, self.Nz, self.Nz))
        for i, ell in enumerate(self.ell_range):
            k_lim = (ell + 0.5) / self.chi_bins          # (Nz,)
            out[i] = np.diag(Plin_fn(k_lim, self.z_bins))
        return out

    # --- Convenience matrix views ------------------------------------------
    @property
    def ell_mat_ell_z_m(self):
        return np.tile(self.ell_range[:, None, None], (1, self.Nz, self.NM))

    @property
    def chi_mat_ell_z_m(self):
        return np.tile(self.chi_bins[None, :, None], (self.N_ell, 1, self.NM))

    @property
    def z_mat_ell_z_m(self):
        return np.tile(self.z_bins[None, :, None], (self.N_ell, 1, self.NM))

    @property
    def M200c_mat_ell_z_m(self):
        return np.tile(self.M200c_bins, (self.N_ell, 1, 1))

    @property
    def R200c_mat_ell_z_m(self):
        return np.tile(self.R200c_bins, (self.N_ell, 1, 1))

    @property
    def c200c_mat_ell_z_m(self):
        return np.tile(self.c200c_bins, (self.N_ell, 1, 1))

    @property
    def k_lim_mat_ell_z_m(self):
        """k = (ell + 1/2) / chi(z) — matches P_lin_lim_mat."""
        return (self.ell_mat_ell_z_m + 0.5) / self.chi_mat_ell_z_m
