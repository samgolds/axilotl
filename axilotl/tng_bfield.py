"""
IllustrisTNG-measured magnetic-field profile B(r), spline-interpolated over
4 redshift bins x 8 halo-mass bins, for use as a MagneticField  in the
ScreeningModel class.

Data: B_prof_rho_mat_TNG{100,300}_1_M_min_..._M_max_..._Msun_mass_weight_
0p1_rR200c_2_11bin_dsp_10.txt, one file per (snapshot, mass bin), under
<data_dir>/TNG_profiles/<snapshot>/. Snapshots {99,67,50,40} correspond to
z = {0, 0.5, 1, 1.5}; mass bins 0-5 (log10 M/Msun in [11,14]) come from the
higher-resolution TNG100-1 run, bins 6-7 (>=14) from TNG300-1 
Note that z=1.5's two highest mass bins have no halos in snapshot 40, 
so we use snapshot 50 (z=1) instead.

These profiles are discussed in more detail in 2409.10514
"""
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

from .profiles import CallableB, GAUSS_TO_GEV

REDSHIFT_BINS = np.array([0, 0.5, 1, 1.5])
SNAPSHOT_INDS = np.array([99, 67, 50, 40])
M_BOUND_BINS = [(11.0, 11.5), (11.5, 12.0), (12.0, 12.5), (12.5, 13.0),
                (13.0, 13.5), (13.5, 14.0), (14.0, 14.5), (14.5, 15.0)]


def _default_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'TNG_profiles')


def build_tng_bfield_spline_matrix(data_dir=None):
    """Fit the UnivariateSpline matrix (Nz x NM) of log10(B) vs log10(r/R200c)
    from the raw TNG profile files."""
    data_dir = data_dir or _default_data_dir()

    spl_mat = np.empty((len(REDSHIFT_BINS), len(M_BOUND_BINS)), dtype=object)
    for i, snap in enumerate(SNAPSHOT_INDS):
        snap_dir = f'{data_dir}/{snap}/'
        for j, (m_min, m_max) in enumerate(M_BOUND_BINS):
            if j < 6:
                sim, d = 'TNG100_1', snap_dir
            else:
                sim = 'TNG300_1'
                # z=1.5, high-mass bins have no halos in snapshot 40 -> use 50
                d = f'{data_dir}/50/' if i == 3 else snap_dir
            fname = (f'B_prof_rho_mat_{sim}_M_min_{m_min:.2f}_M_max_'
                     f'{m_max:.2f}_Msun_mass_weight_0p1_rR200c_2_11bin_dsp_10.txt')
            B_mean = np.nanmean(np.loadtxt(d + fname), axis=0)

            bin_edges = np.logspace(np.log10(0.07), np.log10(2), 12)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            inds = np.where((bin_centers > 0) & (bin_centers < 2))
            spl_mat[i][j] = UnivariateSpline(
                np.log10(bin_centers)[inds], np.log10(B_mean)[inds], k=5)
    return spl_mat


def build_tng_bfield_callable(data_dir=None, bound_low=True, bound_high=True):
    """Return a callable magnetic field profile wrapping the IllustrisTNG-measured B(r) 
    interpolators so they can be used by the  ScreeningModel class."""
    spl_mat = build_tng_bfield_spline_matrix(data_dir)

    def get_B_res_TNG(z, Rres, M200c, R200c, rho_critz):
        x = Rres / R200c
        if bound_low and x < 0.1:
            return 0.0
        if bound_high and x > 1.1:
            return 0.0

        zbin_ind = np.argmin(np.abs(REDSHIFT_BINS - z))
        log10_M200c = np.log10(M200c)
        if log10_M200c < 11.0:
            return 0.0
        elif log10_M200c >= 15:
            Mbin_ind = 7
        else:
            Mbin_ind = np.where([m0 <= log10_M200c < m1
                                 for m0, m1 in M_BOUND_BINS])[0][0]

        B_Gauss = 10 ** spl_mat[zbin_ind][Mbin_ind](np.log10(x))
        return B_Gauss * GAUSS_TO_GEV

    return CallableB(get_B_res_TNG)