"""
Compute axion-photon Cl_Tg theory templates with axilotl, reproducing the
fiducial unWISE-blue x Planck/ACT configuration used in 2409.10514:

  - HOD: Kusiak+22 (K22) parameters for unWISE blue
  - Gas profile: Battaglia+12
  - Magnetic field: IllustrisTNG-measured B(r) (4 z-bins x 8 mass-bins,
    UnivariateSpline-interpolated, floored to 0 outside [0.1, 1.1] R200c;
    see axilotl.tng_bfield, data/TNG_profiles/)
  - nu_ref = 353 GHz, Planck 2018 cosmology; z/M/k grids: z in [0.005, 1.9]
    (50 bins), M in [1e9, 5e15] Msun (100 bins), k in [1e-4, 1e3] Mpc^-1
    (1000 bins)
  - Resonance-radius finding uses axilotl's bracketed brentq root-finder
    (see tests/test_axilotl.py::TestResonance for validation against an
    analytic NFW case).

Run:  python examples/compute_templates.py
Output: templates/Cl_Tg_unWISEblue_K22_nu353_Battaglia_TNG_4zbin_8Mbin.npz
          (ma_list, ell, Cl_Tg [n_ma, n_ell], g_a_ref)
"""
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from axilotl import (
    Cosmology, HaloModelBackend, BattagliaGas,
    ScreeningModel, HODModel, GalaxyKernel, ScreeningKernel, Cl_Tg,
    build_tng_bfield_callable,
)

OUT_DIR = _ROOT + '/templates/'
DNDZ_PATH = _ROOT + '/data/unWISE_dN_dz/dndz_blue.txt'
TNG_DIR = _ROOT + '/data/TNG_profiles/'

NU_GHZ = 353.0
GA_REF = 1e-11   # reference coupling (GeV^-1) the saved Cl_Tg scales as g_a^2
MA_LIST = np.logspace(-13, -11, 70)
ELL_RANGE = np.unique(np.logspace(0, np.log10(5000), 300).astype(int))
ELL_OUT = np.arange(1, 5001)

Z_MIN, Z_MAX, NZ = 0.005, 1.9, 50
M_MIN, M_MAX, NM = 1e9, 5e15, 100
K_MIN, K_MAX, NK = 1e-4, 1e3, 1000

# Planck 2018 cosmology
COSMO_PARAMS = dict(H0=100 * 0.6732, ombh2=0.0224, omch2=0.1201,
                    As=2.100583e-09, ns=0.9661, tau=0.0543)

# HOD K22 params for unWISE blue (Kusiak+22 best-fit, unWISE_dN_dz/dndz_blue)
H = 0.6732
HOD_BLUE_K22 = dict(
    alpha_s=1.3039425,
    sigma_log_m=0.68660116,
    log10_m_min=np.log10(10 ** 11.795964 / H),
    log10_m_star=np.log10(10 ** 12.701308 / H),
)
HOD_BLUE_K22['lambda'] = 1.0868995   # reserved word -> can't set via kwarg above


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    dndz = np.loadtxt(DNDZ_PATH)
    dN_dz_interp = CubicSpline(dndz[:, 0], dndz[:, 1])

    print('Building magnetic field profile from IllustrisTNG...')
    Bfield = build_tng_bfield_callable(data_dir=TNG_DIR)

    print('Building halo-model backend...')
    cosmo = Cosmology(H0=COSMO_PARAMS['H0'], ombh2=COSMO_PARAMS['ombh2'],
                      omch2=COSMO_PARAMS['omch2'], As=COSMO_PARAMS['As'],
                      ns=COSMO_PARAMS['ns'], tau=COSMO_PARAMS['tau'])
    backend = HaloModelBackend(
        cosmo, ELL_RANGE,
        z_min=Z_MIN, z_max=Z_MAX, Nz=NZ,
        Mvir_min=M_MIN, Mvir_max=M_MAX, NM=NM,
        kmin=K_MIN, kmax=K_MAX, Nk=NK,
        verbose=True,
    )
    gas = BattagliaGas(fb=cosmo.fb)
    screening = ScreeningModel(backend, gas, Bfield, dark_comp='axion', nu_GHz=NU_GHZ)
    hod = HODModel(backend, HOD_BLUE_K22, dN_dz_interp, mass_def='200c')
    galaxy = GalaxyKernel(hod)

    Cl_out = np.zeros((len(MA_LIST), len(ELL_OUT)))
    for i, m_a in enumerate(tqdm(MA_LIST)):
        sk = ScreeningKernel(screening, coupling=GA_REF, m_dark_eV=m_a, progress=False)
        Cl_1h, Cl_2h = Cl_Tg(sk, galaxy)
        Cl_interp = CubicSpline(backend.ell_range, Cl_1h + Cl_2h)
        Cl_out[i] = Cl_interp(ELL_OUT)

    out_path = OUT_DIR + 'Cl_Tg_unWISEblue_K22_nu353_Battaglia_TNG_4zbin_8Mbin.npz'
    np.savez(out_path, ma_list=MA_LIST, ell=ELL_OUT, Cl_Tg=Cl_out, g_a_ref=GA_REF)
    print(f'saved -> {out_path}')
