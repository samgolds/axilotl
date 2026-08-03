"""axilotl — halo-model dark-screening theory code.

Recommended top-level imports:

    from axilotl import (
        Cosmology, HaloModelBackend,
        BattagliaGas, NFWGas, ConstantB, CallableB,
        ScreeningModel,
        HODModel,
        GalaxyKernel, ScreeningKernel, KappaKernel,
        limber_Cl, Cl_Tg, Cl_TT, Cl_gg,
    )
"""
from .backend  import Cosmology, HaloModelBackend
from .profiles import (GasProfile, BattagliaGas, NFWGas,
                       MagneticField, ConstantB, CallableB,
                       M_P_MSUN, GAUSS_TO_GEV)
from .screening import ScreeningModel, find_Rres, N_res
from .spectra  import (HODModel,
                       Kernel, GalaxyKernel, ScreeningKernel, KappaKernel,
                       limber_Cl, Cl_Tg, Cl_TT, Cl_gg, tau_to_T_factor)
from .tng_bfield import build_tng_bfield_callable, build_tng_bfield_spline_matrix

__all__ = [
    'Cosmology', 'HaloModelBackend',
    'GasProfile', 'BattagliaGas', 'NFWGas',
    'MagneticField', 'ConstantB', 'CallableB',
    'M_P_MSUN', 'GAUSS_TO_GEV',
    'ScreeningModel', 'find_Rres', 'N_res',
    'HODModel',
    'Kernel', 'GalaxyKernel', 'ScreeningKernel', 'KappaKernel',
    'limber_Cl', 'Cl_Tg', 'Cl_TT', 'Cl_gg', 'tau_to_T_factor',
    'build_tng_bfield_callable', 'build_tng_bfield_spline_matrix',
]
