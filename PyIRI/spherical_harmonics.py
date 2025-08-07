#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyIRI software.

References
----------
Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
International Reference Ionosphere Modeling Implemented in Python,
Space Weather, ESS Open Archive, September 28, 2023,
doi:10.22541/essoar.169592556.61105365/v1.

Bilitza et al. (2022), The International Reference Ionosphere
model: A review and description of an ionospheric benchmark, Reviews
of Geophysics, 60.

Nava et al. (2008). A new version of the NeQuick ionosphere
electron density model. J. Atmos. Sol. Terr. Phys., 70 (15),
doi:10.1016/j.jastp.2008.01.015.

Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
in ionospheric mapping by numerical methods.

"""

import datetime as dt
import numpy as np
import scipy.special as ss
import opt_einsum as oe
import netCDF4 as nc
import os
from apexpy import Apex
import pandas as pd
from typing import Union, Tuple
import psutil
import warnings
from tqdm import tqdm

import PyIRI
import PyIRI.igrf_library as igrf
import PyIRI.main_library as main


def IRI_monthly_mean_par(year, mth, aUT, alon, alat, coeff_dir, 
                         foF2_coeff='URSI', hmF2_model='SHU2015',
                         coord='geo'):
    """Output monthly mean ionospheric parameters using spherical harmonics.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month of year.
    aUT : float, list, or array-like
        Array of universal time (UT) in hours. Must be Numpy array of any size
        [N_T].
    alon : float, list, or array-like
        Flattened array of geographic or magnetic quasi-dipole longitudes in
        degrees or magnetic local times in hours. Must be Numpy array of any
        size [N_G].
    alat : float, list, or array-like
        Flattened array of geographic or magnetic quasi-dipole latitude in
        degrees. Must be Numpy array of any size [N_G].
    coeff_dir : str
        Place where coefficients are.
    foF2_coeff : str
        Coefficients to use for foF2. Options are 'URSI' and 'CCIR'.
        (default='URSI')
    hmF2_model : str
        Model to use for hmF2. Options are 'SHU2015', 'AMTB2013', and
        'BSE1979'. (default='SHU2015')
    coord : str
        Coordinate system. Options are 'geo' for geographic and 'mlt' for
        quasi-dipole latitude and magnetic local time. (default='geo')

    Returns
    -------
    F2 : dict
        'Nm' is peak density of F2 region in m-3.
        'fo' is critical frequency of F2 region in MHz.
        'M3000' is the obliquity factor for a distance of 3,000 km.
        Defined as refracted in the ionosphere, can be received at a
        distance of 3,000 km, unitless.
        'hm' is height of the F2 peak in km.
        'B_top' is top thickness of the F2 region in km.
        'B_bot' is bottom thickness of the F2 region in km.
        Shape [N_T, N_G, 2].
    F1 : dict
        'Nm' is peak density of F1 region in m-3.
        'fo' is critical frequency of F1 region in MHz.
        'P' is the probability occurrence of F1 region, unitless.
        'hm' is height of the F1 peak in km.
        'B_bot' is bottom thickness of the F1 region in km.
        Shape [N_T, N_G, 2].
    E : dict
        'Nm' is peak density of E region in m-3.
        'fo' is critical frequency of E region in MHz.
        'hm' is height of the E peak in km.
        'B_top' is bottom thickness of the E region in km.
        'B_bot' is bottom thickness of the E region in km.
        Shape [N_T, N_G, 2].
    Es : dict
        'Nm' is peak density of Es region in m-3.
        'fo' is critical frequency of Es region in MHz.
        'hm' is height of the Es peak in km.
        'B_top' is bottom thickness of the Es region in km.
        'B_bot' is bottom thickness of the Es region in km.
        Shape [N_T, N_G, 2].
    sun : dict
        'lon' is longitude of subsolar point in degrees.
        'lat' is latitude of subsolar point in degrees.
        Shape [N_G]
    mag : dict
        'inc' is inclination of the magnetic field in degrees.
        'modip' is modified dip angle in degrees.
        'mag_dip_lat' is magnetic dip latitude in degrees.
        Shape [N_G]

    Notes
    -----
    This function returns monthly mean ionospheric parameters for min
    and max levels of solar activity that correspond to the Ionosonde
    Index IG12 of 0 and 100.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # -------------------------------------------------------------------------
    # Just to make life easier
    aUT  = to_numpy_array(aUT)
    alon = to_numpy_array(alon)
    alat = to_numpy_array(alat)
    apdtime = (pd.to_datetime(dt.datetime(year, mth, 15)) 
                                               + pd.to_timedelta(aUT, 'hours'))

    # -------------------------------------------------------------------------
    # Coordinate system conversion to geographic and mlt
    if coord == 'geo':
        aglat, aglon = alat, alon
        aqdlat, amlt = np.zeros((2, aUT.size, alat.size))
        for iUT in range(len(aUT)):
            pdtime = apdtime[iUT]
            A = Apex(pdtime)
            aqdlat[iUT, :], amlt[iUT, :] = A.convert(aglat, aglon,
                                        'geo', 'mlt', height=0.,
                                        datetime=pdtime)
    elif coord == 'mlt':
        aqdlat, amlt = alat, alon
        aglat, aglon = np.zeros((2, aUT.size, alat.size))
        for iUT in range(len(aUT)):
            pdtime = apdtime[iUT]
            A = Apex(pdtime)
            aglat[iUT, :], aglon[iUT, :] = A.convert(aqdlat, amlt,
                                        'mlt', 'geo', height=0.,
                                        datetime=pdtime)
    else:
        raise ValueError('Coordinate system not implemented yet.')

    # Set limits for solar driver based of IG12 = 0 - 100.
    aIG = np.array([0., 100.])

    # Date and time for the middle of the month (day=15) that will be used to
    # find magnetic inclination
    dtime = dt.datetime(year, mth, 15)
    decimal_year = main.decimal_year(dtime)
    # -------------------------------------------------------------------------
    # Calculating magnetic inclanation, modified dip angle, and magnetic dip
    # latitude using IGRF at 300 km of altitude
    inc = igrf.inclination(coeff_dir,
                           decimal_year,
                           alon, alat, 300.0, only_inc=True)
    modip = igrf.inc2modip(inc, alat)
    mag_dip_lat = igrf.inc2magnetic_dip_latitude(inc)

    # -------------------------------------------------------------------------
    # Extracting coefficient matrices and resonstructing parameters
    C = load_coeff_matrices(mth, coeff_dir, foF2_coeff, hmF2_model) 
    # dimensions: (n_params, n_IG12, n_DFT, n_SH)
    # n_IG12=2: IG12 0 or 100
    # n_params=4 (or 5): foF2, B0, B1, M3000F2 (+1 hmF2 optional)
    n_time = aUT.size
    n_pos = alat.size
    n_param = C.shape[0]
    n_IG12 = C.shape[1]

    n_FS_r = C.shape[2]
    n_FS_c = n_FS_r // 2 + 1
    F_FS = real_FS_func(aUT, n_FS_c)

    n_SH = C.shape[3]
    lmax = int(np.sqrt(n_SH)) - 1
    atheta = np.deg2rad(-(aqdlat - 90))
    aphi = np.deg2rad(amlt * 15)

    if coord == 'geo':
        Params = np.zeros((n_param, n_time, n_pos, n_IG12))
        array_shape = (lmax + 1, 2 * lmax + 1, n_time, n_pos)
        if n_time < n_pos:
            chunk_size, num_chunks = get_safe_chunk_plan(array_shape, axis=2)
            if num_chunks > 1:
                warnings.warn(f"Memory insufficient; chunking over time. \
                              ({num_chunks} chunks)", 
                              RuntimeWarning)
            for ichunk in range(num_chunks):
                start = ichunk * chunk_size
                end = min(start + chunk_size, n_time)
                F_SH = real_SH_func(lmax,
                    atheta[start:end, :],
                    aphi[start:end, :])
                res = oe.contract('ij,opjk,kil->oilp', F_FS[start:end, :],
                                  C, F_SH)
                Params[:, start:end, :, :] = res
        elif n_time > n_pos:
            chunk_size, num_chunks = get_safe_chunk_plan(array_shape, axis=3)
            if num_chunks > 1:
                warnings.warn(f"Memory insufficient; chunking over space. \
                              ({num_chunks} chunks)", 
                              RuntimeWarning)
            for ichunk in range(num_chunks):
                start = ichunk * chunk_size
                end = min(start + chunk_size, n_pos)
                F_SH = real_SH_func(lmax,
                    atheta[:, start:end],
                    aphi[:, start:end])
                res = oe.contract('ij,opjk,kil->oilp', F_FS, C, F_SH)
                Params[:, :, start:end, :] = res
    elif coord == 'mlt':
        F_SH = real_SH_func(lmax, atheta, aphi)
        # Params = oe.contract('ij,opjk,kl->oilp', F_FS, C, F_SH)
        Params = oe.contract('ij,opjk,kl->oilp', F_SH.T, C.transpose(0, 1, 3, 2), F_FS.T)
        # F_SH (n_SH, n_pos) (k, l) # F_SH.T (n_pos, n_SH)
        # C (2, 5, n_FS, n_SH) (o, p, j, k) # C.T (n_SH, n_FS)
        # F_FS (n_time, n_FS) (i, j) # F_FS.T (n_FS, n_time)

    foF2 = Params[0, :, :, :]
    B0 = Params[1, :, :, :]
    B1 = Params[2, :, :, :]
    M3000 = Params[3, :, :, :]
    if hmF2_model != 'BSE1979':
        hmF2 = Params[4, :, :, :]

    NmF2 = main.freq2den(foF2) * 1e11

    # --------------------------------------------------------------------------
    # Solar driven E region and locations of subsolar points
    foE, solzen, solzen_eff, slon, slat = main.gammaE(year, mth, aUT, alon,
                                                      alat, aIG)

    # --------------------------------------------------------------------------
    # Probability of F1 layer to appear based on SZA
    P_F1 = Probability_F1(year, mth, aUT, alon, alat)

    # --------------------------------------------------------------------------
    # Convert critical frequency to the electron density (m-3)
    # we don't yet have the foF1, therefore we are passing an array of zeros
    NmE = main.freq2den(foE)

    # Introduce a minimum limit for the peaks to avoid negative density
    NmE = main.limit_Nm(NmE)
    # We should not limit the F1 region because it is a function of F2

    # --------------------------------------------------------------------------
    # Find heights of the F2 and E ionospheric layers
    if hmF2_model == 'BSE1979':
        hmF2, hmE, hmEs = main.hm_IRI(M3000, foE, foF2, modip, aIG)
    else:
        hmE = 110. + np.zeros(M3000.shape)
        hmEs = 110. + np.zeros(M3000.shape)

    # --------------------------------------------------------------------------
    # Find thicknesses of the F2 and E ionospheric layers
    (B_F2_bot,
     B_F2_top,
     B_E_bot,
     B_E_top,
     B_Es_bot,
     B_Es_top) = main.thickness(foF2,
                                M3000,
                                hmF2,
                                hmE,
                                mth,
                                aIG)

    # --------------------------------------------------------------------------
    # Find height of the F1 layer based on the P and F2
    NmF1, foF1, hmF1, B_F1_bot = derive_dependent_F1_parameters(P_F1,
                                                                NmF2,
                                                                hmF2,
                                                                B_F2_bot,
                                                                hmE)
    
    # --------------------------------------------------------------------------
    # Placeholders for NmEs and foEs
    NmEs = np.zeros_like(NmF2)
    foEs = np.zeros_like(NmF2)

    # --------------------------------------------------------------------------
    # Add all parameters to dictionaries:
    F2 = {'Nm': NmF2,
          'fo': foF2,
          'M3000': M3000,
          'hm': hmF2,
          'B_top': B_F2_top,
          'B_bot': B_F2_bot}

    F1 = {'Nm': NmF1,
          'fo': foF1,
          'P': P_F1,
          'hm': hmF1,
          'B_bot': B_F1_bot}

    E = {'Nm': NmE,
         'fo': foE,
         'hm': hmE,
         'B_bot': B_E_bot,
         'B_top': B_E_top,
         'solzen': solzen,
         'solzen_eff': solzen_eff}

    Es = {'Nm': NmEs,
          'fo': foEs,
          'hm': hmEs,
          'B_bot': B_Es_bot,
          'B_top': B_Es_top}

    sun = {'lon': slon,
           'lat': slat}

    mag = {'inc': inc,
           'modip': modip,
           'mag_dip_lat': mag_dip_lat}

    return F2, F1, E, Es, sun, mag


def IRI_density_1day(year, mth, day, aUT, alon, alat, aalt, F107, 
                    coeff_dir, ccir_or_ursi=0):
    """Output ionospheric parameters for a particular day.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month of year.
    day : int
        Day of month.
    aUT : array-like
        Array of universal time (UT) in hours. Must be Numpy array of any size
        [N_T].
    alon : array-like
        Flattened array of geographic longitudes in degrees. Must be Numpy array
        of any size [N_G].
    alat : array-like
        Flattened array of geographic latitudes in degrees. Must be Numpy array
        of any size [N_G].
    aalt : array-like
        Array of altitudes in km. Must be Numpy array of any size [N_V].
    F107 : float
        User provided F10.7 solar flux index in SFU.
    coeff_dir : str
        Place where coefficients are located.
    ccir_or_ursi : int
        If 0 is given CCIR will be used for F2 critical frequency. If 1 then
        URSI coefficients. (default=0)

    Returns
    -------
    F2 : dict
        'Nm' is peak density of F2 region in m-3.
        'fo' is critical frequency of F2 region in MHz.
        'M3000' is the obliquity factor for a distance of 3,000 km.
        Defined as refracted in the ionosphere, can be received at a distance
        of 3,000 km, unitless.
        'hm' is height of the F2 peak in km.
        'B_topi is top thickness of the F2 region in km.
        'B_bot' is bottom thickness of the F2 region in km.
        Shape [N_T, N_G, 2].
    F1 : dict
        'Nm' is peak density of F1 region in m-3.
        'fo' is critical frequency of F1 region in MHz.
        'P' is the probability occurrence of F1 region, unitless.
        'hm' is height of the F1 peak in km.
        'B_bot' is bottom thickness of the F1 region in km.
        Shape [N_T, N_G, 2].
    E : dict
        'Nm' is peak density of E region in m-3.
        'fo' is critical frequency of E region in MHz.
        'hm' is height of the E peak in km.
        'B_top' is bottom thickness of the E region in km.
        'B_bot' is bottom thickness of the E region in km.
        Shape [N_T, N_G, 2].
    Es : dict
        'Nm' is peak density of Es region in m-3.
        'fo' is critical frequency of Es region in MHz.
        'hm' is height of the Es peak in km.
        'B_top' is bottom thickness of the Es region in km.
        'B_bot' is bottom thickness of the Es region in km.
        Shape [N_T, N_G, 2].
    sun : dict
        'lon' is longitude of subsolar point in degrees.
        'lat' is latitude of subsolar point in degrees.
        Shape [N_G].
    mag : dict
        'inc' is inclination of the magnetic field in degrees.
        'modip' is modified dip angle in degrees.
        'mag_dip_lat' is magnetic dip latitude in degrees.
        Shape [N_G].
    EDP : array-like
        Electron density profiles in m-3 with shape [N_T, N_V, N_G]

    Notes
    -----
    This function returns ionospheric parameters and 3-D electron density
    for a given day and provided F10.7 solar flux index.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """

    # find out what monthly means are needed first and what their weights
    # will be
    t_before, t_after, fr1, fr2 = main.day_of_the_month_corr(year, mth, day)

    F2_1, F1_1, E_1, Es_1, sun_1, mag_1 = IRI_monthly_mean_par(t_before.year,
                                                               t_before.month,
                                                               aUT,
                                                               alon,
                                                               alat,
                                                               coeff_dir,
                                                               ccir_or_ursi)
    F2_2, F1_2, E_2, Es_2, sun_2, mag_2 = IRI_monthly_mean_par(t_after.year,
                                                               t_after.month,
                                                               aUT,
                                                               alon,
                                                               alat,
                                                               coeff_dir,
                                                               ccir_or_ursi)

    F2 = main.fractional_correction_of_dictionary(fr1, fr2, F2_1, F2_2)
    F1 = main.fractional_correction_of_dictionary(fr1, fr2, F1_1, F1_2)
    E = main.fractional_correction_of_dictionary(fr1, fr2, E_1, E_2)
    Es = main.fractional_correction_of_dictionary(fr1, fr2, Es_1, Es_2)
    sun = main.fractional_correction_of_dictionary(fr1, fr2, sun_1, sun_2)
    mag = main.fractional_correction_of_dictionary(fr1, fr2, mag_1, mag_2)

    # interpolate parameters in solar activity
    F2 = main.solar_interpolation_of_dictionary(F2, F107)
    F1 = main.solar_interpolation_of_dictionary(F1, F107)
    E = main.solar_interpolation_of_dictionary(E, F107)
    Es = main.solar_interpolation_of_dictionary(Es, F107)

    # Introduce a minimum limit for the peaks to avoid negative density as a
    # result of the interpolation in case the F10.7 is high the extrapolation
    # can cause NmF2 to go negative
    F2['Nm'] = main.limit_Nm(F2['Nm'])
    E['Nm'] = main.limit_Nm(E['Nm'])
    Es['Nm'] = main.limit_Nm(Es['Nm'])
    # We should not limit the F1 region because it is a function of F2

    # Derive_dependent_F1_parameters one more time after the interpolation
    # so that the F1 location does not carry the little errors caused by the
    # interpolation
    NmF1, foF1, hmF1, B_F1_bot = derive_dependent_F1_parameters(F1['P'],
                                                                F2['Nm'],
                                                                F2['hm'],
                                                                F2['B_bot'],
                                                                E['hm'])
    # Update the F1 dictionary with the re-derived parameters
    F1['Nm'] = NmF1
    F1['hm'] = hmF1
    F1['fo'] = foF1
    F1['B_bot'] = B_F1_bot

    # Construct density
    EDP = reconstruct_density_from_parameters_1level(F2, F1, E, aalt)

    return F2, F1, E, Es, sun, mag, EDP


def reconstruct_density_from_parameters(F2, F1, E, alt):
    """Construct vertical EDP for 2 levels of solar activity.

    Parameters
    ----------
    F2 : dict
        Dictionary of parameters for F2 layer.
    F1 : dict
        Dictionary of parameters for F1 layer.
    E : dict
        Dictionary of parameters for E layer.
    alt : array-like
        1-D array of altitudes [N_V] in km.

    Returns
    -------
    x_out : array-like
        Electron density for two levels of solar activity [2, N_T, N_V, N_G]
        in m-3.

    Notes
    -----
    This function calculates 3-D density from given dictionaries of
    the parameters for 2 levels of solar activity.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    s = F2['Nm'].shape
    N_G = s[1]
    N_T = s[0]
    N_V = alt.size

    x_out = np.full((2, N_T, N_V, N_G), np.nan)

    for isolar in range(0, 2):
        x = np.full((11, N_T, N_G), np.nan)

        x[0, :, :] = F2['Nm'][:, :, isolar]
        x[1, :, :] = F1['Nm'][:, :, isolar]
        x[2, :, :] = E['Nm'][:, :, isolar]
        x[3, :, :] = F2['hm'][:, :, isolar]
        x[4, :, :] = F1['hm'][:, :, isolar]
        x[5, :, :] = E['hm'][:, :, isolar]
        x[6, :, :] = F2['B_bot'][:, :, isolar]
        x[7, :, :] = F2['B_top'][:, :, isolar]
        x[8, :, :] = F1['B_bot'][:, :, isolar]
        x[9, :, :] = E['B_bot'][:, :, isolar]
        x[10, :, :] = E['B_top'][:, :, isolar]

        EDP = EDP_builder(x, alt)
        x_out[isolar, :, :, :] = EDP

    return x_out


def reconstruct_density_from_parameters_1level(F2, F1, E, alt):
    """Construct vertical EDP for 1 level of solar activity.

    Parameters
    ----------
    F2 : dict
        Dictionary of parameters for F2 layer.
    F1 : dict
        Dictionary of parameters for F1 layer.
    E : dict
        Dictionary of parameters for E layer.
    alt : array-like
        1-D array of altitudes [N_V] in km.

    Returns
    -------
    x_out : array-like
        Electron density for two levels of solar activity [N_T, N_V, N_G]
        in m-3.

    Notes
    -----
    This function calculates 3-D density from given dictionaries of
    the parameters for 1 level of solar activity.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    s = F2['Nm'].shape

    N_T = s[0]
    N_G = s[1]

    x = np.full((11, N_T, N_G), np.nan)

    x[0, :, :] = F2['Nm'][:, :]
    x[1, :, :] = F1['Nm'][:, :]
    x[2, :, :] = E['Nm'][:, :]
    x[3, :, :] = F2['hm'][:, :]
    x[4, :, :] = F1['hm'][:, :]
    x[5, :, :] = E['hm'][:, :]
    x[6, :, :] = F2['B_bot'][:, :]
    x[7, :, :] = F2['B_top'][:, :]
    x[8, :, :] = F1['B_bot'][:, :]
    x[9, :, :] = E['B_bot'][:, :]
    x[10, :, :] = E['B_top'][:, :]

    EDP = EDP_builder(x, alt)

    return EDP


def EDP_builder(x, aalt):
    """Construct vertical EDP.

    Parameters
    ----------
    x : array-like
        Array where 1st dimention indicates the parameter (total 11
        parameters), second dimension is time, and third is horizontal grid
        [11, N_T, N_G].
    aalt : array-like
        1-D array of altitudes [N_V] in km.

    Returns
    -------
    density_out : array-like
        3-D electron density [N_T, N_V, N_G] in m-3.

    Notes
    -----
    This function builds the EDP from the provided parameters for all time
    frames, all vertical and all horizontal points.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    """
    # number of elements in time dimention
    nUT = x.shape[1]

    # number of elements in horizontal dimention of grid
    nhor = x.shape[2]

    # time and horisontal grid dimention
    ngrid = nhor * nUT

    # vertical dimention
    nalt = aalt.size

    # empty arrays
    density_out = np.zeros((nalt, ngrid))
    density_F2 = np.zeros((nalt, ngrid))
    full_F1 = np.zeros((nalt, ngrid))
    density_F1 = np.zeros((nalt, ngrid))
    density_E = np.zeros((nalt, ngrid))

    # shapes:
    # for filling with altitudes because the last dimentions should match
    # the source
    shape1 = (ngrid, nalt)

    # for filling with horizontal maps because the last dimentions should
    # match the source
    shape2 = (nalt, ngrid)

    order = 'F'
    NmF2 = np.reshape(x[0, :, :], ngrid, order=order)
    NmF1 = np.reshape(x[1, :, :], ngrid, order=order)
    NmE = np.reshape(x[2, :, :], ngrid, order=order)
    hmF2 = np.reshape(x[3, :, :], ngrid, order=order)
    hmF1 = np.reshape(x[4, :, :], ngrid, order=order)
    hmE = np.reshape(x[5, :, :], ngrid, order=order)
    B_F2_bot = np.reshape(x[6, :, :], ngrid, order=order)
    B_F2_top = np.reshape(x[7, :, :], ngrid, order=order)
    B_F1_bot = np.reshape(x[8, :, :], ngrid, order=order)
    B_E_bot = np.reshape(x[9, :, :], ngrid, order=order)
    B_E_top = np.reshape(x[10, :, :], ngrid, order=order)

    # Set to some parameters if zero or lower (just in case)
    B_F2_top[np.where(B_F2_top <= 0)] = 10.

    # Array of hmFs, importantly, with same dimensions as result, to
    # later search for regions using argwhere
    a_alt = np.full(shape1, aalt, order='F')
    a_alt = np.swapaxes(a_alt, 0, 1)

    # Fill arrays with parameters to add height dimension and populate it
    # with same values, this is important to keep all operations in matrix
    # form
    a_NmF2 = np.full(shape2, NmF2)
    a_NmF1 = np.full(shape2, NmF1)
    a_NmE = np.full(shape2, NmE)
    a_hmF2 = np.full(shape2, hmF2)
    a_hmF1 = np.full(shape2, hmF1)
    a_hmE = np.full(shape2, hmE)
    a_B_F2_bot = np.full(shape2, B_F2_bot)
    a_B_F2_top = np.full(shape2, B_F2_top)
    a_B_F1_bot = np.full(shape2, B_F1_bot)
    a_B_E_top = np.full(shape2, B_E_top)
    a_B_E_bot = np.full(shape2, B_E_bot)

    # Drop functions to reduce contributions of the layers when adding them up
    multiplier_down_F2 = drop_down(a_alt, a_hmF2, a_hmE)
    multiplier_down_F1 = drop_down(a_alt, a_hmF1, a_hmE)
    multiplier_up = drop_up(a_alt, a_hmE, a_hmF2)

    # In the where statements all 3 dimensions are needed.
    # This is the same as density[a]= density[a[0], a[1], a[2]].

    # ------F2 region------
    a = np.where(a_alt >= a_hmF2)
    density_F2[a] = main.epstein_function_top_array(4. * a_NmF2[a], a_hmF2[a],
                                                    a_B_F2_top[a], a_alt[a])
    a = np.where((a_alt < a_hmF2) & (a_alt >= a_hmE))
    density_F2[a] = (main.epstein_function_array(4. * a_NmF2[a],
                                                 a_hmF2[a],
                                                 a_B_F2_bot[a],
                                                 a_alt[a])
                     * multiplier_down_F2[a])

    # ------E region-------
    a = np.where((a_alt >= a_hmE) & (a_alt < a_hmF2))
    density_E[a] = main.epstein_function_array(4. * a_NmE[a],
                                               a_hmE[a],
                                               a_B_E_top[a],
                                               a_alt[a]) * multiplier_up[a]
    a = np.where(a_alt < a_hmE)
    density_E[a] = main.epstein_function_array(4. * a_NmE[a],
                                               a_hmE[a],
                                               a_B_E_bot[a],
                                               a_alt[a])

    # Add F2 and E layers
    density = density_F2 + density_E

    # ------F1 region------
    a = np.where((a_alt > a_hmE) & (a_alt < a_hmF1))
    full_F1[a] = main.epstein_function_array(4. * a_NmF1[a],
                                             a_hmF1[a],
                                             a_B_F1_bot[a],
                                             a_alt[a]) * multiplier_down_F1[a]
    # Find the difference between the EDP and the F1 layer and add to the EDP
    # the positive part
    density_F1 = full_F1 - density
    density_F1[density_F1 < 0] = 0.

    density = density + density_F1

    # Make 1 everything that is <= 0 (just in case)
    density[np.where(density <= 1.0)] = 1.0

    # Reshape to the [N_T, N_V, N_G]
    density_out = np.reshape(density, (nalt, nUT, nhor), order='F')
    density_out = np.swapaxes(density_out, 0, 1)

    return density_out


def run_iri_reg_grid(year, month, day, f107, hr_res=1, lat_res=1, lon_res=1,
                     alt_res=10, alt_min=0, alt_max=700, ccir_or_ursi=0):
    """Run IRI for a single day on a regular grid.

    Parameters
    ----------
    year : int
        Four digit year in C.E.
    month : int
        Integer month (range 1-12)
    day : int
        Integer day of month (range 1-31)
    f107 : int or float
        F10.7 index for the given day
    hr_res : int or float
        Time resolution in hours (default=1)
    lat_res : int or float
        Latitude resolution in degrees (default=1)
    lon_res : int or float
        Longitude resolution in degrees (default=1)
    alt_res : int or float
        Altitude resolution in km (default=10)
    alt_min : int or float
        Altitude minimum in km (default=0)
    alt_max : int or float
        Altitude maximum in km (default=700)
    ccir_or_ursi : int
        If 0 use CCIR coefficients, if 1 use URSI coefficients

    Returns
    -------
    alon : array-like
        1D longitude grid
    alat : array-like
        1D latitude grid
    alon_2d : array-like
        2D longitude grid
    alat_2d : array-like
        2D latitude grid
    aalt : array-like
        Altitude grid
    ahr : array-like
        UT grid
    f2 : array-like
        F2 peak
    f1 : array-like
        F1 peak
    epeak : array-like
        E peak
    es_peak : array-like
        Sporadic E (Es) peak
    sun : array-like
        Solar zenith angle in degrees
    mag : array-like
        Magnetic inclination in degrees
    edens_prof : array-like
        Electron density profile in per cubic m

    See Also
    --------
    create_reg_grid

    """
    # Define the grids
    alon, alat, alon_2d, alat_2d, aalt, ahr = main.create_reg_grid(
        hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
        alt_res=alt_res, alt_min=alt_min,
        alt_max=alt_max)
    # -------------------------------------------------------------------------
    # Monthly mean density for min and max of solar activity:
    # -------------------------------------------------------------------------
    # The original IRI model further interpolates between 2 levels of solar
    # activity to estimate density for a particular level of F10.7.
    # Additionally, it interpolates between 2 consecutive months to make a
    # smooth seasonal transition. Here is an example of how this interpolation
    # can be done. If you need to run IRI for a particular day, you can just
    # use this function
    f2, f1, epeak, es_peak, sun, mag, edens_prof = IRI_density_1day(
        year, month, day, ahr, alon, alat, aalt, f107, PyIRI.coeff_dir,
        ccir_or_ursi)

    return (alon, alat, alon_2d, alat_2d, aalt, ahr, f2, f1, epeak,
            es_peak, sun, mag, edens_prof)


def run_seas_iri_reg_grid(year, month, hr_res=1, lat_res=1, lon_res=1,
                          alt_res=10, alt_min=0, alt_max=700, ccir_or_ursi=0):
    """Run IRI for monthly mean parameters on a regular grid.

    Parameters
    ----------
    year : int
        Four digit year in C.E.
    month : int
        Integer month (range 1-12)
    f107 : int or float
        F10.7 index for the given day
    hr_res : int or float
        Time resolution in hours (default=1)
    lat_res : int or float
        Latitude resolution in degrees (default=1)
    lon_res : int or float
        Longitude resolution in degrees (default=1)
    alt_res : int or float
        Altitude resolution in km (default=10)
    alt_min : int or float
        Altitude minimum in km (default=0)
    alt_max : int or float
        Altitude maximum in km (default=700)
    ccir_or_ursi : int
        If 0 use CCIR coefficients, if 1 use URSI coefficients

    Returns
    -------
    alon : array-like
        1D longitude grid
    alat : array-like
        1D latitude grid
    alon_2d : array-like
        2D longitude grid
    alat_2d : array-like
        2D latitude grid
    aalt : array-like
        Altitude grid
    ahr : array-like
        UT grid
    f2 : array-like
        F2 peak
    f1 : array-like
        F1 peak
    epeak : array-like
        E peak
    es_peak : array-like
        Sporadic E (Es) peak
    sun : array-like
        Solar zenith angle in degrees
    mag : array-like
        Magnetic inclination in degrees
    edens_prof : array-like
        Electron density profile in per cubic m

    See Also
    --------
    create_reg_grid

    """
    # Define the grids
    alon, alat, alon_2d, alat_2d, aalt, ahr = main.create_reg_grid(
        hr_res=hr_res, lat_res=lat_res, lon_res=lon_res,
        alt_res=alt_res, alt_min=alt_min,
        alt_max=alt_max)

    # -------------------------------------------------------------------------
    # Monthly mean ionospheric parameters for min and max of solar activity:
    # -------------------------------------------------------------------------
    # This is how PyIRI needs to be called to find monthly mean values for all
    # ionospheric parameters, for min and max conditions of solar activity:
    # year and month should be integers, and ahr and alon, alat should be 1-D
    # NumPy arrays. alon and alat should have the same size. coeff_dir is the
    # coefficient directory. Matrix size for all the output parameters is
    # [N_T, N_G, 2], where 2 indicates min and max of the solar activity that
    # corresponds to Ionospheric Global (IG) index levels of 0 and 100.

    f2, f1, epeak, es_peak, sun, mag = IRI_monthly_mean_par(
        year, month, ahr, alon, alat, PyIRI.coeff_dir, ccir_or_ursi)

    # -------------------------------------------------------------------------
    # Monthly mean density for min and max of solar activity:
    # -------------------------------------------------------------------------
    # Construct electron density profiles for min and max levels of solar
    # activity for monthly mean parameters.  The result will have the following
    # dimensions [2, N_T, N_V, N_G]
    edens_prof = reconstruct_density_from_parameters(f2, f1, epeak, aalt)

    return (alon, alat, alon_2d, alat_2d, aalt, ahr, f2, f1, epeak, es_peak,
            sun, mag, edens_prof)


def derive_dependent_F1_parameters(P, NmF2, hmF2, B_F2_bot, hmE):
    """Combine DA with background F1 region.

    Parameters
    ----------
    P : array-like
        Probability of F1 to occurre from PyIRI.
    NmF2 : array-like
        NmF2 parameter - peak density of F2 layer.
    hmF2 : array-like
        hmF2 parameter - height of the peak of F2.
    B_F2_bot : array-like
        B_F2_bot parameter - thickness of F2.
    hmE : array-like
        hmE parameter - height of E layer.

    Returns
    -------
    NmF1 : array_like
        NmF1 parameter - peak of F1 layer.
    foF1 : array_like
        foF1 parameter - critical freqeuncy of F1 layer.
    hmF1 : array_like
        hmF1 parameter - peak height of F1 layer.
    B_F1_bot : array_like
        B_F1_bot - thickness of F1 layer.

    Notes
    -----
    This function derives F1 from F2 fields.

    """

    # Estimate the F1 layer peak height (hmF1) as 0.4 between the F2 peak
    # height (hmF2) and the E layer peak height (hmE)
    hmF1 = hmF2 - (hmF2 - hmE) * 0.4

    # Compute B_F1_bot using normalized probability P with a flexible
    # threshold.
    threshold = 0.1
    P_clipped = np.clip(P, threshold, 1)
    norm_shift = P_clipped - threshold
    max_shift = np.max(norm_shift)
    # Prevent division by zero
    norm_shifted = np.divide(norm_shift,
                             max_shift,
                             out=np.zeros_like(norm_shift),
                             where=max_shift != 0)
    # Map to [0.5, 1], then to [0, 1]
    norm_P = (np.clip(norm_shifted + 0.5, 0.5, 1) - 0.5) / 0.5
    B_F1_bot = (hmF1 - hmE) * 0.5 * norm_P

    # Find the exact NmF1 at the hmF1 using F2 bottom function with the drop
    # down function
    NmF1 = (main.epstein_function_array(4. * NmF2, hmF2, B_F2_bot, hmF1)
            * drop_down(hmF1, hmF2, hmE))

    # Convert plasma density to freqeuncy
    foF1 = main.den2freq(NmF1)

    return NmF1, foF1, hmF1, B_F1_bot


def logistic_curve(h, h0, B):
    """Logistic function centered at h0 with width parameter B.

    Parameters
    ----------
    h : float or array-like
        Input value(s), e.g., altitude in km.
    h0 : float
        Center point of the logistic curve (where it equals 0.5).
    B : float
        Width parameter controlling the steepness of the transition.

    Returns
    -------
    f : float or array
        Logistic function value(s) in the range (0, 1).
    """
    h = np.asarray(h)

    return 1. / (1. + np.exp(-(h - h0) / B))


def drop_up(h, hmE, hmF2, drop_fraction=0.2):
    """Smooth function of altitude with a sharp drop.

    Parameters
    ----------
    h : float or array-like
        Altitude in km.
    hmE : float
        Altitude where function = 1.
    hmF2 : float
        Altitude where function = 0.
    drop_fraction : float, optional
        Fraction of the distance (hmF2 - hmE) over which the drop occurs.

    Returns
    -------
    f : float or array
        Function value at altitude h.
    """
    h = np.asarray(h)
    D = hmF2 - hmE
    # if D <= 0:
    #     raise ValueError("hmF2 must be greater than hmE.")
    B = drop_fraction * D
    ht = hmF2 - B  # center of the drop, near hmF2
    sigma_h = logistic_curve(h, ht, B)
    sigma_E = logistic_curve(hmE, ht, B)
    sigma_F2 = logistic_curve(hmF2, ht, B)

    return (sigma_F2 - sigma_h) / (sigma_F2 - sigma_E)


def drop_down(h, hmF2, hmE, drop_fraction=0.1):
    """Smooth function of altitude with a sharp rise.

    Parameters
    ----------
    h : float or array-like
        Altitude in km.
    hmF2 : float
        Altitude where function = 1.
    hmE : float
        Altitude where function = 0.
    drop_fraction : float, optional
        Fraction of the total distance (hmF2 - hmE) over which the transition
        occurs.

    Returns
    -------
    f : float or array
        Function value at altitude h.
    """
    h = np.asarray(h)
    D = hmF2 - hmE
    # if D <= 0:
    #     raise ValueError("hmF2 must be greater than hmE.")
    B = drop_fraction * D
    ht = hmE + B  # center of logistic curve
    sigma_h = logistic_curve(h, ht, B)
    sigma_E = logistic_curve(hmE, ht, B)
    sigma_F2 = logistic_curve(hmF2, ht, B)

    return (sigma_h - sigma_E) / (sigma_F2 - sigma_E)


def Probability_F1(year, mth, utime, alon, alat):
    """Calculate probability occurrence of F1 layer.

    Parameters
    ----------
    year : int
        Year.
    mth : int
        Month.
    time : array-like
        Array of UTs in hours.
    alon : array-like
        Flattened array of longitudes in degrees.
    alat : array-like
        Flattened array of latitudes in degrees.

    Returns
    -------
    a_P : array-like
        Probability occurrence of F1 layer.

    Notes
    -----
    This function calculates numerical maps probability of F1 layer.

    References
    ----------
    Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
    International Reference Ionosphere Modeling Implemented in Python,
    Space Weather.

    Bilitza et al. (2022), The International Reference Ionosphere
    model: A review and description of an ionospheric benchmark, Reviews
    of Geophysics, 60.

    """
    # make arrays to hold numerical maps for 2 levels of solar activity
    a_P = np.zeros((utime.size, alon.size, 2))

    # simplified constant value from Bilitza review 2022
    gamma = 2.36

    # solar zenith angle for day 15 in the month of interest
    solzen, _, _ = main.solzen_timearray_grid(year, mth, 15,
                                              utime, alon,
                                              alat)

    for isol in range(0, 2):
        a_P[:, :, isol] = (0.5 + 0.5 * np.cos(np.deg2rad(solzen)))**gamma

    return a_P

'''
Nina's functions
'''

def real_SH_func(lmax: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Generate real-valued spherical harmonic basis functions up to degree lmax
    (unnormalized). Includes Condon-Shortley phase. To use with SH coefficients
    created with pyshtools, set cspase=-1 and normalization='unnorm' in
    pyshtools.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree (and order).
    theta : ndarray (n_time, n_pos) or (n_pos,)
        User input colatitudes in radians [0-π].
    phi : ndarray (n_time, n_pos) or (n_pos,)
        User input longitudes in radians [0-2π).

    Returns
    -------
    F_SH : ndarray (n_SH, n_time, n_pos) or (n_SH, n_pos)
        Real-valued spherical harmonic basis matrix, with `n_SH = (lmax+1)**2`.
        Each column corresponds to a pair of coordinates, each row to an SH
        mode. The SH mode of degree l and order m is stored in row 
        i = l * (l + 1) + m.
    """
    mlt_flag = 0
    if len(theta.shape) == 1:
        mlt_flag = 1
        theta = theta[np.newaxis, :]
        phi = phi[np.newaxis, :]

    z = np.cos(theta)

    P_all = ss.assoc_legendre_p_all(lmax, lmax, z, norm = False).squeeze(0)
    P_cropped = P_all[:, :lmax + 1, :, :]

    if -1.0 in z:
        idx_180_time = np.where(z == -1.0)[0]
        idx_180_pos = np.where(z == -1.0)[1]
        P_cropped[1::2, 0, idx_180_time, idx_180_pos] *= -1.0

    l_mat = np.ones_like(P_cropped, dtype=float)
    l_range = np.arange(0, lmax + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    l_mat *= l_range
    m_mat = np.ones_like(P_cropped, dtype=float)
    m_range = np.arange(0, lmax + 1)[np.newaxis, :, np.newaxis, np.newaxis]
    m_mat *= np.abs(m_range)

    del_m0 = np.array(m_mat == 0, dtype=float)
    Norm = np.ones_like(P_cropped, dtype=float)
    Norm *= np.sqrt((2 - del_m0) * (2 * l_mat + 1) *
                ss.factorial(l_mat - m_mat) / (ss.factorial(l_mat + m_mat)))

    P_cropped *= Norm

    phi = phi[np.newaxis, :]
    m_range = np.arange(0, lmax + 1)[:, np.newaxis, np.newaxis]

    cos_phi = np.cos(m_range * phi)[np.newaxis, :, :]
    sin_phi = np.sin(m_range * phi)[np.newaxis, :, :]

    Y_all_pos = P_cropped * cos_phi
    Y_all_neg = P_cropped * sin_phi

    n_pos = Y_all_pos.shape[3]
    n_time = Y_all_pos.shape[2]
    lmax = Y_all_pos.shape[0] - 1

    mask = np.triu(np.ones((lmax + 1, lmax + 1), dtype = bool), k = 1)

    Y_all_pos[mask] = np.nan
    Y_all_neg[mask] = np.nan

    Y_all_neg_flip = np.flip(Y_all_neg, axis = 1)

    Y_all_recombined = np.concatenate((Y_all_neg_flip[:, :-1, :, :],
                                       Y_all_pos), axis = 1)
    Y_all_flat = Y_all_recombined.reshape(((lmax + 1) * (2 * lmax + 1),
                                           n_time, n_pos))
    mask_flat = np.isfinite(Y_all_flat[:, 0, 0])
    F_SH = Y_all_flat[mask_flat, :, :]

    if mlt_flag:
        F_SH = F_SH.squeeze(1)

    return F_SH

def real_FS_func(atime_UT: np.ndarray, n_FS_c: int) -> np.ndarray:
    """
    Generate a real-valued Fourier Series (FS) basis matrix for time-series
    expansion.

    Parameters
    ----------
    atime_UT : ndarray of shape (n_time,)
        User input time values in Universal Time (UT) hours [0-24).

    n_FS_c : int
        Number of complex Fourier coefficients to use (i.e., truncation level).

    Returns
    -------
    F_FS : ndarray of shape (n_time, 2 * n_FS_c - 1)
        Real-valued DFT basis matrix.
    """
    n_time = atime_UT.size
    n_FS_r = 2 * n_FS_c - 1
    F_FS = np.empty((n_time, n_FS_r))

    k_vals = np.arange(1, n_FS_c)
    omega = 2 * np.pi * k_vals / 24 
    phase = np.outer(atime_UT, omega)

    F_FS[:, 0] = 1
    F_FS[:, 1::2] = np.cos(phase)
    F_FS[:, 2::2] = np.sin(phase)

    return F_FS


def load_coeff_matrices(mth: int, coeff_dir: str, foF2_coeff: str, 
                        hmF2_model: str) -> np.ndarray:
    """
    Load ionospheric model coefficient matrices from NetCDF files.

    Parameters:
    mth: int, month
    coeff_dir: str, path to coefficient files
    foF2_coeff: str, name of foF2 coefficient set
    hmF2_model: str, name of the hmF2 model to use

    Returns:
    C: np.ndarray (n_params, n_IG12, n_FS, n_SH)
        Coefficient matrix. n_params is the number of parameters (n_params=5
        if hmF2_model != 'BSE1979' else n_params=6), n_IG12=2 is the number of
        IG12 values stored (IG12=0 and IG12=100), n_FS=9 is the number of
        real DFT coefficients used, and n_SH=900 is the number of real SH
        coefficients used.
    """

    filenames = [f'foF2_{foF2_coeff}.nc', 'B0.nc', 'B1.nc', 'M3000F2.nc']

    path = os.path.join(coeff_dir, 'SH_new', filenames[0])
    with nc.Dataset(path) as ds:
        C_mth = ds['Coefficients'][:, mth - 1, :, :]
        n_IG12 = C_mth.shape[0]
        n_DFT = C_mth.shape[1]
        n_SH = C_mth.shape[2]
        coeffs = np.zeros((len(filenames), n_IG12, n_DFT, n_SH))

    for ids in range(len(filenames)):
        fname = filenames[ids]
        path = os.path.join(coeff_dir, 'SH_new', fname)
        with nc.Dataset(path) as ds:
            C_mth = ds['Coefficients'][:, mth - 1, :, :]
            coeffs[ids, :, :, :] = C_mth

    # Optionally add hmF2 coefficients
    if hmF2_model != 'BSE1979':
        hmF2_path = os.path.join(coeff_dir, 'SH_new', f'hmF2_{hmF2_model}.nc')
        with nc.Dataset(hmF2_path) as ds:
            C_mth = ds['Coefficients'][:, mth - 1, :, :]
            C_mth = C_mth[np.newaxis, :, :, :]
            coeffs = np.concatenate([coeffs, C_mth], axis=0)

    return coeffs

def to_numpy_array(x: int | float | list[int | float] |
                   np.ndarray) -> np.ndarray:
    if isinstance(x, (int, float)):
        return np.array([x], dtype=float)
    elif isinstance(x, list):
        if all(isinstance(i, (int, float)) for i in x):
            return np.array(x, dtype=float)
        else:
            raise TypeError("List elements must be int or float.")
    elif isinstance(x, np.ndarray):
        return x.astype(float)
    else:
        raise TypeError("Input must be an int, float, or list of those.")

def get_safe_chunk_plan(array_shape: Tuple[int, ...], axis: int = 0,
    dtype: Union[np.dtype, type] = np.float64,
    max_memory_fraction: float = 0.3,
    extra_bytes_per_element: int = 0) -> Tuple[int, int]:
    """
    Determine a safe chunk size and number of chunks along a given axis
    based on available system memory.

    Parameters:
        array_shape (tuple of int): Shape of the array to be chunked.
        axis (int): Axis to chunk along (default: 0).
        dtype (np.dtype or type): Data type of array elements
        (default: float64).
        max_memory_fraction (float): Fraction of available memory to use
        (0 < f <= 1).
        extra_bytes_per_element (int): Additional bytes per element
        (e.g. temp buffers).

    Returns:
        chunk_size (int): Number of elements along axis to include in each chunk.
        num_chunks (int): Total number of chunks to fully cover the array.
    """
    available_bytes = psutil.virtual_memory().available
    usable_bytes = available_bytes * max_memory_fraction
    dtype = np.dtype(dtype)

    per_element_bytes = dtype.itemsize + extra_bytes_per_element
    slice_shape = list(array_shape)
    slice_shape[axis] = 1
    slice_bytes = np.prod(slice_shape) * per_element_bytes

    max_chunk_size = int(usable_bytes // slice_bytes)
    full_axis_size = array_shape[axis]

    chunk_size = min(max_chunk_size, full_axis_size)
    num_chunks = int(np.ceil(full_axis_size / chunk_size))

    return chunk_size, num_chunks
