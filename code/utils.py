#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for reproducing Harrold et al 2015 on SDSS J160036.83+272117.8.

"""


# Import standard packages.
from __future__ import absolute_import, division, print_function
import collections
import copy
import pdb
import re
import warnings
# Import installed packages.
import astroML.density_estimation as astroML_dens
import astroML.stats as astroML_stats
import astroML.time_series as astroML_ts
import astropy.constants as astropy_con
import binstarsolver as bss
import gatspy.periodic as gatspy_per
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.constants as scipy_con
import seaborn as sns


# Set environment
sns.set() # Set matplotlib styles by seaborn.


class Container(object):
    """Empty class to contain dynamically allocated attributes.
    Use to minimize namespace clutter from variable names.
    Use for heirarchical data storage like a `dict`.
    
    Examples
    --------
    ```
    model = Container()
    model.fit = Container()
    model.fit.values = fits
    ```
    """
    pass


# TODO: rename to use lower case
def flux_ADU_to_electrons(flux_ADU, gain_eADU):
    """Convert flux from ADU to electrons using CCD gain.
    
    Parameters
    ----------
    flux_ADU : float
        Flux count in ADU.
    gain_eADU : float
        Conversion gain of CCD in electons/ADU.
    
    Returns
    -------
    flux_electrons : float
        Flux count in electrons.
    
    Notes
    -----
    TODO: Use gain correction methods from tsphot and ccdproc.
    """
    return flux_ADU * gain_eADU


def flux_intg_to_rate(flux_intg, exptime):
    """Divide integrated flux by exposure time to get flux rate.
      
    Parameters
    ----------
    flux_intg : float
        Integrated flux of star over exposure time and area on detector (e.g. ADU)
    exptime : float
        Exposure time over which flux was integrated (e.g. seconds)
    
    Returns
    -------
    flux_rate : float
        flux_rate = flux_intg / exptime
    """
    return flux_intg / exptime


def flux_rate_to_magC(flux_rate):
    """Compute magnitude omitting a correction constant C from a flux rate.
      
    Parameters
    ----------
    flux_rate : float
        Flux rate of star in electrons/second
    
    Returns
    -------
    magC_dex : float
        magC_dex = -2.5*log_10(flux_rate)
        
    Notes
    -----
    From sec 5.1.3 of [1]_:
    Definition: mag = -2.5*log_10(flux) + C
    C is correction constant, 23.5-26 for Earth-based sites.
    C is omitted from the returned result.
        
    References
    ----------
    .. [1] Howell, 2009, "Handbook of CCD Astronomy"
    """
    return -2.5*np.log10(flux_rate)


def extinct_airmass(airmass):
    """Compute atmospheric extinction factor due to airmass.
      
    Parameters
    ----------
    airmass : float
        Number of atmospheres through which observation is being made.
        Example: 1.15
    
    Returns
    -------
    extinct : float
        Extinction factor omitting extinction coefficient k.
        extinct = 2.5*np.log10(np.exp(1.0))*airmass
        
    Notes
    -----
    From eqns 4.3 and 4.4 of [1]_ and with sec 5.1.3 of [2]_::
    flux_obs = flux_ref*exp(−k_obs*X_obs)
    m_obs = m_ref + 2.5*log10(exp)*k_obs*X_obs + C_obs
    k = extinction coefficient as a function of wavelength
    X = airmass
        
    References
    ----------
    .. [1] Budding, 2007, "Introduction to Astronomical Photometry"
    .. [2] Howell, 2009, "Handbook of CCD Astronomy"
    """
    return 2.5*np.log10(np.exp(1.0))*airmass


def composite_flux(flux_ADU, gain_eADU, exptime):
    """Composite function for flux. Convenience function.
    
    Parameters
    ----------
    flux_ADU : float
    gain_eADU : float
    exptime : float
    
    Returns
    -------
    magC_dex : float
    """
    # TODO: restructure as lambda functions.
    return \
        flux_rate_to_magC(flux_rate=
            flux_intg_to_rate(flux_intg=
                flux_ADU_to_electrons(flux_ADU=flux_ADU,
                                      gain_eADU=gain_eADU),
                exptime=exptime))


def try_color_to_Teff(color, colorbands='g-r'):
    """Fault-tolerant wrapper for `astropysics.photometry.color_to_Teff`
    
    Passes `ValueError`.
    
    Parameters
    ----------
    color : float
    colorbands : {'g-r'}, string, optional
    
    Returns
    -------
    Teff : float
        Effective temperature of star in Kelvin.
    """
    try:
        Teff = apy_phot.color_to_Teff(color, colorbands=colorbands)
    except ValueError as err:
        print(("WARNING: astropysics.photometry.color_to_Teff raised ValueError.\n" +
               "Message:\n" +
               err.message + "\n" +
               "Arguments:\n" +
               "color = {color}\n" +
               "colorbands = {colorbands}").format(color=color, colorbands=colorbands),
              file=sys.stderr)
        Teff = np.NaN
    return Teff


def rolling_window(arr, window):
    r"""Efficient rolling window for statistics.
    From [1]_.

    Parameters
    ----------
    arr : numpy.ndarray
    window : int
        Number of elements in window.

    Returns
    -------
    arr_windowed : numpy.ndarray
        `arr` where each element has been replaced by a `numpy.ndarray`
        of length `window`, centered on the element. Edge elements are dropped.

    Notes
    -----
    * For examples, see [1]_, [2]_.

    References
    ----------
    .. [1] http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    .. [2] http://stackoverflow.com/questions/6811183/
           rolling-window-for-1d-arrays-in-numpy
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def calc_period_limits(times):
    r"""Calculate the region of dectable periods.
    
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data.
        Unit is time, e.g. seconds.
    
    Returns
    -------
    min_period : float
    max_period : float
        Min, max periods detectable from `times`.
        Units are sames as `times`.
    num_periods : int
        Number of distinguishable periods limited by period resolution.

    Notes
    -----
    - The concept of Nyquist limits does not apply to irregularly sampled
      data ([1]_, [2]_). However, as a conservative constraint, consider only
      periods between 2*median_sampling_period and 0.5*acquisition_duration
      adapted from [3]_. antialias_factor = 2.56 from [3]_.
        med_sampling_period = np.median(np.diff(times))
        acquisition_duration = max(times) - min(times)
        min_period = 2.0 * med_sampling_period
        max_period = 0.5 * aquisition_duration
        antialias_factor = 2.56
        num_periods = \
            int(antialias_factor * acquisition_duration / med_sampling_period)
    
    References
    ----------
    .. [1] VanderPlas and Ivezic, 2015,
           http://adsabs.harvard.edu/abs/2015arXiv150201344V
    .. [2] https://github.com/astroML/gatspy/issues/3
    .. [3] http://zone.ni.com/reference/en-XX/help/372416B-01/
           svtconcepts/fft_funda/
    
    """
    # Account for unsorted, input with duplicate data.
    times = np.sort(times)
    diffs = np.diff(times)
    diffs_nonzero = diffs[diffs > 0]
    med_sampling_period = np.median(diffs_nonzero)
    acquisition_duration = max(times) - min(times)
    min_period = 2.0 * med_sampling_period
    max_period = 0.5 * acquisition_duration
    antialias_factor = 2.56
    num_periods = \
        int(antialias_factor * acquisition_duration / med_sampling_period)
    return (min_period, max_period, num_periods)


def calc_sig_levels(
    model, sig_periods, sigs=(95.0, 99.0, 99.9), num_shuffles=1000):
    r"""Calculate relative powers that correspond to significance levels for
    a multiband generalized Lomb-Scargle periodogram. The noise is modeled by
    shuffling the time series. Convenience function for methods from [1]_, [2]_.

    Parameters
    ----------
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.
    sig_periods : numpy.ndarray
        Periods at which significance levels are to be computed.
        Units are time, same as `model.t`. The significance level changes
        slowly as a function of period.
    sigs : {(95.0, 99.0, 99.9)}, tuple, optional
        `tuple` of `float` that are the levels of statistical significance.
    num_shuffles : {1000}, int, optional
        Number of shuffles used to compute significance levels.
        For 1000 shuffles, the significance level can be computed to a max
        resolution of 0.1 (i.e. 99.9 significance can be computed, not 99.99).

    Returns
    -------
    sig_powers : dict
        `dict` of `numpy.ndarray`. Keys are `sigs`. Values are relative powers
        for each significance level as a `numpy.ndarray`. Units are relative
        power from `model.periodogram`.

    See Also
    --------
    gatspy.periodic.LombScargleMultiband, calc_period_limits, plot_periodogram

    Notes
    -----
    - Significance levels are calculated periods within
        `model.optimizer.period_range`
    - For a given period, a power is "signficant to the 99th percentile" if
        that power is greater than 99% of all other powers due to noise at that
        period. The noise is modeled by shuffling the time series.
    - The time complexity for computing noise levels is approximately linear
        with `num_periods`*`num_shuffles`:
        exec_time ~ 13.6 sec * (num_periods/20) * (num_shuffles/1000)

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] VanderPlas and Ivezic, 2015,
           http://adsabs.harvard.edu/abs/2015arXiv150201344V

    """
    # Calculate percentiles of powers from noise.
    noise_model = copy.deepcopy(model)
    np.random.seed(seed=0) # for reproducibility
    sig_powers_arr = []
    for _ in xrange(num_shuffles):
        np.random.shuffle(noise_model.t)
        sig_powers_arr.append(noise_model.periodogram(periods=sig_periods))
    sig_powers = \
        {sig: np.percentile(a=sig_powers_arr, q=sig, axis=0) for sig in sigs}
    return sig_powers


def plot_periodogram(
    periods, powers, best_period=None, sig_periods=None, sig_powers=None,
    xscale='log', period_unit='seconds', flux_unit='relative', return_ax=False):
    r"""Plot relative power vs period for a Lomb-Scargle periodogram.
    Convenience function for methods from [1]_, [2]_.

    Parameters
    ----------
    periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds.
    powers : numpy.ndarray
        1D array of powers. Unit is relative Lomb-Scargle power spectral density
        from flux and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
    best_period : {None}, float, optional
        Period of light curve model that best represents the time series data.
        Unit is same as `periods`.
    sig_periods : {None}, numpy.ndarray, optional
        Periods at which significance levels were computed.
        Units are same as `periods`.
    sig_powers : {None}, dict, optional
        `dict` of `numpy.ndarray`. Keys are significance levels. Values are
        relative powers for each significance level as a `numpy.ndarray`.
        Units are same as `powers`.
    xscale : {'log', 'linear'}, string, optional
        `matplotlib.pyplot` attribute to plot periods x-scale in
        'log' (default) or 'linear' scale.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label
        the x-axis with "Period (seconds)"
        and label the y-axis with
        "Relative Lomb-Scargle Power Spectral Density" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    return_ax : {False, True}, bool, optional
        If `False` (default), show the periodogram plot. Return `None`.
        If `True`, do not show the periodogram plot. Return a `matplotlib.axes`
        instance for additional modification.

    Returns
    -------
    ax : matplotlib.axes
        Returned only if `return_ax` is `True`. Otherwise returns `None`.

    See Also
    --------
    plot_phased_light_curve, calc_sig_levels

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] VanderPlas and Ivezic, 2015,
           http://adsabs.harvard.edu/abs/2015arXiv150201344V
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, xscale=xscale)
    ax.plot(
        periods, powers, color=sns.color_palette()[0],
        marker='.', label='relative L-S PSD')
    if best_period is not None:
        ax.axvline(
            x=best_period, color=sns.color_palette()[1],
            linestyle=':', label='best period')
    if (sig_periods is not None) and (sig_powers is not None):
        sig_label = str(sig_powers.keys())+'\npercentile sig.'
        for sig in sig_powers.keys():
            ax.plot(
                sig_periods, sig_powers[sig], color=sns.color_palette()[2],
                linestyle='--', marker='.', label=sig_label)
            sig_label=None
    ax.legend(loc='upper left')
    ax.set_xlim(min(periods), max(periods))
    ax.set_title("Multiband generalized Lomb-Scargle periodogram")
    ax.set_xlabel(("Period ({punit})").format(punit=period_unit))
    ax.set_ylabel(
        ("Relative Lomb-Scargle power spectral density\n" +
         "(from flux in {funit}, ang. freq. in 2*pi/{punit})").format(
            funit=flux_unit, punit=period_unit))
    if return_ax:
        return_obj = ax
    else:
        plt.show()
        return_obj = None
    return return_obj


# TODO: speed up with numba
def calc_min_flux_time(
    model, filt, best_period=None, lwr_time_bound=None, upr_time_bound=None,
    tol=0.1, maxiter=10):
    r"""Calculate the time at which the minimum flux occurs. Use to define a
    phase offset in time units so that phase=0 at minimum flux.

    Parameters
    ----------
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.
    filt : string
        Filter from `model` for which to calculate time of minimum flux.
    best_period : {None}, float, optional
        Period of light curve model that best represents the time series data.
        Unit is same as times in `model.t`, e.g. seconds.
        If `None` (default), uses `model.best_period`.
    lwr_time_bound : {None}, float, optional
    upr_time_bound : {None}, float, optional
        Lower and upper bounds for finding minimum flux. Use if global minimum
        for one filter is a local minimum for another filter.
        Unit is same as times in `model.t`, e.g. seconds.
        Required: 0.0 <= lwr_time_bound < upr_time_bound <= best_period
    tol : {0.1}, float, optional
        Tolerance for maximum permissible uncertainty in solved `min_flux_time`.
        Unit is same as times in `model.t`, e.g. seconds.
    maxiter : {10}, int, optional
        Maximum number of iterations permitted in solving `min_flux_time`.
        Example: For `best_period` = 86400 seconds and `tol` = 0.1 seconds,
        `min_flux_time` is typically solved to within `tol` by ~5 iterations.

    Returns
    -------
    min_flux_time : float
        Time at which minimum flux occurs.

    See Also
    --------
    gatspy.periodic.LombScargleMultiband, calc_phases

    Notes
    -----
    - To create a phased light curve with minimum flux at phase=0:
        Instead of `plt.plot(times % best_period, fluxes)`
        do `plt.plot((times - min_flux_time) % best_period, fluxes)`
    - A global minimum in a light curve for one filter may be a local minimum
        in the light curve for another filter. Example light curve: [1]_

    Raises
    ------
    ValueError
        - Raised if not
            0.0 <= lwr_time_bound < upr_time_bound <= best_period
    warnings.warn
        - Raised if solution for `min_flux_time` did not converge to within
            tolerance.
    AssertionError
        - Raised if not 0 <= `lhs_time_init` <= `min_flux_time` <= `rhs_time_init`
            <= `best_period`, where `lhs_time_init` and `rhs_time_init` are
            initial bounds for time of global minimum flux.
        - Raised if not `min_flux` <= `min_flux_init`, where `min_flux_init`
            is the initial bound for global minimum flux and `min_flux` is the
            global minimum flux.

    References
    ----------
    .. [1] https://github.com/astroML/gatspy/blob/master/examples/
           MultiBand.ipynb

    """
    # Check input.
    if best_period is None:
        best_period = model.best_period
    if lwr_time_bound is None:
        start = 0.0
    else:
        if not (0.0 <= lwr_time_bound):
            raise ValueError("Required: 0.0 <= `lwr_time_bound`")
        start = lwr_time_bound
    if upr_time_bound is None:
        stop = best_period
    else:
        if not (upr_time_bound <= best_period):
            raise ValueError("Required: `upr_time_bound` <= `best_period`")
        stop = upr_time_bound
    if (lwr_time_bound is not None) and (upr_time_bound is not None):
        if not (lwr_time_bound < upr_time_bound):
            raise ValueError("Required: `lwr_time_bound` < `upr_time_bound`")
    # Compute initial phased times and fit fluxes.
    phased_times_fit = \
        np.linspace(start=start, stop=stop, num=1000, endpoint=False)
    phased_fluxes_fit = \
        model.predict(
            t=phased_times_fit, filts=[filt]*len(phased_times_fit),
            period=best_period)
    fmt_parameters = \
        ("best_period = {bp}\n" +
         "tol = {tol}\n" +
         "maxiter = {maxiter}").format(
            bp=best_period, tol=tol, maxiter=maxiter)
    # Prepend/append data if min flux is at beginning/end of folded time series.
    min_idx = np.argmin(phased_fluxes_fit)
    if min_idx == 0:
        phased_times_fit = np.append(phased_times_fit[-1:], phased_times_fit)
        phased_fluxes_fit = np.append(phased_fluxes_fit[-1:], phased_fluxes_fit)
        min_idx = 1
    elif min_idx == len(phased_fluxes_fit)-1:
        phased_times_fit = np.append(phased_times_fit, phased_times_fit[:0])
        phased_fluxes_fit = np.append(phased_fluxes_fit, phased_fluxes_fit[:0])
    # Bound left- and right-hand-sides of time at which flux is global min. 
    (lhs_idx, rhs_idx) = (min_idx - 1, min_idx + 1)
    (lhs_time, lhs_flux) = \
        (phased_times_fit[lhs_idx], phased_fluxes_fit[lhs_idx])
    (min_flux_time, min_flux) = \
        (phased_times_fit[min_idx], phased_fluxes_fit[min_idx])
    (rhs_time, rhs_flux) = \
        (phased_times_fit[rhs_idx], phased_fluxes_fit[rhs_idx])
    (lhs_time_init, rhs_time_init) = (lhs_time, rhs_time)
    min_flux_init = min_flux
    # Zoom in on time of global flux min.
    itol = rhs_time - lhs_time
    inum = 0
    while (itol > tol) and (inum < maxiter):
        phased_times_subfit = \
            np.linspace(start=lhs_time, stop=rhs_time, num=10, endpoint=True)
        phased_fluxes_subfit = \
            model.predict(
                t=phased_times_subfit, filts=[filt]*10, period=best_period)
        min_subidx = np.argmin(phased_fluxes_subfit)
        (lhs_subidx, rhs_subidx) = (min_subidx - 1, min_subidx + 1)
        (lhs_time, lhs_flux) = \
            (phased_times_subfit[lhs_subidx], phased_fluxes_subfit[lhs_subidx])
        (min_flux_time, min_flux) = \
            (phased_times_subfit[min_subidx], phased_fluxes_subfit[min_subidx])
        (rhs_time, rhs_flux) = \
            (phased_times_subfit[rhs_subidx], phased_fluxes_subfit[rhs_subidx])
        itol = rhs_time - lhs_time
        inum += 1
    # Check that solution converged within `tol` and `maxiter` constraints.
    if (itol > tol) and (inum >= maxiter):
        warnings.warn(
            "\n" +
            "Solution for `min_flux_time` did not converge to within tolerance.\n" +
            "Input parameters:\n" +
            fmt_parameters)
    # Check that program executed correctly.
    if not (
        (0 <= lhs_time_init) and (lhs_time_init <= min_flux_time) and
        (min_flux_time <= rhs_time_init) and (rhs_time_init <= best_period)):
        raise AssertionError(
            ("Program error.\n" +
             "Required: 0 <= `lhs_time_init` <= `min_flux_time`" +
             " <= `rhs_time_init` <= `best_period`:\n" +
             "lhs_time_init = {lhi}\n" +
             "min_flux_time = {mt}\n" +
             "rhs_time_init = {rhi}\n" +
             "best_period = {bp}").format(
                lhi=lhs_time_init, mt=min_flux_time,
                rhi=rhs_time_init, bp=best_period))
    # TODO: This assertion case fails during MCMC. Fix.
    # Use numpy.less_equal for high precision.
    # if not np.less_equal(min_flux, min_flux_init):
    #     raise AssertionError(
    #         ("Program error.\n" +
    #          "Required: `min_flux` <= `min_flux_init`\n" +
    #          "min_flux = {mf}\n" +
    #          "min_flux_init = {mfi}").format(
    #             mf=min_flux, mfi=min_flux_init))
    return min_flux_time


# TODO: speed up with numba
def calc_phases(times, best_period, min_flux_time=0.0):
    r"""Calculate phases of a light curve.

    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates of observed data.
        Units are time, e.g. seconds.
    best_period : float
        Period that best represents the time series. Unit is same as `times`.
    min_flux_time : {0.0}, float, optional
        Time at which minimum flux occurs. Unit is same as `times`.
        Required: 0.0 <= `min_flux_time` <= `best_period`

    Returns
    -------
    phases : numpy.ndarray
        Phases of `times` such that 0 <= `phases` <= 1.0. Unit is decimal phase.
        `numpy.shape(phases) == numpy.shape(times)`.

    See Also
    --------
    calc_min_flux_time, calc_next_phase0_time, plot_phased_light_curve

    Raises
    ------
    ValueError
        - Raised if not 0.0 <= `min_flux_time` <= best_period.

    Notes
    -----
    - See section 8.1 of [1]_ for time-phase converison.

    References
    ----------
    .. [1] Budding, 2007, "Introduction to Astronomical Photometry"

    """
    # Check input.
    if not ((0.0 <= min_flux_time) and (min_flux_time <= best_period)):
        raise ValueError("Required: 0.0 <= `min_flux_time` <= `best_period`")
    # Calculate phased times.
    phases = \
        np.true_divide(
            np.mod(
                np.subtract(times, min_flux_time),
                best_period),
            best_period)
    return phases


def calc_next_phase0_time(time, phase, best_period):
    r"""Calculate next time at which light curve has phase=0.

    Parameters
    ----------
    time : float
        Time coordinate. Units is time, e.g. seconds.
    phase : float
        Phase of `time`. Unit is decimal phase.
        Required: 0.0 <= phase <= 1.0
    best_period : float
        Period that best represents the time series. Unit is same as `times`.

    Returns
    -------
    next_phase0_time : float
        Next time at which phase=0.

    See Also
    --------
    calc_min_flux_time

    Raises
    ------
    ValueError
        - Raised if not 0.0 <= `phase` <= 1.0

    Notes
    -----
    - See section 8.1 of [1]_ for time-phase converison.

    References
    ----------
    .. [1] Budding, 2007, "Introduction to Astronomical Photometry"

    """
    # Check input.
    if not ((0.0 <= phase) and (phase <= 1.0)):
        raise ValueError("Required: 0.0 <= `phase` <= 1.0")
    # Calculate time.
    next_phase0_time = time + ((1.0 - phase)*best_period)
    return next_phase0_time


def plot_phased_light_curve(
    phases, fluxes, fluxes_err, fit_phases, fit_fluxes,
    flux_unit='relative', return_ax=False):
    r"""Plot a phased light curve. Convenience function for plot formats
    from [1]_, [2]_.

    Parameters
    ----------
    phases : numpy.ndarray
        1D array of phased time coordinates of observed data.
        Required: 0.0 <= `phases` <= 1.0. Units are decimal phase.
    fluxes : numpy.ndarray
        1D array of fluxes corresponding to `phases`.
        Units are relative integrated flux.
    fluxes_err : numpy.ndarray
        1D array of errors for `fluxes`. Units are same as `fluxes`.
    fit_phases : numpy.ndarray
        1D array of phased time coordinates of the best-fit light curve.
        Required: 0.0 <= `fit_phases` <= 1.0. Units are same as `phases`.
    fit_fluxes : numpy.ndarray
        1D array of fluxes corresponding to `fit_phases`.
        Units are same as `fluxes`.
    flux_unit : {'relative'}, string, optional
        String describing flux units for labeling the plot.
        Example: flux_unit='relative' will label the y-axis
        with "Flux (relative)".
    return_ax : {False, True}, bool
        If `False` (default), show the periodogram plot. Return `None`.
        If `True`, return a `matplotlib.axes` instance
        for additional modification.

    Returns
    -------
    ax : matplotlib.axes
        Returned only if `return_ax` is `True`. Otherwise returns `None`.
    
    See Also
    --------
    plot_periodogram, calc_min_flux_time, calc_phases

    Raises
    ------
    ValueError
        - Raised if not 0.0 <= {`phases`, `fit_phases`} <= 1.0
    
    Notes
    -----
    - The phased light curve is plotted through 3 complete cycles
        to illustrate the light curve shape.

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] VanderPlas and Ivezic, 2015,
           http://adsabs.harvard.edu/abs/2015arXiv150201344V
    
    """
    # Check input
    if not \
        (np.all(np.less_equal(0.0, phases)) and
         np.all(np.less_equal(phases, 1.0))):
        raise ValueError("Required: 0.0 <= `phases` <= 1.0")
    if not \
        (np.all(np.less_equal(0.0, fit_phases)) and
         np.all(np.less_equal(fit_phases, 1.0))):
        raise ValueError("Required: 0.0 <= `fit_phases` <= 1.0")
    # Append the data to itself (tesselate) 2 times for total of 3 cycles.
    (tess_phases, tess_fluxes, tess_fluxes_err) = (phases, fluxes, fluxes_err)
    (tess_fit_phases, tess_fit_fluxes) = (fit_phases, fit_fluxes)
    for begin_phase in xrange(0, 2):
        tess_phases = \
            np.append(tess_phases, np.add(phases, begin_phase + 1.0))
        tess_fluxes = np.append(tess_fluxes, fluxes)
        tess_fluxes_err = np.append(tess_fluxes_err, fluxes_err)
        tess_fit_phases = \
            np.append(tess_fit_phases, np.add(fit_phases, begin_phase + 1.0))
        tess_fit_fluxes = np.append(tess_fit_fluxes, fit_fluxes)
    # Plot tesselated light curve.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(
        tess_phases, tess_fluxes, tess_fluxes_err, ecolor='gray',
        marker='.', linestyle='', linewidth=1, label='observed')
    ax.plot(tess_fit_phases, tess_fit_fluxes,
        marker='', linestyle='-', label='fit')
    ax.legend(loc='upper left')
    ax.set_title("Phased light curve")
    ax.set_xlabel("Orbital phase (decimal)")
    ax.set_ylabel("Flux ({unit})".format(unit=flux_unit))
    if return_ax:
        return_obj = ax
    else:
        plt.show()
        return_obj = None
    return return_obj


def calc_residual_fluxes(
    phases, fluxes, fit_phases, fit_fluxes):
    r"""Calculate the residual fluxes at phased time coordinates from a fit
    to a light curve.
    
    Parameters
    ----------
    phases : numpy.ndarray
        1D array of phased time coordinates of observed data.
        Units are decimal orbital phase.
    fluxes : numpy.ndarray
        1D array of fluxes corresponding to `phases`.
        Units are relative integrated flux.
    fit_phases : numpy.ndarray
        1D array of phased time coordinates of the best-fit light curve.
        Units are same as `phases`.
    fit_fluxes : numpy.ndarray
        1D array of fluxes corresponding to `fit_phases`.
        Units are same as `fluxes`.

    Returns
    -------
    residual_fluxes : numpy.ndarray
        1D array of the differences between `fluxes` and `fit_fluxes`
        resampled at `phases`:
        `residual_fluxes = fluxes - resampled_fit_fluxes`
        `numpy.shape(residual_fluxes) == numpy.shape(fluxes)`
    resampled_fit_fluxes : numpy.ndarray
        1D array of `fit_fluxes` resampled at `phases`.
        `numpy.shape(resampled_fit_fluxes) == numpy.shape(fluxes)`

    """
    # NOTE: `numpy.interp` requires that `xp` is monotonically increasing.
    (sorted_fit_phases, sorted_fit_fluxes) = \
        zip(*sorted(zip(fit_phases, fit_fluxes), key=lambda tup: tup[0]))
    resampled_fit_fluxes = \
        np.interp(x=phases, xp=sorted_fit_phases, fp=sorted_fit_fluxes)
    residual_fluxes = np.subtract(fluxes, resampled_fit_fluxes)
    return (residual_fluxes, resampled_fit_fluxes)


def calc_z1_z2(
    dist):
    r"""Calculate a rank-based measure of Gaussianity in the core
    and tail of a distribution.
    
    Parameters
    ----------
    dist : array_like
        Distribution to evaluate. 1D array of `float`.
    
    Returns
    -------
    z1 : float
    z2 : float
        Departure of distribution core (z1) or tails (z2) from that of a
        Gaussian distribution in number of sigma.
        
    Notes
    -----
    - From section 4.7.4 of [1]_:
        z1 = 1.3 * (abs(mu - median) / sigma) * sqrt(num_dist)
        z2 = 1.1 * abs((sigma / sigmaG) - 1.0) * sqrt(num_dist)
        where mu = mean(residuals), median = median(residuals),
        sigma = standard_deviation(residuals), num_dist = len(dist)
        sigmaG = sigmaG(dist) = rank_based_standard_deviation(dist) (from [1]_)
    - Interpretation:
        For z1 = 1.0, the probability of a true Gaussian distribution also with
        z1 > 1.0 is ~32% and is equivalent to a two-tailed p-value |z1| > 1.0.
        The same is true for z2.
    
    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    (mu, sigma) = astroML_stats.mean_sigma(dist)
    (median, sigmaG) = astroML_stats.median_sigmaG(dist)
    num_dist = len(dist)
    z1 = 1.3 * (abs(mu - median) / sigma) * np.sqrt(num_dist)
    z2 = 1.1 * abs((sigma / sigmaG) - 1.0) * np.sqrt(num_dist)
    return (z1, z2)


def calc_nterms_base(
    model, zoom_periods, nterms_base_list=range(1, 20),
    show_summary_plots=True,  show_periodograms=False,
    period_unit='seconds', flux_unit='relative'):
    r"""Calculate the number of Fourier terms that best represent a `gatspy`
    base model of the data's variability. The model is a multi-band, multi-term
    generalized Lomb-Scargle periodogram. Convenience function for methods
    from [1]_, [2]_.
       
    Parameters
    ----------
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.
    zoom_perdiods : numpy.ndarray
        Zoomed period space around best period for computing the powers to
        evaluate the `model`. Ensure that period space is sampled at much
        greater resolution than the data.
    nterms_base_list : range(1, 20), iterable, optional
        Iterable (e.g. `list`) over number of Fourier terms to attempt fitting
        for `gatspy` base model.
    show_periodograms : {False, True}, bool, optional
        If `False` (default), do not display periodograms (power vs period)
        for each candidate number of base terms. Used for debugging.
    show_summary_plots : {True, False}, bool, optional
        If `True` (default), display summary plots of delta BIC vs number of
        base terms, periodogram and phased light curve for best fit number of
        base terms.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plots.

    Returns
    -------
    model_best : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy` with optimal number of terms for the base model.
        The number of terms is determined by the maximum
        Bayesian Information Criterion, adapted from section 10.3.3 of [1]_
        and `astroML.time_series.lomb_scargle_BIC`. A new best period is
        computed for the optimized model.

    See Also
    --------
    gatspy.periodic.LombScargleMultiband, calc_perod_limits

    Notes
    -----
    - From [2]_, the "base" model is the relative shape of a light curve and
        is common to all filters. The "band" model is the phase offset of the
        light curve between filters.
    - From 10.3.3 of [1]_, many light curves of eclipses are well
        represented with ~6 terms.
    - The number of terms in the "band" model is not changed when computing the
        Bayesian Information Criterion values. From [2]_, `gatspy`
        performs best when model.Nterms_band >= model.Nterms_base.
    - Both the number of terms in the "base" model and in the "band" model
        are included in the calculation of the Bayesian Information Criterion.

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] VanderPlas and Ivezic, 2015,
           http://adsabs.harvard.edu/abs/2015arXiv150201344V
    
    """
    # Check input.
    zoom_periods = sorted(zoom_periods)
    nterms_base_list = sorted(nterms_base_list)
    # Recursive copy input models to avoid altering.
    model_init  = copy.deepcopy(model)
    # Compute Bayesian Information Criterion values for
    # Nterms_band <= nterms_base <= max_nterms_base.
    # NOTE: model_test.Nterms_band should always == model_init.Nterms_band,
    #     i.e. only model_test.Nterms_base ever changes.
    nterms_base_bics = []
    for nterms_base in nterms_base_list:
        model_test = copy.deepcopy(model_init)
        model_test.Nterms_base = nterms_base
        # Refit the model to the data with the updated nterms_base
        model_test.fit(
            t=model_test.t, y=model_test.y,
            dy=model_test.dy, filts=model_test.filts)
        zoom_powers = model_test.periodogram(periods=zoom_periods)
        rel_bic = max(
            astroML_ts.lomb_scargle_BIC(
                P=zoom_powers, y=model_test.y, dy=model_test.dy,
                n_harmonics=model_test.Nterms_base+model_test.Nterms_band))
        nterms_base_bics.append((nterms_base, rel_bic))
        if show_periodograms:
            print()
            print(80*'-')
            print("Number of Fourier terms for base model: {num}".format(
                num=model_test.Nterms_base))
            plot_periodogram(
                periods=zoom_periods, powers=zoom_powers, xscale='linear',
                period_unit='seconds', flux_unit='relative', return_ax=False)
            print("Bayesian Information Criterion: {bic}".format(
                bic=rel_bic))
        assert model_init.Nterms_band == model_test.Nterms_band
    # Choose the best number of Fourier terms from the maximum delta BIC.
    # Create optimized model and recompute best period using
    # intial `period_range`.
    best_idx = np.argmax(zip(*nterms_base_bics)[1])
    (best_nterms_base, best_bic) = nterms_base_bics[best_idx]
    model_best = \
        gatspy_per.LombScargleMultiband(
            Nterms_base=best_nterms_base, Nterms_band=model_init.Nterms_band)
    model_best.fit(
            t=model_init.t, y=model_init.y,
            dy=model_init.dy, filts=model_init.filts)
    model_best.optimizer.period_range = model_init.optimizer.period_range
    # Calculate best period.
    model_best.best_period
    if show_summary_plots:
        # Plot delta BICs after all terms have been fit.
        print(80*'-')
        nterms_base_bics_t = zip(*nterms_base_bics)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(nterms_base_bics_t[0], nterms_base_bics_t[1], marker='.')
        ax.set_title("Bayesian Information Criterion vs\n" +
                     "number of Fourier terms for base model")
        ax.set_xlabel("Number of Fourier terms")
        ax.set_ylabel("BIC")
        plt.show()
        zoom_powers = model_best.periodogram(periods=zoom_periods)
        plot_periodogram(
            periods=zoom_periods, powers=zoom_powers, xscale='linear',
            period_unit='seconds', flux_unit='relative', return_ax=False)
        print("Best number of Fourier terms for base model: {num}".format(
            num=best_nterms_base))
        print("Bayesian Information Criterion: {bic}".format(
            bic=best_bic))
        print("Best period for base model: {per} {unit}".format(
            per=model_best.best_period, unit=period_unit))
    return model_best


# TODO: speedup with @numba.jit(nopython=True)
def ls_are_valid_params(params, model):
    r"""Check if parameters are valid for Lomb-Scargle light curve model.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
        `params = (best_period)`
        Units are:
            {best_period} = time, e.g. seconds
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.

    Returns
    -------
    are_valid : bool
        True if all of the following hold:
            If 0 < `min_period` <= `best_period` <= `max_period`
            where `(min_period, max_period) = model.optimizer.period_range`

    See Also
    --------
    ls_model_fluxes_rel

    Notes
    -----
    - See `seg_model_fluxes_rel` for description of parameters.

    """
    (best_period) = params
    (min_period, max_period) = model.optimizer.period_range
    if ((0 < min_period) and (min_period <= best_period) and
        (best_period <= max_period)):
        are_valid = True
    else:
        are_valid = False
    return are_valid


# TODO: speed up with @numba.jit(nopython=True)
# requires rewriting methods within gatspy.
def ls_model_fluxes_rel(params, model):
    r"""Calculate relative fluxes for a Lomb-Scargle model of the light curve.

    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
        `params = (best_period)`
            best_period :  Period that best represents the time series data.
        Units are:
            {best_period} = time, e.g. seconds
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.

    Returns
    -------
    modeled_fluxes_rel : numpy.ndarray
        1D array of modeled relative fluxes. Units are relative integrated flux.   

    See Also
    --------
    ls_are_valid_params

    Notes
    -----
    - Requires that all input parameters are already checked as valid.

    """
    (best_period) = params
    modeled_fluxes_rel = model.predict(
        t=model.t, filts=model.filts, period=best_period)
    return modeled_fluxes_rel


# TODO: speedup with @numba.jit(nopython=True)
def ls_log_prior(params, model):
    r"""Log prior of Lomb-Scargle light curve model parameters up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.

    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta))
        if parameters are outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    ls_are_valid_params, ls_model_fluxes_rel

    Notes
    -----
    - See `ls_model_fluxes_rel` for description of parameters.
    - This is an uninformative prior:
        `lnp = 0.0` if `theta` is within constraints.
        `lnp = -numpy.inf` otherwise.

    References
    ----------
    ..[1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    ..[2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    ..[3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if ls_are_valid_params(params=params, model=model):
        lnp = 0.0
    else:
        lnp = -np.inf
    return lnp


# TODO: speedup with @numba.jit(nopython=True)
def ls_log_likelihood(params, model):
    r"""Log likelihood of Lomb-Scargle light curve model's relative flux values
    given model parameters. Log likelihood is calculated up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.

    Returns
    -------
    lnp : float
        Log probability of relative flux values: ln(p(y|x, theta))
        If parameters are outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    ls_are_valid_params, ls_model_fluxes_rel

    Notes
    -----
    - See `ls_model_fluxes_rel` for description of parameters.

    References
    ----------
    .. [1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    .. [2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    .. [3] http://dan.iel.fm/emcee/current/user/line/

    """
    if ls_are_valid_params(params=params, model=model):
        modeled_fluxes_rel = ls_model_fluxes_rel(params=params, model=model)
        # Calculation for `lnp` adapted from [1]_.
        log_terms = np.log(2.0*np.pi*model.dy**2.0)
        res_terms = ((model.y - modeled_fluxes_rel) / model.dy)**2.0
        lnp = -0.5*np.sum(log_terms + res_terms)
    else:
        lnp = -np.inf
    return lnp


# TODO: speed up with @numba.jit(nopython=True)
def ls_log_posterior(params, model):
    r"""Log probability of Lomb-Scargle light curve model parameters
    given the data. Log probability is calculated up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.

    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta|x, y))
        If parameters outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    ls_are_valid_params, ls_model_fluxes_rel,
    ls_log_prior, ls_log_likelihood

    Notes
    -----
    - See `ls_model_fluxes_rel` for description of parameters.

    References
    ----------
    .. [1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    .. [2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    .. [3] http://dan.iel.fm/emcee/current/user/line/

    """
    if ls_are_valid_params(params=params, model=model):
        # Calculation of `lnp` adapted from [1]_.
        lnpr = ls_log_prior(params=params, model=model)
        lnlike = ls_log_likelihood(params=params, model=model)
        lnp = lnpr + lnlike
    else:
        lnp = -np.inf
    return lnp


@numba.jit(nopython=True)
def seg_are_valid_params(params):
    r"""Check if parameters are valid for segmented, symmetric light curve model.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
        `params = 
            (phase_rel_int, phase_rel_ext,
             flux_pri_eclipse, flux_out_eclipse, flux_sec_eclipse,
             flux_sigma)`
        Units are:
            {phase_*} = decimal orbital phase
            {flux_*} = relative flux

    Returns
    -------
    are_valid : bool
        True if all of the following hold:
            If 0 < `phase_rel_int` < `phase_rel_ext` < 0.5
            If `flux_pri_eclipse` < `flux_sec_eclipse` < `flux_out_eclipse`
            If 0 < `flux_sigma`
        False otherwise.

    See Also
    --------
    seg_model_fluxes_rel

    Notes
    -----
    * See `seg_model_fluxes_rel` for description of parameters.
    * Return `bool` rather than raise exception for optimization with `numba`.

    """
    # Allow arbitrary flux values for flexibility.
    # Some models may require flux between minima >~ 1 as a normalization
    # factor (e.g. Ch 7 of Budding 2007).
    (p1, p2, b0, b2, b4, sig) = params
    if ((0.0 < p1 and p1 < p2 and p2 < 0.5) and
        (b0 < b4 and b4 < b2) and
        (sig > 0)):
        are_valid = True
    else:
        are_valid = False
    return are_valid


@numba.jit(nopython=True)
def seg_model_fluxes_rel(params, phases):
    r"""Calculate relative fluxes for a segmented, symmetric model of the
    folded eclipse light curve.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
        `params = 
            (phase_rel_int, phase_rel_ext,
             flux_pri_eclipse, flux_out_eclipse, flux_sec_eclipse,
             flux_sigma)`
            phase_rel_int: Relative phase of internal tangencies
                (end ingress/begin egress; p1, p4 under 'Notes').
            phase_rel_ext: Relative phase of external tangencies
                (begin ingress/end egress; p2, p3 under 'Notes').
            flux_pri_eclipse: Flux at primary minimum (f0 under 'Notes').
            flux_out_eclipse: Flux for phases that occur between minima
                (f2 under 'Notes').
            flux_sec_eclipse: Flux at secondary minimum (f4 under 'Notes').
            flux_sigma: Standard deviation of relative fluxes. Assumes that
                all fluxes are drawn from the same distribution.
        Units are:
            {phase_*} = decimal orbital phase
            {flux_*} = relative flux
    phases : numpy.ndarray
        1D array of orbital phases. Units are decimal orbital phase folded at
        phase=0.5. 0 <= `phase` <= 0.5.
        
    Returns
    -------
    modeled_fluxes_rel : numpy.ndarray
        1D array of modeled relative fluxes. Units are relative integrated flux.

    See Also
    --------
    seg_are_valid_params
    
    Notes
    -----
    * Requires that all input parameters are already checked as valid.
    * Segmented, symmetric eclipse light curve model:
        Durations of primary minimum and secondary minimum are equal.
        Durations of primary ingress/egress and secondary ingress/egress
        are equal. Light curve is segmented into functions f(x).
        x values are `phase`, p0, p1, ...

        light      |  |  |--------|  |  |
        curve:     |  | /|        |\\|__|
                   |__|/ |        |  |  |
        function:  |f0|f1|   f2   |f3|f4|
        phase:     p0 p1 p2       p3 p4 p5
        
        boundary condition:       p0     = 0.0
        primary minima:           f0(x)  = b0; 0.0 <= x < p1
        boundary condition:       f0(p1) = f1(p1)
        primary ingress/egress:   f1(x)  = m1*x + b1; p1 <= x < p2
        boundary condition:       f1(p2) = f2(p2)
        between minima:           f2(x)  = b2; p2 <= x < p3
        boundary condition:       f2(p3) = f3(p3)
        secondary ingress/egress: f3(x)  = m3*x + b3; p3 <= x < p4
        boundary condition:       f3(p4) = f4(p4)
        secondary minima:         f4(x)  = b4; p4 <= x <= 0.5
        boundary condition:       p4     = p5 - p1
        boundary condition:       p3     = p5 - p2
        boundary condition:       p5     = 0.5
    - Model parameters are defined relative to primary minima since
        primary minima are deeper and easier to detect from observations.
    - The out-of-eclipse flux level (f2 above), is left variable as a
        normalization factor following [1]_.

    References
    ----------
    .. [1] Budding, 2007, "Introduction to Astronomical Photometry"
    
    """
    # Compute modeled relative flux...
    (p1, p2, b0, b2, b4, _) = params
    p5 = 0.5
    p4 = p5 - p1
    p3 = p5 - p2
    num_phases = len(phases)
    modeled_fluxes_rel = np.empty(num_phases)
    idx = 0
    m1 = (b2 - b0)/(p2 - p1)
    b1 = b0 - m1*p1
    m3 = (b4 - b2)/(p4 - p3)
    b3 = b2 - m3*p3
    while idx < num_phases:
        phase = phases[idx]
        # ...for primary minima.
        if phase < p1:
            modeled_flux_rel = b0
        # ...for primary ingress/egress.
        elif p1 <= phase and phase < p2:
            modeled_flux_rel = m1*phase + b1
        # ...for between minima.
        elif p2 <= phase and phase < p3:
            modeled_flux_rel = b2
        # ...for secondary ingress/egress.
        elif p3 <= phase and phase < p4:
            modeled_flux_rel = m3*phase + b3
        # ...for secondary minima.
        elif p4 <= phase:
            modeled_flux_rel = b4
        modeled_fluxes_rel[idx] = modeled_flux_rel
        idx += 1
    return modeled_fluxes_rel


@numba.jit(nopython=True)
def seg_log_prior(params):
    r"""Log prior of segmented, symmetric light curve model parameters up to a
    constant. Light curve is folded so that phase is from 0 to 0.5

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
    
    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta))
        If parameters are outside of acceptable range, `-numpy.inf`.
    
    See Also
    --------
    seg_are_valid_params, seg_model_fluxes_rel

    Notes
    -----
    * See `seg_model_fluxes_rel` for description of parameters.
    * This is an uninformative prior:
        `lnp = 0.0` if `theta` is within constraints.
        `lnp = -numpy.inf` otherwise.
    
    References
    ----------
    ..[1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    ..[2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    ..[3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if seg_are_valid_params(params=params):
        lnp = 0.0
    else:
        lnp = -np.inf
    return lnp


@numba.jit(nopython=True)
def seg_log_likelihood(params, phases, fluxes_rel):
    r"""Log likelihood of segmented, symmetric light curve model's relative flux
    values given phase values and model parameters. Log likelihood is calculated
    up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters, the last
        of which is the standard deviation of all measurements
        of relative flux `params = (..., flux_sigma)`.
        Unit is relative flux. This assumes that all measurements are
        drawn from the same distribution.
    phases : numpy.ndarray
        1D array of phases. Unit is decimal orbital phase.
    fluxes_rel : numpy.ndarray
        1D array of observed fluxes. Unit is relative integrated flux.
        Required: `len(fluxes_rel) == len(phases)`
        
    Returns
    -------
    lnp : float
        Log probability of relative flux values: ln(p(y|x, theta))
        If parameters are outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    seg_are_valid_params, seg_model_fluxes_rel
    
    Notes
    -----
    * See `seg_model_fluxes_rel` for description of parameters.

    References
    ----------
    .. [1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    .. [2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    .. [3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if seg_are_valid_params(params=params):
        # numba does not support negative indexing `params[-1]`
        sig = params[len(params)-1]
        modeled_fluxes_rel = seg_model_fluxes_rel(params=params, phases=phases)
        # Calculation for `lnp` adapted from [1]_.
        # All data are presumed to have the same sigma.
        log_term = np.log(2.0*np.pi*sig**2.0)
        idx = 0
        lnp = 0.0
        while idx < len(fluxes_rel):
            res = fluxes_rel[idx] - modeled_fluxes_rel[idx]
            res_term = (res / sig)**2.0
            lnp += -0.5*(log_term + res_term)
            idx += 1
    else:
        lnp = -np.inf
    return lnp


@numba.jit(nopython=True)
def seg_log_posterior(params, phases, fluxes_rel):
    r"""Log probability of segmented, symmetric light curve model parameters
    given the data. Log probability is calculated up to a constant.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
    phases : numpy.ndarray
        1D array of phases. Unit is decimal orbital phase.
    fluxes_rel : numpy.ndarray
        1D array of observed fluxes. Unit is relative integrated flux.
        Required: `len(fluxes_rel) == len(phases)`
        
    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta|x, y))
        If parameters are outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    seg_are_valid_params, seg_model_fluxes_rel,
    seg_log_prior, seg_log_likelihood

    Notes
    -----
    * See `seg_model_fluxes_rel` for description of parameters.

    References
    ----------
    .. [1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    .. [2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    .. [3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if seg_are_valid_params(params=params):
        # Calculation of `lnp` adapted from [1]_.
        lnpr = seg_log_prior(params=params)
        lnlike = seg_log_likelihood(
            params=params, phases=phases, fluxes_rel=fluxes_rel)
        lnp = lnpr + lnlike
    else:
        lnp = -np.inf
    return lnp


def read_quants_gianninas(fobj):
    """Read and parse custom file format of physical stellar parameters from
    Gianninas et al 2014, [1]_.
    
    Parameters
    ----------
    fobj : file object
        An opened file object to the text file with parameters.
        Example file format:
        line 0: 'Name         SpT    Teff   errT  log g errg '...
        line 1: '==========  ===== ======= ====== ===== ====='...
        line 2: 'J1600+2721  DA6.0   8353.   126. 5.244 0.118'...
    
    Returns
    -------
    dobj : collections.OrderedDict
        Ordered dictionary with parameter field names as keys and
        parameter field quantities as values.
        
    Examples
    --------
    >>> with open('path/to/file.txt', 'rb') as fobj:
    ...     dobj = read_quants_gianninas(fobj)
    
    References
    ----------
    .. [1] http://adsabs.harvard.edu/abs/2014ApJ...794...35G
    
    """
    # Read in lines of file and use second line (line number 1, 0-indexed)
    # to parse fields. Convert string values to floats. Split specific values
    # that have mixed types (e.g. '1.000 Gyr').
    lines = []
    for line in fobj:
        lines.append(line.strip())
    if len(lines) != 3:
        warnings.warn(
            ("\n" +
             "File has {num_lines}. File is expected to only have 3 lines.\n" +
             "Example file format:\n" +
             "line 0: 'Name         SpT    Teff   errT  log g'...\n" +
             "line 1: '==========  ===== ======= ====== ====='...\n" +
             "line 2: 'J1600+2721  DA6.0   8353.   126. 5.244'...").format(
                 num_lines=len(lines)))
    dobj = collections.OrderedDict()
    for mobj in re.finditer('=+', lines[1]):
        key = lines[0][slice(*mobj.span())].strip()
        value = lines[2][slice(*mobj.span())].strip()
        try:
            value = float(value)
        except ValueError:
            try:
                value = float(value.rstrip('Gyr'))
            except ValueError:
                pass
        if key == 'og L/L':
            key = 'log L/Lo'
        dobj[key] = value
    return dobj


def has_nans(obj):
    """Recursively iterate through an object to find a `numpy.nan` value.
    
    Parameters
    ----------
    obj : object
        Object may be a singleton. If the object has the '__iter__' attribute,
        nested objects such as `dict`, `list`, `tuple` are iterated through.
    
    Returns
    -------
    found_nan : bool
        If `True`, a `numpy.nan` value was found within `obj`.
        If `False`, no `numpy.nan` values were found within `obj`.
    
    """
    found_nan = False
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            for value in obj.itervalues():
                found_nan = has_nans(value)
                if found_nan:
                    break
        else:
            for item in obj:
                found_nan = has_nans(item)
                if found_nan:
                    break
    else:
        try:
            if np.isnan(obj):
                found_nan = True
        except TypeError:
            pass
    return found_nan


# TODO: speed up with numba
def model_geometry_from_light_curve(params, show_plots=False):
    """Calculate geometric parameters of a spherical binary
    model from light curve parameters.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model light curve parameters.
        `params = \
            (phase_rel_int, phase_rel_ext, light_oc, light_ref, light_tr)`.
        Units are:
        {phase_rel_*} = phases of external/internal
            events (tangencies) in decimal phase.
            int: internal tangencies, end/begin ingress/egress
            ext: external tangencies, begin/end ingress/egress
        {light_*} = relative flux
            oc:  occulatation event
            ref: between-eclipse reference light level
            tr:  transit event
    show_plots : {False, True}, bool, optional
        If False (default): Don't show plots of optimized fit for inclination.
        If True: Show plots of optimized fit for inclination.
    
    Returns
    -------
    geoms : tuple
        Tuple of floats as the geometric parameters of a spherical binary model.
        _s/_g denotes smaller/greater-radius star
        `geoms = \
            (flux_intg_rel_s, flux_intg_rel_g, radii_ratio_lt,
             incl_deg, radius_sep_s, radius_sep_g)`
        Units are:
        {flux_intg_rel_*} = relative integrated flux
        {radii_ratio_lt} = radius_s / radius_g from light levels
        {incl_deg} = orbital inclination in degrees
        {radius_sep_*} = radius in star-star separation distance
        
    See Also
    --------
    seg_model_fluxes_rel

    Notes
    -----
    - Eclipse light levels are referred to by transit or occultaton events
      of the smaller-radius star, not by "primary" or "secondary", which can
      depend on context. For an example, see [1]_.

    References
    ----------
    .. [1] https://github.com/ccd-utexas/binstarsolver/wiki/Supported_examples
    
    """
    # TODO: Check input.
    (phase_rel_int, phase_rel_ext,
     light_oc, light_ref, light_tr) = params
    phase_orb_int = 2.0*np.pi * phase_rel_int
    phase_orb_ext = 2.0*np.pi * phase_rel_ext
    (flux_intg_rel_s, flux_intg_rel_g) = \
        bss.utils.calc_fluxes_intg_rel_from_light(
            light_oc=light_oc, light_ref=light_ref)
    radii_ratio_lt = \
        bss.utils.calc_radii_ratio_from_light(
            light_oc=light_oc, light_tr=light_tr, light_ref=light_ref)
    incl_rad = \
        bss.utils.calc_incl_from_radii_ratios_phase_incl(
            radii_ratio_lt=radii_ratio_lt, phase_orb_ext=phase_orb_ext,
            phase_orb_int=phase_orb_int, tol=1e-4, maxiter=10,
            show_plots=show_plots)
    if incl_rad is np.nan:
        warnings.warn(
            ("\n" +
             "Inclination does not yield self-consistent solution.\n"))
        incl_rad = np.deg2rad(90.0)
    sep_proj_ext = \
        bss.utils.calc_sep_proj_from_incl_phase(
            incl=incl_rad, phase_orb=phase_orb_ext)
    sep_proj_int = \
        bss.utils.calc_sep_proj_from_incl_phase(
            incl=incl_rad, phase_orb=phase_orb_int)
    (radius_sep_s, radius_sep_g) = \
        bss.utils.calc_radii_sep_from_seps(
            sep_proj_ext=sep_proj_ext, sep_proj_int=sep_proj_int)
    incl_deg = np.rad2deg(incl_rad)
    geoms = \
        (flux_intg_rel_s, flux_intg_rel_g, radii_ratio_lt,
         incl_deg, radius_sep_s, radius_sep_g)
    return geoms


# TODO: speedup with @numba.jit(nopython=True)
def model_quants_from_velrs_lc_geoms(
    velr_s, velr_g, period, light_oc, light_ref, light_tr,
    radius_sep_s, radius_sep_g, incl_deg):
    r"""Calculate physical quantities of a spherical binary system model
    from its radial velocities, light curve paramters, and geometric parameters.
    The system is assumed to be an eclipsing double-line spectroscopic binary.

    Parameters
    ----------
    velr_s : float
    velr_g : float
        Semi-amplitude (half peak-to-peak) of radial velocities of
        smaller-radius (_s), greater-radius (_g) stars.
        Unit is barycentric kilometers per second.
    period : float
        Period of light curve model that best represents the time series data.
        Unit is seconds.
    light_oc  : float
    light_ref : float
    light_tr  : float
        Light level at occultation event (_oc), between eclipses (_ref),
        and at transit event (_tr). Unit is relative integrated flux.
    radius_sep_s : float
    radius_sep_g : float
        Stellar radius in star-star separation distance.
    incl_deg : float
        Orbital inclination. Angle between line of sight and the axis of the
        orbit. Unit is degrees.

    Returns
    -------
    quants : tuple
        Tuple of floats as the physical quantities of a spherical binary model.
        _s/_g denotes smaller/greater-radius star
        `quants = \
            (axis_s, axis_g, radius_s, radius_g, mass_s, mass_g,
             flux_rad_ratio)`
        Units are:
            {axis_*} : Semi-major axis of star's orbit. Unit is AU.
            {radius_*} : Radius of star. Unit is Rsun.
            {mass_*} : Mass of star. Unit is Msun.
            {flux_rad_ratio} : Ratio of radiative fluxes of smaller star to
                greater star. `flux_rad_ratio = flux_rad_s / flux_rad_g`
                Note: flux_rad = sigma*Teff**4 = Lstar/(4*pi*Rstar**2)

    See Also
    --------
    model_geometry_from_light_curve

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/ccd-utexas/binstarsolver/blob/
        master/examples/20150419T163000_binstarsolver_book_examples.ipynb

    """
    # Following [1]_.
    # TODO: Check input.
    # Convert input from astronomical units to MKS units.
    velr_s *= scipy_con.kilo
    velr_g *= scipy_con.kilo
    incl_rad = np.deg2rad(incl_deg)
    # Calculate semimajor axes.
    axis_s = bss.utils.calc_semimaj_axis_from_period_velr_incl(
        period=period, velr=velr_s, incl=incl_rad)
    axis_g = bss.utils.calc_semimaj_axis_from_period_velr_incl(
        period=period, velr=velr_g, incl=incl_rad)
    # Calculate stellar radii.
    sep = bss.utils.calc_sep_from_semimaj_axes(axis_1=axis_s, axis_2=axis_g)
    radius_s = bss.utils.calc_radius_from_radius_sep(
        radius_sep=radius_sep_s, sep=sep)
    radius_g = bss.utils.calc_radius_from_radius_sep(
        radius_sep=radius_sep_g, sep=sep)
    # Calculate stellar masses.
    mass_ratio = bss.utils.calc_mass_ratio_from_velrs(
        velr_1=velr_s, velr_2=velr_g)
    mass_sum = bss.utils.calc_mass_sum_from_period_velrs_incl(
        period=period, velr_1=velr_s, velr_2=velr_g, incl=incl_rad)
    (mass_s, mass_g) = bss.utils.calc_masses_from_ratio_sum(
        mass_ratio=mass_ratio, mass_sum=mass_sum)
    # Calculate ratio of radiative fluxes.
    flux_rad_ratio = bss.utils.calc_flux_rad_ratio_from_light(
        light_oc=light_oc, light_tr=light_tr, light_ref=light_ref)
    # Convert output from MKS units to astronomical units.
    axis_s   /= astropy_con.au.value
    axis_g   /= astropy_con.au.value
    radius_s /= astropy_con.R_sun.value
    radius_g /= astropy_con.R_sun.value
    mass_s   /= astropy_con.M_sun.value
    mass_g   /= astropy_con.M_sun.value
    return (axis_s, axis_g, radius_s, radius_g, mass_s, mass_g, flux_rad_ratio)


@numba.jit(nopython=True)
def rv_are_valid_params(params):
    r"""Check if parameters are valid for sine curve model of radial velocities.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
        `params = (rvel_amp, phase_offset, rvel_offset, rvel_sigma)`
        Units are:
            {rvel_*} = radial velocity in km/s
            {phase_*} = decimal orbital phase

    Returns
    -------
    are_valid : bool
        True if all of the following hold:
            If -1 <= `phase_offset` <= 1
            If 0 < `rvel_sigma`
        False otherwise.

    See Also
    --------
    rv_model_radial_velocities

    Notes
    -----
    * See `rv_model_radial_velocities` for description of paramters.
    * Return `bool` rather than raise exception for optimization with `numba`.

    """
    # Negative phase_offset incase data have phase_offset like 0.00+/-0.01.
    # Allow arbitrary radial velocitity values to model both primary and
    # secondary stars.
    (_, phase_offset, _, rvel_sigma) = params
    if ((-1.0 <= phase_offset) and (phase_offset <= 1.0) and
        (0.0 < rvel_sigma)):
        are_valid = True
    else:
        are_valid = False
    return are_valid


@numba.jit(nopython=True)
def rv_model_radial_velocities(params, phases):
    r"""Calculate the radial velocities for a sine model of the radial velocity
    curve.

    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
        `params = (rvel_amp, phase_offset, rvel_offset, rvel_sigma)`
            rvel_amp: Radial velocity amplitude of the sine model.
            phase_offset: Orbital phase offset of the sine model.
            rvel_offset: Radia velocity offset of the sine model.
            rvel_sigma: Standard deviation of the radial velocities. Assumes
                that all radial velocities are drawn from the same distribution.
        Units are:
            {rvel_*} = radial velocity in km/s
            {phase_*} = decimal orbital phase
    phases : numpy.ndarray
        1D array of orbital phases. Units are decimal orbital phase.
        0 <= `phase` <= 1.0.

    Returns
    -------
    rvels : numpy.ndarray
        1D array of modeled radial velocities. Units are km/s.

    See Also
    --------
    rv_are_valid_params

    Notes
    -----
    * Requires that all input parameters are already checked as valid.
    * Sine model of radial velocities:
        rvel = rvel_amp * sin(2*pi(phase + phase_offset)) + rvel_offset

    """
    (rvel_amp, phase_offset, rvel_offset, _) = params
    num_phases = len(phases)
    rvels = np.empty(num_phases)
    idx = 0
    while idx < num_phases:
        phase = phases[idx]
        rvel = (
            (rvel_amp * np.sin(2.0*np.pi * (phase + phase_offset))) + 
            rvel_offset)
        rvels[idx] = rvel
        idx += 1
    return rvels


@numba.jit(nopython=True)
def rv_log_prior(params):
    r"""Log prior of sine model for radial velocities up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.

    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta))
        If parameters are outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    rv_are_valid_params, rv_model_radial_velocities

    Notes
    -----
    * See `rv_model_radial_velocities` for description of parameters.
    * This is an uninformative prior:
        `lnp = 0.0` if `theta` is within constraints.
        `lnp = -numpy.inf` otherwise.

    References
    ----------
    ..[1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    ..[2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    ..[3] http://dan.iel.fm/emcee/current/user/line/

    """
    if rv_are_valid_params(params=params):
        lnp = 0.0
    else:
        lnp = -np.inf
    return lnp


@numba.jit(nopython=True)
def rv_log_likelihood(params, phases, rvels):
    r"""Log likelihood of sine model's radial velocities given phases and model
    parameters. Log likelihood is calculated up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters, the last of which is the
        standard deviation of all measuremetns of radial velocity
        `params = (..., rvel_sigma)`. Unit is km/s. This assumes that all
        measurements are drawn from the same distribution.
    phases : numpy.ndarray
        1D array of phases. Unit is decimal orbital phase.
    rvels : numpy.ndarray
        1D array of observed radial velocities. Unit is km/s.
        Required: `len(rvels) = len(phases)`

    Returns
    -------
    lnp : float
        Log probability of radial velocity values: ln(p(y|x, theta))
        If parameters are outside of acceptable range, `-numpy.inf`.

    Notes
    -----
    * See `rv_model_fluxes_rel` for description of parameters.

    References
    ----------
    .. [1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    .. [2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    .. [3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    pass


@numba.jit(nopython=True)
def rv_log_posterior(params, phases, rvels):
    r"""Log probability of sine model's parameters givne the data. Log
    probability is calculated up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats as the model parameters.
    phases : numpy.ndarray
        1D array of phases. Unit is decimal orbital phase.
    rvels : numpy.ndarray
        1D array of observed radial velocities. Unit is km/s.
        Required: `len(rvels) == len(phases)`

    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta|x, y))
        If parameters are outside of acceptable range, `-numpy.inf`.

    See Also
    --------
    rv_are_valid_params, rv_model_radial_velocities,
    rv_log_prior, rv_log_likelihood

    Notes
    -----
    * See `rv_model_radial_velocities` for description of parameters.

    References
    ----------
    .. [1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    .. [2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    .. [3] http://dan.iel.fm/emcee/current/user/line/

    """
    if rv_are_valid_params(params=params):
        # Calculation of `lnp` adapted from [1]_.
        lnpr = rv_log_prior(params=params)
        lnlike = rv_log_likelihood(params=params, phases=phases, rvels=rvels)
        lnp = lnpr + lnlike
    else:
        lnp = -np.inf
    return lnp
