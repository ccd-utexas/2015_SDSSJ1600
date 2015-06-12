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
import binstarsolver as bss
import gatspy.periodic as gastpy_per
import matplotlib.pyplot as plt
import numba
import numpy as np
import seaborn as sns


# Set environment
sns.set() # Set matplotlib styles by seaborn.


def calc_period_limits(times):
    r"""Calculate the region of dectable periods.
    
    Paramters
    ---------
    times : numpy.ndarray
        1D array of time coordinates for data.
        Unit is time, e.g. seconds or days.
    
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
    med_sampling_period = np.median(np.diff(times))
    acquisition_duration = max(times) - min(times)
    min_period = 2.0 * med_sampling_period
    max_period = 0.5 * acquisition_duration
    antialias_factor = 2.56
    num_periods = \
        int(antialias_factor * acquisition_duration / med_sampling_period)
    return (min_period, max_period, num_periods)


def plot_periodogram(
    periods, powers, xscale='log', period_unit='seconds',
    flux_unit='relative', legend=False, return_ax=False):
    r"""Plot the periods and relative powers for a multiband generalized
    Lomb-Scargle periodogram. Convenience function for methods from
    [1]_, [2]_.

    Parameters
    ----------
    periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds or days.
    powers : numpy.ndarray
        1D array of powers. Unit is relative Lomb-Scargle power spectral density
        from flux and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
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
    legend : {False, True}, bool, optional
        If `False` (default), do not show a legend.
        If `True`, label the periodogram plot and create a legend.
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
    plot_phased_light_curve

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] VanderPlas and Ivezic, 2015,
           http://adsabs.harvard.edu/abs/2015arXiv150201344V
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, xscale=xscale)
    if legend:
        label = 'relative L-S PSD'
    else:
        label = None
    ax.plot(periods, powers, marker='.', label=label)
    if legend:
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


def calc_sig_levels(
    model, sigs=(95.0, 99.0, 99.9), num_periods=20, num_shuffles=1000):
    r"""Calculate relative powers that correspond to significance levels for
    a multiband generalized Lomb-Scargle periodogram. The noise is modeled by
    shuffling the time series. Convenience function for methods from [1]_, [2]_.

    Parameters
    ----------
    model : gatspy.periodic.LombScargleMultiband
        Instance of multiband generalized Lomb-Scargle light curve model
        from `gatspy`.
    sigs : {(95.0, 99.0, 99.9)}, tuple, optional
        `tuple` of `float` that are the levels of statistical significance.
    num_periods : {20}, int, optional
        Number of periods at which to compute significance levels.
        The significance level changes slowly as a function of period.
    num_shuffles : {1000}, int, optional
        Number of shuffles used to compute significance levels.
        For 1000 shuffles, the significance level can be computed to a max
        resolution of 0.1 (i.e. 99.9 significance can be computed, not 99.99).

    Returns
    -------
    sig_periods : numpy.ndarray
        Periods at which significance levels were computed.
    sig_powers : dict
        `dict` of `numpy.ndarray`. Keys are `sigs`. Values are relative powers
        for each significance level as a `numpy.ndarray`.

    See Also
    --------
    gatspy.periodic.LombScargleMultiband, calc_period_limits

    Notes
    -----
    - Significance levels are calculated periods within
        `model.optimizer.period_range`
    - For a given period, a power is "signficant to the 99th percentile" if
        that power is greater than 99% of all other powers due to noise at that
        period. The noise is modeled by shuffling the time series.
    - Period space is sampled linearly in angular frequency.
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
    # Check input.
    # Period space is sampled linearly in angular frequency.
    noise_model = copy.deepcopy(model)
    (min_period, max_period) = noise_model.optimizer.period_range
    min_omega = 2.0*np.pi / max_period
    max_omega = 2.0*np.pi / min_period
    num_omegas = num_periods
    sig_omegas = \
        np.linspace(
            start=min_omega, stop=max_omega, num=num_omegas, endpoint=True)
    sig_periods = 2.0*np.pi / sig_omegas
    # Calculate percentiles of powers from noise.
    np.random.seed(seed=0) # for reproducibility
    sig_powers_arr = []
    for _ in xrange(num_shuffles):
        np.random.shuffle(noise_model.t)
        sig_powers_arr.append(noise_model.periodogram(periods=sig_periods))
    sig_powers = \
        {sig: np.percentile(a=sig_powers_arr, q=sig, axis=0) for sig in sigs}
    return (sig_periods, sig_powers)


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
            t=phased_times_fit, filts=[filt]*1000, period=best_period)
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
    if not min_flux <= min_flux_init:
        raise AssertionError(
            ("Program error.\n" +
             "Required: `min_flux` <= `min_flux_init`\n" +
             "min_flux = {mf}\n" +
             "min_flux_init = {mfi}").format(
                mf=min_flux, mfi=min_flux_init))
    return min_flux_time


def calc_phases(times, best_period, min_flux_time=0.0):
    r"""Calculate phases of a light curve.

    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates of observed data.
        Units are time, e.g. seconds or days.
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
        Time coordinate. Units is time, e.g. seconds or days.
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
    flux_unit='relative', legend=False, return_ax=False):
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
    legend : {False, True}, bool, optional
        If `False` (default), do not show a legend.
        If `True`, label the periodogram plot and create a legend.
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
    if legend:
        (label_obs, label_fit) = ('observed', 'fit')
    else:
        (label_obs, label_fit) = (None, None)
    plt_kwargs = dict(marker='.', linestyle='', linewidth=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(
        tess_phases, tess_fluxes, tess_fluxes_err, ecolor='gray',
        label=label_obs, **plt_kwargs)
    ax.plot(tess_fit_phases, tess_fit_fluxes, label=label_fit, **plt_kwargs)
    if legend:
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


@numba.jit
def are_valid_params(params):
    """Check if parameters are valid.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
        `params = (p1, p2, b0, b2, b4, sig)`.
        Units are:
        {p1, p2} = decimal orbital phase
        {b0, b2, b4, sig} = relative flux

    Returns
    -------
    are_valid : bool
        True if all of the following hold:
            If input phase parameters are all between 0 and 0.5, inclusive.
            If input p1 is less than p2.
            If sigma is not greater than 0.
        False otherwise.
    
    See Also
    --------
    model_flux_rel

    """
    # Allow arbitrary flux values for flexibility.
    # Some models may require flux between minima >~ 1 as a normalization factor
    # (e.g. Ch 7 of Budding 2007).
    (p1, p2, b0, b2, b4, sig) = params
    if ((p1 < p2) and
        (0.0 <= p1 and p1 <= 0.5) and
        (0.0 <= p2 and p2 <= 0.5) and
        (sig > 0)):
        are_valid = True
    else:
        are_valid = False
    return are_valid

@numba.jit
def model_flux_rel(params, phase):
    """Model of folded eclipse light curve.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
        `params = (p1, p2, b0, b2, b4, sig)`.
        Units are:
        {p1, p2} = decimal orbital phase
        {b0, b2, b4, sig} = relative flux
    phase : float
        Eclipse phase. 0 <= `phase` <= 0.5.
        Unit is decimal orbital phase.
        
    Returns
    -------
    flux_rel : float
        Modeled relative flux. Unit is relative integrated flux.
    
    Notes
    -----
    - Trapezoidal model for folded light curve:
        Durations of primary minimum and secondary minimum are equal.
        Durations of primary ingress/egress and secondary ingress/egress are equal.
        Light curve is segmented into functions f(x).
        x values are `phase`, p0, p1, ...
        light      |  |  |--------|  |  |
        curve:     |  | /|        |\\|__|
                   |__|/ |        |  |  |
        function:  |f0|f1|   f2   |f3|f4|
        phase:   0.0  p1 p2       p3 p4 0.5
        primary minima:           f0(x)  = b0; 0.0 <= x < p1
        boundary condition:       f0(p1) = f1(p1)
        primary ingress/egress:   f1(x)  = m1*x + b1; p1 <= x < p2
        boundary condition:       f1(p2) = f2(p2)
        between minima:           f2(x)  = b2; p2 <= x < p3
        boundary condition:       f2(p3) = f3(p3)
        secondary ingress/egress: f3(x)  = m3*x + b3; p3 <= x < p4
        boundary condition:       f3(p4) = f4(p4)
        secondary minima:         f4(x)  = b4; p4 <= x <= 0.5
        boundary condition:       p4 = 0.5 - p1
        boundary condition:       p3 = 0.5 - p2
    - Model parameters are defined relative to primary minima when possible since
        primary minima are deeper and easier to detect from observations.
    - `sig` is the standard deviation of all measurements of relative flux, which
        assumes they are all drawn from the same distribution.
    
    Raises
    ------
    - ValueError: If parameters are not valid (see `are_params_valid`)
    
    See Also
    --------
    are_params_valid
    
    """
    # Check input.
    # NOTE: commented out for numba
    #if not are_valid_params(params=params):
    #    raise ValueError(
    #        ("`params` are not valid:\n" +
    #         "params = {params}").format(params=params))
    # Compute modeled relative flux...
    (p1, p2, b0, b2, b4, sig) = params
    p4 = 0.5 - p1
    p3 = 0.5 - p2
    # ...for primary minima.
    if phase < p1:
        flux_rel = b0
    # ...for primary ingress/egress.
    elif p1 <= phase and phase < p2:
        m1 = (b2 - b0)/(p2 - p1)
        b1 = b0 - m1*p1
        flux_rel = m1*phase + b1
    # ...for between minima.
    elif p2 <= phase and phase < p3:
        flux_rel = b2
    # ...for secondary ingress/egress.
    elif p3 <= phase and phase < p4:
        m3 = (b4 - b2)/(p4 - p3)
        b3 = b2 - m3*p3
        flux_rel = m3*phase + b3
    # ...for secondary minima.
    elif p4 <= phase:
        flux_rel = b4
    # NOTE: commented out for numba
    #else:
    #    raise AssertionError(
    #        "Program error. `phase` was not classified as an eclipse event.")
    return flux_rel


def log_prior(params):
    """Log prior of light curve model parameters up to a constant.
    Light curve is folded so that phase is from 0 to 0.5
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
        `params = (p1, p2, b0, b2, b4, sig)`.
        Units are:
        {p1, p2} = decimal orbital phase
        {b0, b2, b4, sig} = relative flux
    
    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta))
        If parameters are outside of acceptible range, `-numpy.inf`.
    
    Notes
    -----
    - See `model_flux_rel` for description of parameters.
    - This is an uninformative prior:
        `lnp = 0.0` if `theta` is within above constraints.
        `lnp = -numpy.inf` otherwise.
    
    See Also
    --------
    model_flux_rel
    
    References
    ----------
    ..[1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    ..[2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    ..[3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if are_valid_params(params=params):
        lnp = 0.0
    else:
        lnp = -np.inf
    return lnp


def log_likelihood(params, phase, flux_rel):
    """Log likelihood of light curve relative flux values given
    phase values and light curve model parameters. Log likelihood
    is computed up to a constant.

    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters, the last
        of which is the standard deviation of all measurements
        of relative flux `params = (..., sig)`. Unit is relative flux.
        This assumes that all measurements are drawn from the same
        distribution.
    phase : float
        Eclipse phase. Unit is decimal orbital phase.
    flux_rel : float
        Light level at `phase`. Unit is relative integrated flux.
        
    Returns
    -------
    lnp : float
        Log probability of relative flux values: ln(p(y|x, theta))
        If parameters outside of acceptible range, `-numpy.inf`.
    
    Notes
    -----
    - See `model_flux_rel` for description of parameters.

    References
    ----------
    ..[1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    ..[2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    ..[3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if are_valid_params(params=params):
        sig = params[-1]
        modeled_flux_rel = \
            map(lambda phs: model_flux_rel(phase=phs, params=params),
                phase)
        lnp = -0.5 * np.sum(np.log(2*np.pi*sig**2) +
                            ((flux_rel - modeled_flux_rel)**2 / sig**2))
    else:
        lnp = -np.inf
    return lnp


def log_posterior(params, phase, flux_rel):
    """Log probability of light curve model parameters
    given the data. Log probability is computed
    up to a constant.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model parameters.
    phase : float
        Eclipse phase. Unit is decimal orbital phase.
    flux_rel : float
        Light level at `phase`. Unit is relative integrated flux.
        
    Returns
    -------
    lnp : float
        Log probability of parameters: ln(p(theta|x,y))
        If parameters outside of acceptible range, `-numpy.inf`.

    Notes
    -----
    - See `model_flux_rel` for description of parameters.

    References
    ----------
    ..[1] Vanderplas, 2014. http://arxiv.org/pdf/1411.5018v1.pdf
    ..[2] Hogg, et al, 2010. http://arxiv.org/pdf/1008.4686v1.pdf
    ..[3] http://dan.iel.fm/emcee/current/user/line/
    
    """
    if are_valid_params(params=params):
        lnpr = log_prior(params=params)
        lnlike = log_likelihood(phase=phase,
                                flux_rel=flux_rel,
                                params=params)
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


def model_geometry_from_light_curve(params, show_plots=False):
    """Calculate geometric parameters of a spherical binary
    model from light curve parameters.
    
    Parameters
    ----------
    params : tuple
        Tuple of floats representing the model light curve parameters.
        `params = \
            (phase_orb_int, phase_orb_ext,
             light_oc, light_ref, light_tr, sig)`.
        Units are:
        {phase_orb_int/ext} = phase of external/internal
            events (tangencies) in radians
            int: internal tangencies, end/begin ingress/egress
            ext: external tangencies, begin/end ingress/egress
        {light_oc/ref/tr, sig} = relative flux
            oc:  occulatation event
            ref: between-eclipse reference light level
            tr:  transit event
    show_plots : {False, True}, bool, optional
        If False (default): Don't show plots of optimized fit for inclination.
        If True: Show plots of optimized fit for inclination.
    
    Returns
    -------
    geoms : tuple
        Tuple of floats representing the geometric parameters
        of a spherical binary model from light curve values.
        _s/_g denotes smaller/greater-radius star
        geoms = \
            (flux_intg_rel_s, flux_intg_rel_g, radii_ratio_lt,
             incl_rad, radius_sep_s, radius_sep_g)
        Units are:
        {flux_intg_rel_s, flux_intg_rel_g} = relative integrated flux
        {radii_ratio_lt} = radius_s / radius_g from light levels
        {incl_rad} = orbital inclination in radians
        {radius_sep_s, radius_sep_g} = radius in star-star separation distance
        
    See Also
    --------
    model_flux_rel, model_quantities_from_lc_velr_stellar

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
    (phase_orb_int, phase_orb_ext,
     light_oc, light_ref, light_tr, sig) = params
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
        incl_rad = np.deg2rad(90)
    sep_proj_ext = \
        bss.utils.calc_sep_proj_from_incl_phase(
            incl=incl_rad, phase_orb=phase_orb_ext)
    sep_proj_int = \
        bss.utils.calc_sep_proj_from_incl_phase(
            incl=incl_rad, phase_orb=phase_orb_int)
    (radius_sep_s, radius_sep_g) = \
        bss.utils.calc_radii_sep_from_seps(
            sep_proj_ext=sep_proj_ext, sep_proj_int=sep_proj_int)
    geoms = \
        (flux_intg_rel_s, flux_intg_rel_g, radii_ratio_lt,
         incl_rad, radius_sep_s, radius_sep_g)
    return geoms


def model_quantities_from_lc_velr_stellar(
    phase0, period, lc_params, velr_b, stellar_b):
    """Calculate physical quantities of a spherical binary system model
    from its light curve parameters, radial velocity of the brighter star,
    and a stellar model of the brighter star modeled from a spectrum.
    The system is assumed to be an eclipsing single-line spetroscopic binary.
    
    Parameters
    ----------
    phase0 : float
    period : float
        TODO: define `period`, `phase0` from `lc_params`
    lc_params : tuple
        Tuple of floats representing the model light curve parameters.
        `lc_params = \
            (phase_orb_int, phase_orb_ext,
             light_oc, light_ref, light_tr, sig)`.
        Units are:
        {phase_orb_int/ext} = phase of external/internal
            events (tangencies) in radians
            int: internal tangencies, end/begin ingress/egress
            ext: external tangencies, begin/end ingress/egress
        {light_oc/ref/tr, sig} = relative flux
            oc:  occulatation event
            ref: between-eclipse reference light level
            tr:  transit event
    velr_b : float
        Semi-amplitude (half peak-to-peak) of radial velocity
        of the brighter star (greater integrated flux).
        Unit is meters/second.
    stellar_b : tuple
        Tuple of floats representing the parameters of a stellar model
        that was fit to the brighter star (greater integrated flux) from
        single-line spectroscopy of the system.
        `stellar_b = (mass_b, radius_b, teff_b)`
        Units are MKS:
        {mass} = stellar mass in kg
        {radius} = stellar radius in meters
        {teff} = stellar effective temperature in Kelvin
    
    Returns
    -------
    quants : tuple
        Tuple of floats representing the physical quantities
        of a spherical binary model from geometric parameters.
        `quants = \
            (# Quantities for the entire binary system
             phase0, period, incl_rad, sep, massfunc,
             # Quantities for the smaller-radius star ('_s')
             velr_s, axis_s, mass_s, radius_s, teff_s,
             # Quantities for the greater-radius star ('_g')
             velr_g, axis_g, mass_g, radius_g, teff_g)`
        Units are MKS:
        {phase0} = time at which phase of orbit is 0 in
            Unixtime Barycentric Coordinate Time
        {period} = period of orbit in seconds
        {incl_rad} = orbital inclination in radians
        {sep} = star-star separation distance in meters
        {massfunc} = mass function of system in kg
            massfunc = (m2 * sin(i))**3 / (m1 + m2)**2
            where star 1 is the brighter star
        {velr} = radial velocity amplitude (half peak-to-peak) in m/s
        {axis} = semimajor axis of star's orbit in meters
        {radius} = stellar radius in meters
        {mass} = stellar mass in kg
        {teff} = stellar effective temperature in Kelvin

    See Also
    --------
    model_geometry_from_light_curve, model_flux_rel

    Notes
    -----
    * Eclipse light levels are referred to by transit or occultaton events
      of the smaller-radius star, not by "primary" or "secondary", which can
      depend on context. For an example, see [1]_.
    * Quantities are calculated as follows:
      * The system geometry is modeled from `lc_params`. The relative
        integrated fluxes of the stars determine whether brighter/dimmer
        star ('_b'/'_d') has the smaller/greater radius ('_s'/'_g').
      * `phase0, period, incl_rad`: Defined from `lc_params`.
      * `massfunc`: Calculated from `lc_params`, `velr_b`.
      * `mass_b`, `radius_b`, `teff_b`: Defined from `stellar_b`.
      * `axis_b`: Calculated from `lc_params`, `velr_b`.
      * `mass_d`, `velr_d`, `axis_d`: Calculated from
        `lc_params`, `velr_b`, `stellar_b`
      * `sep`: Calculated from `lc_params`, `velr_b`, `stellar_b`.
      * `radius_d`, `teff_d`: Calculated from
        `lc_params`, `velr_b`, `stellar_b`
      * From `lc_params`, the ratio of radii as determined by light levels may
        be different from that determined by timings if there was no
        self-consistent solution for inclination.

    References
    ----------
    .. [1] https://github.com/ccd-utexas/binstarsolver/wiki/Supported_examples
    .. [2] Budding, 2007, "Introduction to Astronomical Photometry"

    """
    ########################################
    # Check input and define and compute physical quantities.
    # Quantities for:
    #     brighter star: '_b'
    #     dimmer star: '_d'
    #     smaller-radius star: '_s'
    #     greater-radius star: '_g'
    # Brightness is total integrated flux (total luminosity).
    # TODO: Insert check input here.
    # For system:
    #     From light curve:
    #         Define the phase, period, inclination.
    #     From light curve and radial velocity:
    #         Calculate the mass function.
    # TODO: get phase0 and period from lc_params.
    (phase_orb_int, phase_orb_ext,
     light_oc, light_ref, light_tr, _) = lc_params
    time_begin_ingress = -phase_orb_ext*period / (2.0*np.pi)
    time_end_ingress   = -phase_orb_int*period / (2.0*np.pi)
    time_begin_egress  = -time_end_ingress
    (flux_intg_rel_s, flux_intg_rel_g, radii_ratio_lt,
     incl_rad, radius_sep_s, radius_sep_g) = \
        model_geometry_from_light_curve(params=lc_params, show_plots=False)
    massfunc = \
        bss.utils.calc_mass_function_from_period_velr(
            period=period, velr1=velr_b)
    # For brighter star:
    #     From stellar model:
    #         Define the mass, radius, temperature.
    #     From light curve, radial velocity:
    #         Calculate the semi-major axis.
    (mass_b, radius_b, teff_b) = stellar_b
    axis_b = \
        bss.utils.calc_semimaj_axis_from_period_velr_incl(
            period=period, velr=velr_b, incl=incl_rad)
    # For dimmer star:
    #     From light curve, radial velocity, stellar model:
    #         Calculate the mass.
    #         Calculate the radial velocity.
    #         Calculate the semi-major axis.
    mass_d = \
        bss.utils.calc_mass2_from_period_velr1_incl_mass1(
            period=period, velr1=velr_b, incl=incl_rad, mass1=mass_b)
    velr_d = \
        bss.utils.calc_velr2_from_masses_period_incl_velr1(
            mass1=mass_b, mass2=mass_d, velr1=velr_b,
            period=period, incl=incl_rad)
    axis_d = \
        bss.utils.calc_semimaj_axis_from_period_velr_incl(
            period=period, velr=velr_d, incl=incl_rad)
    # For system:
    #     From light curve, radial velocity, stellar model:
    #         Calculate the star-star separation distance.
    sep = \
        bss.utils.calc_sep_from_semimaj_axes(
            axis_1=axis_b, axis_2=axis_d)
    # Use relative integrated fluxes to determine whether brighter/dimmer star
    # has smaller/greater radius.
    # If smaller-radius star is brighter than the greater-radius star, then the
    # parameters from the radial velocities and the stellar model refer to the
    # smaller-radius star. Otherwise, the parameters refer to the
    # greater-radius star. 
    if flux_intg_rel_s >= flux_intg_rel_g:
        smaller_is_brighter = True
    else:
        smaller_is_brighter = False
    # Assign quantities to respective stars.
    if smaller_is_brighter:
        velr_s = velr_b
        (mass_s, radius_s, teff_s) = (mass_b, radius_b, teff_b)
        axis_s = axis_b
        (mass_g, velr_g, axis_g) = (mass_d, velr_d, axis_d)
    else:
        velr_g = velr_b
        (mass_g, radius_g, teff_g) = (mass_b, radius_b, teff_b)
        axis_g = axis_b
        (mass_s, velr_s, axis_s) = (mass_d, velr_d, axis_d)
    # For dimmer star:
    #     From light curve, radial velocity, stellar model:
    #         Calculate the radius, effective temperature.
    # NOTE: Ratios below are quantities of
    # smaller-radius star / greater-radius star
    radii_ratio_sep = radius_sep_s / radius_sep_g
    flux_rad_ratio = \
        bss.utils.calc_flux_rad_ratio_from_light(
            light_oc=light_oc, light_tr=light_tr, light_ref=light_ref)
    teff_ratio = \
        bss.utils.calc_teff_ratio_from_flux_rad_ratio(
            flux_rad_ratio=flux_rad_ratio)
    if smaller_is_brighter:
        radius_g = radius_s / radii_ratio_sep
        teff_g = teff_s / teff_ratio
    else:
        radius_s = radius_g * radii_ratio_sep
        teff_s = teff_g * teff_ratio
    ########################################
    # Check calculations and return.
    # Check that the masses are calculated consistently.
    assert (mass_d >= massfunc)
    assert np.isclose(
        mass_s / mass_g,
        bss.utils.calc_mass_ratio_from_velrs(
            velr_1=velr_s, velr_2=velr_g))
    assert np.isclose(
        mass_s + mass_g,
        bss.utils.calc_mass_sum_from_period_velrs_incl(
            period=period, velr_1=velr_s, velr_2=velr_g, incl=incl_rad))
    # Check that the semi-major axes are calculated consistently.
    assert np.isclose(sep, axis_s + axis_g)
    # Check that the radii are calculated consistently.
    # There may be a difference if there was no self-consistent solution
    # for inclination.
    assert radius_s <= radius_g
    rtol = 1e-1
    try:
        assert np.isclose(radii_ratio_lt, radii_ratio_sep, rtol=rtol)
    except AssertionError:
        warnings.warn(
            ("\n" +
             "Radii ratios do not agree to within rtol={rtol}.\n" +
             "The solution for inclination from the light curve\n" +
             "may not be self-consistent:\n" +
             "    radii_ratio_lt              = {rrl}\n" +
             "    radius_sep_s / radius_sep_g = {rrs}").format(
                 rtol=rtol,
                 rrl=radii_ratio_lt,
                 rrs=radii_ratio_sep))
    try:
        radius_s_from_velrs_times = \
            bss.utils.calc_radius_from_velrs_times(
                velr_1=velr_s, velr_2=velr_g,
                time_1=time_begin_ingress, time_2=time_end_ingress)
        radius_s_from_radius_sep = \
            bss.utils.calc_radius_from_radius_sep(
                radius_sep=radius_sep_s, sep=sep)
        radius_g_from_velrs_times = \
            bss.utils.calc_radius_from_velrs_times(
                velr_1=velr_s, velr_2=velr_g,
                time_1=time_begin_ingress, time_2=time_begin_egress)
        radius_g_from_radius_sep = \
            bss.utils.calc_radius_from_radius_sep(
                radius_sep=radius_sep_g, sep=sep)
        assert np.isclose(
            radius_s, radius_s_from_velrs_times, rtol=rtol)
        assert np.isclose(
            radius_s, radius_s_from_radius_sep, rtol=rtol)
        assert np.isclose(
            radius_s_from_velrs_times, radius_s_from_radius_sep, rtol=rtol)
        assert np.isclose(
            radius_g, radius_g_from_velrs_times, rtol=rtol)
        assert np.isclose(
            radius_g, radius_g_from_radius_sep, rtol=rtol)
        assert np.isclose(
            radius_g_from_velrs_times, radius_g_from_radius_sep, rtol=rtol)
    except AssertionError:
        warnings.warn(
            ("\n" +
             "Radii computed from the following methods do not agree\n" +
             "to within rtol={rtol}. Units are meters:\n" +
             "    radius_s                  = {rs:.2e}\n" +
             "    radius_s_from_velrs_times = {rs_vt:.2e}\n" +
             "    radius_s_from_radius_sep  = {rs_rs:.2e}\n" +
             "    radius_g                  = {rg:.2e}\n" +
             "    radius_g_from_velrg_times = {rg_vt:.2e}\n" +
             "    radius_g_from_radius_sep  = {rg_rs:.2e}").format(
                 rtol=rtol,
                 rs=radius_s,
                 rs_vt=radius_s_from_velrs_times,
                 rs_rs=radius_s_from_radius_sep,
                 rg=radius_g,
                 rg_vt=radius_g_from_velrs_times,
                 rg_rs=radius_g_from_radius_sep))
    quants = \
        (phase0, period, incl_rad, sep, massfunc,
         velr_s, axis_s, mass_s, radius_s, teff_s,
         velr_g, axis_g, mass_g, radius_g, teff_g)
    return quants
