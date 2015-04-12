"""Utilities for reproducing Harrold et al 2015 on SDSS J160036.83+272117.8.

TODO: remove resolutions. resolution is f1 - f0 or p1 - p0.

"""


# Standard libraries.
from __future__ import absolute_import, division, print_function
import sys
import warnings
# Third-party installed packages.
import astroML.time_series as astroML_ts
import matplotlib.pyplot as plt
import numpy as np


def plot_periodogram(periods, powers, xscale='log', n_terms=1,
                     period_unit='seconds', flux_unit='relative', return_ax=False):
    """Plot the periods and powers for a generalized Lomb-Scargle
    periodogram. Convenience function for plot formats from [1]_.

    Parameters
    ----------
    periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds or days.
    powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density from flux and angular frequency,
        e.g. from relative flux, angular frequency 2*pi/seconds.
    xscale : {'log', 'linear'}, string, optional
        `matplotlib.pyplot` attribute to plot periods x-scale in 'log' (default) or 'linear' scale.
    n_terms : {1}, int, optional
        Number of Fourier terms used to fit the light curve for labeling the plot.
        Example: n_terms=1 will label the title with
        "Generalized Lomb-Scargle periodogram\nwith 1 Fourier terms fit"
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label the x-axis with "Period (seconds)"
        and label the y-axis with "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    return_ax : {False, True}, bool
        If `False` (default), show the periodogram plot. Return `None`.
        If `True`, return a `matplotlib.axes` instance for additional modification.

    Returns
    -------
    ax : matplotlib.axes
        Returned only if `return_ax` is `True`. Otherwise returns `None`.

    TODO
    ----
    TODO: Create test from astroml book.

    References
    ----------
    .. [1] Ivezic et al, 2014, Statistics, Data Mining, and Machine Learning in Astronomy
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, xscale=xscale)
    ax.plot(periods, powers, color='black', linewidth=1)
    ax.set_xlim(min(periods), max(periods))
    ax.set_title(("Generalized Lomb-Scargle periodogram\n" +
                  "with {num} Fourier terms fit").format(num=n_terms))
    ax.set_xlabel("Period ({punit})".format(punit=period_unit))
    ax.set_ylabel(
        ("Lomb-Scargle Power Spectral Density\n" +
         "(from flux in {funit}, ang. freq. in 2*pi/{punit})").format(
            funit=flux_unit, punit=period_unit))
    if return_ax:
        return_obj = ax
    else:
        plt.show()
        return_obj = None
    return return_obj


def calc_periodogram(times, fluxes, fluxes_err, min_period=None, max_period=None, num_periods=None,
                     sigs=(95.0, 99.0), num_bootstraps=100, show_periodogram=True,
                     period_unit='seconds', flux_unit='relative'):
    """Calculate periods, powers, and significance levels using generalized Lomb-Scargle periodogram.
    Convenience function for methods from [1]_.
       
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data. Unit is time, e.g. seconds or days.
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux, e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
    min_period : {None}, float, optional
        Minimum period to sample. Unit is same as `times`.
        If `None` (default), `min_period` = 2x median sampling period,
        (the Nyquist limit from [2]_).
    max_period : {None}, float, optional
        Maximum period to sample. Unit is same as `times`.
        If `None` (default), `max_period` = 0.5x acquisition time,
        (1 / (2x the frequency resolution) adopted from [2]_).
    num_periods : {None}, int, optional
        Number of periods to sample, including `min_period` and `max_period`.
        If `None` (default), `num_periods` = minimum of 
        acquisition time / median sampling period (adopted from [2]_), or
        1e4  (to limit computation time).
    sigs : {(95.0, 99.0)}, tuple of floats, optional
        Levels of statistical significance for which to compute corresponding
        Lomb-Scargle powers via bootstrap analysis.
    num_bootstraps : {100}, int, optional
        Number of bootstrap resamplings to compute significance levels.
    show_periodogram : {True, False}, bool, optional
        If `True` (default), display periodogram plot of Lomb-Scargle power spectral density vs period
        with significance levels.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label the x-axis with "Period (seconds)"
        and label the y-axis with "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    
    Returns
    -------
    periods : numpy.ndarray
        1D array of periods. Unit is same as `times`.
    powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density from flux and angular frequency,
        e.g. from relative flux, angular frequency 2*pi/seconds.
    sigs_powers : list of tuple of floats
        Powers corresponding to levels of statistical significance from bootstrap analysis.
        Example: [(95.0, 0.05), (99.0, 0.06)]

    See Also
    --------
    astroML.time_series.search_frequencies, select_sig_periods_powers
    
    Notes
    -----
    - Minimum period default calculation:
        min_period = 2.0 * np.median(np.diff(times))
    - Maximum period default calculation:
        max_period = 0.5 * (max(times) - min(times))
    - Number of periods default calculation:
        num_periods = int(min(
            (max(times) - min(times)) / np.median(np.diff(times))),
            1e4)
    - Period sampling is linear in angular frequency space with more samples for shorter periods.
    - Computing periodogram of 1e4 periods with 100 bootstraps takes ~85 seconds for a single 2.7 GHz core.
        Computation time is approximately linear with number of periods and number of bootstraps.
    - Call before `astroML.time_series.search_frequencies`.

    TODO
    ----
    TODO: Create test from astroml book.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, Statistics, Data Mining, and Machine Learning in Astronomy
    .. [2] http://zone.ni.com/reference/en-XX/help/372416B-01/svtconcepts/fft_funda/
    
    """
    # Check inputs.
    median_sampling_period = np.median(np.diff(times))
    min_period_nyquist = 2.0 * median_sampling_period
    if min_period is None:
        min_period = min_period_nyquist
    elif min_period < min_period_nyquist:
        warnings.warn(
            ("`min_period` is less than the Nyquist period limit (2x the median sampling period).\n" +
             "Input: min_period = {per}\n" +
             "Nyquist: min_period_nyquist = {per_nyq}").format(per=min_period,
                                                               per_nyq=min_period_nyquist))
    acquisition_time = max(times) - min(times)
    max_period_acqtime = 0.5 * acquisition_time
    if max_period is None:
        max_period = max_period_acqtime
    elif max_period > max_period_acqtime:
        warnings.warn(
            ("`max_period` is greater than 0.5x the acquisition time.\n" +
             "Input: max_period = {per}\n" +
             "From data: max_period_acqtime = {per_acq}").format(per=max_period,
                                                                 per_acq=max_period_acqtime))
    max_num_periods = int(acquisition_time / median_sampling_period)
    if num_periods is None:
        num_periods = int(min(max_num_periods, 1e4))
    elif num_periods > max_num_periods:
        warnings.warn(
            ("`num_periods` is greater than acquisition time div. by median sampling period.\n" +
             "Input: num_periods = {num}\n" +
             "From data: max_num_periods = {max_num}").format(num=num_periods,
                                                              max_num=max_num_periods))        
    comp_time = 85.0 * (num_periods/1e4) * (num_bootstraps/100) # in seconds
    if comp_time > 10.0:
        print("INFO: Estimated computation time: {time:.0f} sec".format(time=comp_time))
    # Compute periodogram.
    max_omega = 2.0 * np.pi / min_period
    min_omega = 2.0 * np.pi / max_period
    omegas = np.linspace(start=min_omega, stop=max_omega, num=num_periods, endpoint=True)
    periods = 2.0 * np.pi / omegas
    powers = astroML_ts.lomb_scargle(t=times, y=fluxes, dy=fluxes_err, omega=omegas, generalized=True)
    dists = astroML_ts.lomb_scargle_bootstrap(t=times, y=fluxes, dy=fluxes_err, omega=omegas, generalized=True,
                                              N_bootstraps=num_bootstraps, random_state=0)
    sigs_powers = zip(sigs, np.percentile(dists, sigs))
    if show_periodogram:
        # Plot custom periodogram with delta BIC.
        ax0 = \
            plot_periodogram(
                periods=periods, powers=powers, xscale='log', n_terms=1,
                period_unit=period_unit, flux_unit=flux_unit, return_ax=True)
        xlim = (min(periods), max(periods))
        ax0.set_xlim(xlim)
        for (sig, power) in sigs_powers:
            ax0.plot(xlim, [power, power], color='black', linestyle=':')
        ax0.set_title("Generalized Lomb-Scargle periodogram with\n" +
                      "relative Bayesian Information Criterion")
        ax1 = ax0.twinx()
        ax1.set_ylim(
            tuple(
                astroML_ts.lomb_scargle_BIC(
                    P=ax0.get_ylim(), y=fluxes, dy=fluxes_err, n_harmonics=1)))
        ax1.set_ylabel("delta BIC")
        plt.show()
        for (sig, power) in sigs_powers:
            print("INFO: At significance = {sig}%, power spectral density = {pwr}".format(sig=sig, pwr=power))
    return (periods, powers, sigs_powers)


def select_sig_periods_powers(peak_periods, peak_powers, cutoff_power):
    """Select the periods with peak powers above the cutoff power.
           
    Parameters
    ----------
    peak_periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds or days.
    peak_powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density from flux and angular frequency,
        e.g. from relative flux, angular frequency 2*pi/seconds.
    cutoff_power : float
        Power corresponding to a level of statistical significance. Only periods
        above this cutoff power level are returned.
    
    Returns
    -------
    sig_periods : numpy.ndarray
        1D array of significant periods. Unit is same as `peak_periods`.
    sig_powers : numpy.ndarray
        1D array of significant powers. Unit is same as `peak_powers`.

    See Also
    --------
    calc_periodogram, astroML.time_series.search_frequencies

    Notes
    -----
    - Call after `astroML.time_series.search_frequencies`.
    - Call before `calc_best_period`.

    TODO
    ----
    TODO: Create test from numpy example.
    
    """
    sig_idxs = np.where(peak_powers > cutoff_power)
    sig_periods = peak_periods[sig_idxs]
    sig_powers = peak_powers[sig_idxs]
    return (sig_periods, sig_powers)


def calc_best_period(times, fluxes, fluxes_err, candidate_periods,
                     n_terms=6, show_periodograms=False, show_summary_plots=True,
                     period_unit='seconds', flux_unit='relative'):
    """Calculate the period that best represents the data from a multi-term generalized Lomb-Scargle
    periodogram. Convenience function for methods from [1]_.
       
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data. Unit is time, e.g. seconds or days.
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux, e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
    candidate_periods : numpy.ndarray
        1D array of candidate periods. Unit is same as `times`.
    n_terms : {6}, int, optional
        Number of Fourier terms to fit the light curve. To fit eclipses well often requires ~6 terms,
        from section 10.3.3 of [1]_.
    show_periodograms : {False, True}, bool, optional
        If `False` (default), do not display periodograms (power vs period) for each candidate period.
    show_summary_plots : {True, False}, bool, optional
        If `True` (default), display summary plots of delta BIC vs period and periodogram for best fit period.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label the x-axis with "Period (seconds)"
        and label the y-axis with "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    
    Returns
    -------
    best_period : float
        Period with the highest relative Bayesian Information Criterion. Unit is same as `times`.

    See Also
    --------
    astroML.time_series.search_frequencies, calc_num_terms

    Notes
    -----
    - Ranges around each candidate period are based on the angular frequency resolution
        of the original data. Adopted from [2]_.
        acquisition_time = max(times) - min(times)
        omega_resolution = 2.0 * np.pi / acquisition_time
        num_omegas = 1000 # chosen to balance fast computation with medium range
        anti_aliasing = 1.0 / 2.56 # remove digital aliasing
        sampling_precision = 0.1 # ensure sampling precision is higher than data precision
        range_omega_halfwidth = (num_omegas/2.0) * omega_resolution * anti_aliasing * sampling_precision
    - Calculating best period from 100 candidate periods takes ~61 seconds for a single 2.7 GHz core.
        Computation time is approximately linear with number of candidate periods.
    - Call after `astroML.time_series.search_frequencies`.
    - Call before `calc_num_terms`.

    TODO
    ----
    TODO: Create test.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, Statistics, Data Mining, and Machine Learning in Astronomy
    .. [2] http://zone.ni.com/reference/en-XX/help/372416A-01/svtconcepts/fft_funda/
    
    """
    # Check input
    candidate_periods = sorted(candidate_periods) # sort to allow combining identical periods
    comp_time = 61 * len(candidate_periods)/100 
    if comp_time > 10.0:
        print("INFO: Estimated computation time: {time:.0f} sec".format(time=comp_time))
    # Calculate the multiterm periodograms using a range around each candidate angular frequency
    # based on the angular frequency resolution of the original data.
    acquisition_time = max(times) - min(times)
    omega_resolution = 2.0 * np.pi / acquisition_time
    num_omegas = 1000 # chosen to balance fast computation with medium range
    anti_aliasing = 1.0 / 2.56 # remove digital aliasing
    sampling_precision = 0.1 # ensure sampling precision is higher than data precision
    range_omega_halfwidth = (num_omegas/2.0) * omega_resolution * anti_aliasing * sampling_precision
    max_period = 0.5 * acquisition_time
    min_omega = 2.0 * np.pi / max_period
    median_sampling_period = np.median(np.diff(times))
    min_period = 2.0 * median_sampling_period
    max_omega = 2.0 * np.pi / (min_period)
    periods_bics = []
    for candidate_period in candidate_periods:
        candidate_omega = 2.0 * np.pi / candidate_period
        range_omegas = \
            np.clip(
                np.linspace(
                    start=candidate_omega - range_omega_halfwidth,
                    stop=candidate_omega + range_omega_halfwidth,
                    num=num_omegas, endpoint=True),
                min_omega, max_omega)
        range_periods = 2.0 * np.pi / range_omegas
        range_powers = astroML_ts.multiterm_periodogram(t=times, y=fluxes, dy=fluxes_err,
                                                        omega=range_omegas, n_terms=n_terms)
        range_bic_max = max(astroML_ts.lomb_scargle_BIC(P=range_powers, y=fluxes, dy=fluxes_err,
                                                        n_harmonics=n_terms))
        range_omega_best = range_omegas[np.argmax(range_powers)]
        range_period_best = 2.0 * np.pi / range_omega_best
        # Combine identical periods, but only keep the larger delta BIC.
        if len(periods_bics) > 0:
            if np.isclose(last_range_omega_best, range_omega_best, atol=omega_resolution):
                if last_range_bic_max < range_bic_max:
                    periods_bics[-1] = (range_period_best, range_bic_max)
            else:
                periods_bics.append((range_period_best, range_bic_max))
        else:
            periods_bics.append((range_period_best, range_bic_max))
        (last_range_period_best, last_range_bic_max) = periods_bics[-1]
        last_range_omega_best = 2.0 * np.pi / last_range_period_best
        if show_periodograms:
            print(80*'-')
            plot_periodogram(periods=range_periods, powers=range_powers, xscale='linear', n_terms=n_terms,
                             period_unit=period_unit, flux_unit=flux_unit, return_ax=False)
            print("Candidate period: {per} seconds".format(per=candidate_period))
            print("Best period within window: {per} seconds".format(per=range_period_best))
            print("Relative Bayesian Information Criterion: {bic}".format(bic=range_bic_max))
    # Choose the best period from the maximum delta BIC.
    best_idx = np.argmax(zip(*periods_bics)[1])
    (best_period, best_bic) = periods_bics[best_idx]
    if show_summary_plots:
        # Plot delta BICs after all periods have been fit.
        print(80*'-')
        periods_bics_t = zip(*periods_bics)
        fig = plt.figure()
        ax = fig.add_subplot(111, xscale='log')
        ax.plot(periods_bics_t[0], periods_bics_t[1], color='black', marker='o')
        ax.set_title("Relative Bayesian Information Criterion vs period")
        ax.set_xlabel("Period (seconds)")
        ax.set_ylabel("delta BIC")
        plt.show()
        best_omega = 2.0 * np.pi / best_period
        range_omegas = \
            np.clip(
                np.linspace(
                    start=best_omega - range_omega_halfwidth,
                    stop=best_omega + range_omega_halfwidth,
                    num=num_omegas, endpoint=True),
                min_omega, max_omega)
        range_periods = 2.0 * np.pi / range_omegas
        range_powers = astroML_ts.multiterm_periodogram(t=times, y=fluxes, dy=fluxes_err,
                                                        omega=range_omegas, n_terms=n_terms)
        plot_periodogram(periods=range_periods, powers=range_powers, xscale='linear', n_terms=n_terms,
                         period_unit=period_unit, flux_unit=flux_unit, return_ax=False)
        print("Best period: {per} seconds".format(per=best_period))
        print("Relative Bayesian Information Criterion: {bic}".format(bic=best_bic))
    return best_period

