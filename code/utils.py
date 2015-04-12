#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Utilities for reproducing Harrold et al 2015 on SDSS J160036.83+272117.8.

TODO
----
TODO: remove resolutions. resolution is f1 - f0 or p1 - p0.

"""


# Import standard packages.
from __future__ import absolute_import, division, print_function
import warnings
# Import installed packages.
import astroML.time_series as astroML_ts
import matplotlib.pyplot as plt
import numpy as np


def plot_periodogram(
        periods, powers, xscale='log', n_terms=1, period_unit='seconds',
        flux_unit='relative', return_ax=False):
    r"""Plot the periods and powers for a generalized Lomb-Scargle
    periodogram. Convenience function for plot formats from [1]_.

    Parameters
    ----------
    periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds or days.
    powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density
        from flux and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
    xscale : {'log', 'linear'}, string, optional
        `matplotlib.pyplot` attribute to plot periods x-scale in
        'log' (default) or 'linear' scale.
    n_terms : {1}, int, optional
        Number of Fourier terms used to fit the light curve for labeling the plot.
        Example: n_terms=1 will label the title with
        "Generalized Lomb-Scargle periodogram\nwith 1 Fourier terms fit"
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label
        the x-axis with "Period (seconds)"
        and label the y-axis with "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    return_ax : {False, True}, bool
        If `False` (default), show the periodogram plot. Return `None`.
        If `True`, return a `matplotlib.axes` instance for additional modification.

    Returns
    -------
    ax : matplotlib.axes
        Returned only if `return_ax` is `True`. Otherwise returns `None`.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    
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


def calc_periodogram(
        times, fluxes, fluxes_err, min_period=None, max_period=None, num_periods=None,
        sigs=(95.0, 99.0), num_bootstraps=100, show_periodogram=True,
        period_unit='seconds', flux_unit='relative'):
    r"""Calculate periods, powers, and significance levels using generalized
    Lomb-Scargle periodogram. Convenience function for methods from [1]_.
       
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data. Unit is time,
        e.g. seconds or days.
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux,
        e.g. relative flux or magnitudes.
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
        If `True` (default), display periodogram plot of Lomb-Scargle
        power spectral density vs period with significance levels.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label
        the x-axis with "Period (seconds)" and label the y-axis with
        "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    
    Returns
    -------
    periods : numpy.ndarray
        1D array of periods. Unit is same as `times`.
    powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density
        from flux and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
    sigs_powers : list of tuple of floats
        Powers corresponding to levels of statistical significance
        from bootstrap analysis.
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
    - Period sampling is linear in angular frequency space
        with more samples for shorter periods.
    - Computing periodogram of 1e4 periods with 100 bootstraps
        takes ~85 seconds for a single 2.7 GHz core.
        Computation time is approximately linear with number
        of periods and number of bootstraps.
    - Call before `astroML.time_series.search_frequencies`.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy'
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


def select_sig_periods_powers(
        peak_periods, peak_powers, cutoff_power):
    r"""Select the periods with peak powers above the cutoff power.
           
    Parameters
    ----------
    peak_periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds or days.
    peak_powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density from flux
        and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
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
   
    """
    # Check input.
    peak_periods = np.asarray(peak_periods)
    peak_powers = np.asarray(peak_powers)
    # Select significant periods and powers.
    sig_idxs = np.where(peak_powers > cutoff_power)
    sig_periods = peak_periods[sig_idxs]
    sig_powers = peak_powers[sig_idxs]
    return (sig_periods, sig_powers)


def calc_best_period(
        times, fluxes, fluxes_err, candidate_periods, n_terms=6,
        show_periodograms=False, show_summary_plots=True,
        period_unit='seconds', flux_unit='relative'):
    r"""Calculate the period that best represents the data from a multi-term
    generalized Lomb-Scargle periodogram. Convenience function for methods
    from [1]_.
       
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data. Unit is time,
        e.g. seconds or days.
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux,
        e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
    candidate_periods : numpy.ndarray
        1D array of candidate periods. Unit is same as `times`.
    n_terms : {6}, int, optional
        Number of Fourier terms to fit the light curve. To fit eclipses well
        often requires ~6 terms, from section 10.3.3 of [1]_.
    show_periodograms : {False, True}, bool, optional
        If `False` (default), do not display periodograms (power vs period)
        for each candidate period.
    show_summary_plots : {True, False}, bool, optional
        If `True` (default), display summary plots of delta BIC vs period and
        periodogram for best fit period.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label the
        x-axis with "Period (seconds)" and label the y-axis with
        "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    
    Returns
    -------
    best_period : float
        Period with the highest relative Bayesian Information Criterion.
        Unit is same as `times`.

    See Also
    --------
    astroML.time_series.search_frequencies, calc_num_terms

    Notes
    -----
    - Ranges around each candidate period are based on the angular frequency
      resolution of the original data. Adopted from [2]_.
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

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
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


def calc_num_terms(
        times, fluxes, fluxes_err, best_period, max_n_terms=20,
        show_periodograms=False, show_summary_plots=True, period_unit='seconds',
        flux_unit='relative'):
    r"""Calculate the number of Fourier terms that best represent the data's underlying
    variability for representation by a multi-term generalized Lomb-Scargle
    periodogram. Convenience function for methods from [1]_.
       
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data. Unit is time, e.g. seconds or days.
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux,
        e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
    best_period : float
        Period that best represents the data. Unit is same as `times`.
    max_n_terms : {20}, int, optional
        Maximum number of terms to attempt fitting.
        Example: From 10.3.3 of [1]_, many light curves of eclipses are well
        represented with ~6 terms and are best fit with ~10 terms.
    show_periodograms : {False, True}, bool, optional
        If `False` (default), do not display periodograms (power vs period)
        for each candidate number of terms.
    show_summary_plots : {True, False}, bool, optional
        If `True` (default), display summary plots of delta BIC vs number of terms,
        periodogram and phased light curve for best fit number of terms.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plots.
        Example: period_unit='seconds', flux_unit='relative' will label the x-axis
        with "Period (seconds)" and label the y-axis with
        "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    
    Returns
    -------
    best_n_terms : int
        Number of Fourier terms that best fit the light curve. The number of terms
        is determined by the maximum relative Bayesian Information Criterion, from
        section 10.3.3 of [1]_.
    phases : ndarray
        The phase coordinates of the best-fit light curve.
        Unit is decimal orbital phase.
    fits_phased : ndarray
        The relative fluxes for the `phases` of the best-fit light curve.
        Unit is same as `fluxes`.
    times_phased : ndarray
        The phases of the corresponding input `times`.
        Unit is decimal orbital phase.

    See Also
    --------
    calc_best_period, refine_best_period

    Notes
    -----
    -  Range around the best period is based on the angular frequency resolution
        of the original data. Adopted from [2]_.
        acquisition_time = max(times) - min(times)
        omega_resolution = 2.0 * np.pi / acquisition_time
        num_omegas = 1000 # chosen to balance fast computation with medium range
        anti_aliasing = 1.0 / 2.56 # remove digital aliasing
        sampling_precision = 0.1 # ensure sampling precision is higher than data precision
        range_omega_halfwidth = (num_omegas/2.0) * omega_resolution * anti_aliasing * sampling_precision
    - Call after `calc_best_period`.
    - Call before `refine_best_period`.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    # Calculate the multiterm periodograms and choose the best number of terms from the maximum relative BIC.
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
    best_omega = 2.0 * np.pi / best_period
    range_omegas = \
        np.clip(
            np.linspace(
                start=best_omega - range_omega_halfwidth,
                stop=best_omega + range_omega_halfwidth,
                num=num_omegas, endpoint=True),
            min_omega, max_omega)
    range_periods = 2.0 * np.pi / range_omegas
    nterms_bics = []
    for n_terms in range(1, max_n_terms+1):
        range_powers = astroML_ts.multiterm_periodogram(t=times, y=fluxes, dy=fluxes_err,
                                                        omega=range_omegas, n_terms=n_terms)
        range_bic_max = max(astroML_ts.lomb_scargle_BIC(P=range_powers, y=fluxes, dy=fluxes_err,
                                                        n_harmonics=n_terms))
        nterms_bics.append((n_terms, range_bic_max))
        if show_periodograms:
            print(80*'-')
            plot_periodogram(periods=range_periods, powers=range_powers, xscale='linear', n_terms=n_terms,
                             period_unit=period_unit, flux_unit=flux_unit, return_ax=False)
            print("Number of Fourier terms: {num}".format(num=n_terms))
            print("Relative Bayesian Information Criterion: {bic}".format(bic=range_bic_max))
    # Choose the best number of Fourier terms from the maximum delta BIC.
    best_idx = np.argmax(zip(*nterms_bics)[1])
    (best_n_terms, best_bic) = nterms_bics[best_idx]
    mtf = astroML_ts.MultiTermFit(omega=best_omega, n_terms=best_n_terms)
    mtf.fit(t=times, y=fluxes, dy=fluxes_err)
    (phases, fits_phased, times_phased) = mtf.predict(Nphase=1000, return_phased_times=True, adjust_offset=True)
    if show_summary_plots:
        # Plot delta BICs after all terms have been fit.
        print(80*'-')
        nterms_bics_t = zip(*nterms_bics)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(nterms_bics_t[0], nterms_bics_t[1], color='black', marker='o')
        ax.set_xlim(min(nterms_bics_t[0]), max(nterms_bics_t[0]))
        ax.set_title("Relative Bayesian Information Criterion vs\n" +
                     "number of Fourier terms")
        ax.set_xlabel("number of Fourier terms")
        ax.set_ylabel("delta BIC")
        plt.show()
        range_powers = astroML_ts.multiterm_periodogram(t=times, y=fluxes, dy=fluxes_err,
                                                        omega=range_omegas, n_terms=best_n_terms)
        plot_periodogram(periods=range_periods, powers=range_powers, xscale='linear', n_terms=best_n_terms,
                         period_unit=period_unit, flux_unit=flux_unit, return_ax=False)
        print("Best number of Fourier terms: {num}".format(num=best_n_terms))
        print("Relative Bayesian Information Criterion: {bic}".format(bic=best_bic))
    return (best_n_terms, phases, fits_phased, times_phased)


def plot_phased_light_curve(
        phases, fits_phased, times_phased, fluxes, fluxes_err, n_terms=1,
        flux_unit='relative', return_ax=False):
    r"""Plot a phased light curve. Convenience function for plot formats from [1]_.

    Parameters
    ----------
    phases : ndarray
        The phase coordinates of the best-fit light curve.
        Unit is decimal orbital phase.
    fits_phased : ndarray
        The relative fluxes for corresponding `phases` of the best-fit
        light curve. Unit is integrated flux, e.g. relative flux or magnitudes.
    times_phased : ndarray
        The phases of the time coordinates for the observed data.
        Unit is decimal orbital phase.
    fluxes : numpy.ndarray
        1D array of fluxes corresponding to `times_phased`.
        Unit is integrated flux, e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
    n_terms : {1}, int, optional
        Number of Fourier terms used to fit the light curve.
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
    plot_periodogram
    
    Notes
    -----
    - The phased light curve is plotted through two complete cycles
      to illustrate the primary minimum.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.append(times_phased, np.add(times_phased, 1.0)), np.append(fluxes, fluxes),
                np.append(fluxes_err, fluxes_err), fmt='.k', ecolor='gray', linewidth=1)
    ax.plot(np.append(phases, np.add(phases, 1.0)), np.append(fits_phased, fits_phased), color='blue', linewidth=2)
    ax.set_title(("Phased light curve\n" +
                  "with {num} Fourier terms fit").format(num=n_terms))
    ax.set_xlabel("Orbital phase (decimal)")
    ax.set_ylabel("Flux ({unit})".format(unit=flux_unit))
    if return_ax:
        return_obj = ax
    else:
        plt.show()
        return_obj = None
    return return_obj


def refine_best_period(
        times, fluxes, fluxes_err, best_period, n_terms=6, show_plots=True,
        period_unit='seconds', flux_unit='relative'):
    r"""Refine the best period to a higher precision from a multi-term generalized
    Lomb-Scargle periodogram. Convenience function for methods from [1]_.
       
    Parameters
    ----------
    times : numpy.ndarray
        1D array of time coordinates for data. Unit is time, e.g. seconds or days.
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux,
        e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
    best_period : float
        Period that best represents the data. Unit is same as `times`.
    n_terms : {6}, int, optional
        Number of Fourier terms to fit the light curve. To fit eclipses well
        often requires ~6 terms, from section 10.3.3 of [1]_.
    show_plots : {True, False}, bool, optional
        If `True`, display plots of periodograms and phased light curves.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot.
        Example: period_unit='seconds', flux_unit='relative' will label a
        periodogram x-axis with "Period (seconds)" and label a periodogram
        y-axis with
        "Lomb-Scargle Power Spectral Density\n
        (from flux in relative, ang. freq. in 2*pi/seconds)".

    Returns
    -------
    refined_period : float
        Refined period. Unit is same as `times`.
    refined_period_resolution : float
        Precision of refined period. Unit is same as `times`.
    phases : ndarray
        The phase coordinates of the best-fit light curve.
        Unit is decimal orbital phase.
    fits_phased : ndarray
        The relative fluxes for the `phases` of the best-fit light curve.
        Unit is same as `fluxes`.
    times_phased : ndarray
        The phases of the corresponding input `times`.
        Unit is decimal orbital phase.
    mtf : astroML.time_series.MultiTermFit
        Instance of the `astroML.time_series.MultiTermFit` class
        for the best fit model.

    See Also
    --------
    calc_best_period, calc_num_terms

    Notes
    -----
    -  Range around the best period is based on the angular frequency
        resolution of the original data. Adopted from [2]_.
        acquisition_time = max(times) - min(times)
        omega_resolution = 2.0 * np.pi / acquisition_time
        num_omegas = 2000 # chosen to balance fast computation with small range
        anti_aliasing = 1.0 / 2.56 # remove digital aliasing
        sampling_precision = 0.01 # ensure sampling precision very high relative to data precision
        range_omega_halfwidth = (num_omegas/2.0) * omega_resolution * anti_aliasing * sampling_precision
    - Resolution of refined period (period precision) is based on time resolution of original data.
        max_period = 0.5 * acquisition_time
        min_omega = 2.0 * np.pi / max_period
        median_sampling_period = np.median(np.diff(times))
        min_period = 2.0 * median_sampling_period
        max_omega = 2.0 * np.pi / (min_period)
        period_resolution = 2.0 * np.pi / (max_omega - min_omega)
    - Call after `calc_num_terms`.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] http://zone.ni.com/reference/en-XX/help/372416A-01/svtconcepts/fft_funda/
    
    """
    # Calculate the multiterm periodogram and choose the best period from the maximal power.
    acquisition_time = max(times) - min(times)
    omega_resolution = 2.0 * np.pi / acquisition_time
    num_omegas = 2000 # chosen to balance fast computation with medium range
    anti_aliasing = 1.0 / 2.56 # remove digital aliasing
    sampling_precision = 0.01 # ensure sampling precision very high relative to data precision
    range_omega_halfwidth = (num_omegas/2.0) * omega_resolution * anti_aliasing * sampling_precision
    max_period = 0.5 * acquisition_time
    min_omega = 2.0 * np.pi / max_period
    median_sampling_period = np.median(np.diff(times))
    min_period = 2.0 * median_sampling_period
    max_omega = 2.0 * np.pi / (min_period)
    refined_period_resolution = 2.0 * np.pi / (max_omega - min_omega)
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
    refined_omega = range_omegas[np.argmax(range_powers)]
    refined_period = 2.0 * np.pi / refined_omega
    mtf = astroML_ts.MultiTermFit(omega=refined_omega, n_terms=n_terms)
    mtf.fit(t=times, y=fluxes, dy=fluxes_err)
    (phases, fits_phased, times_phased) = mtf.predict(Nphase=1000, return_phased_times=True, adjust_offset=True)
    if show_plots:
        plot_periodogram(periods=range_periods, powers=range_powers, xscale='linear', n_terms=n_terms,
                         period_unit=period_unit, flux_unit=flux_unit, return_ax=False)
        plot_phased_light_curve(phases=phases, fits_phased=fits_phased, times_phased=times_phased,
                                fluxes=fluxes, fluxes_err=fluxes_err, n_terms=n_terms,
                                flux_unit=flux_unit, return_ax=False)
        print("Refined period: {per} seconds".format(per=refined_period))
        print("Refined period resolution: {per_res} seconds".format(per_res=refined_period_resolution))
    return (refined_period, refined_period_resolution, phases, fits_phased, times_phased, mtf)


def calc_flux_fits_residuals(
        phases, fits_phased, times_phased, fluxes):
    r"""Calculate the fluxes and their residuals at phased times from a fit
    to a light curve.
    
    Parameters
    ----------
    phases : numpy.ndarray
        The phase coordinates of the best-fit light curve.
        Unit is decimal orbital phase.
        Required: numpy.shape(phases) == numpy.shape(fits_phased)
    fits_phased : numpy.ndarray
        The relative fluxes for the `phases` of the best-fit light curve.
        Unit is relative flux.
        Required: numpy.shape(phases) == numpy.shape(fits_phased)
    times_phased : numpy.ndarray
        The phases coordinates of the `fluxes`. Unit is decimal orbital phase.
        Required: numpy.shape(times_phased) == numpy.shape(fluxes)
    fluxes : numpy.ndarray
        1D array of fluxes. Unit is integrated flux,
        e.g. relative flux or magnitudes.
        Required: numpy.shape(times_phased) == numpy.shape(fluxes)

    Returns
    -------
    fit_fluxes : numpy.ndarray
        1D array of `fits_phased` resampled at `times_phased`.
        numpy.shape(fluxes_fit) == numpy.shape(fluxes)
    residuals : numpy.ndarray
        1D array of the differences between `fluxes` and `fits_phased` resampled
        at `times_phased`:
        residuals = fluxes - fits_phased_resampled
        numpy.shape(residuals) == numpy.shape(fluxes)

    """
    fit_fluxes = np.interp(x=times_phased, xp=phases, fp=fits_phased)
    residuals = np.subtract(fluxes, fit_fluxes)
    return (fit_fluxes, residuals)


def calc_z1_z2(
        dist):
    """Calculate a rank-based measure of Gaussianity in the core
    and tail of a distribution.
    
    Parameters
    ----------
    dist : array_like
        Distribution to evaluate. 1D array of `float`.
    
    Returns
    -------
    z1 : float
        Departure of distribution core from that of a Gaussian
        in number of sigma.
    z2 : float
        Departure of distribution tails from that of a Gaussian
        in number of sigma.
        
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
        z1 > 1.0 is ~32% and is equivalent to a two-tailed p-value |z| > 1.0.
        The same is true for z2.
    
    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    (mu, sigma) = astroML_stats.mean_sigma(dist)
    (median, sigmaG) = astroML_stats.median_sigmaG(dist)
    num_dist = len(dist)
    z1 = 1.3 * (abs(mu - median) / sigma) * np.sqrt(num_dist)
    z2 = 1.1 * abs((sigma / sigmaG) - 1.0) * np.sqrt(num_dist)
    return (z1, z2)


def plot_phased_histogram(
    hist_phases, hist_fluxes, hist_fluxes_err, times_phased, fluxes, fluxes_err,
    flux_unit='relative', return_ax=False):
    """Plot a Bayesian blocks histogram for a phased light curve.
    Convenience function for methods from [1]_.

    Parameters
    ----------
    hist_phases : numpy.ndarray
        1D array of the phased times of the right edge of each histogram bin.
        Unit is decimal orbital phase.
    hist_fluxes : numpy.ndarray
        1D array of the median fluxes corresponding to the histogram bins
        with right edges of `hist_phases`. Unit is integrated flux,
        e.g. relative flux or magnitudes.
    hist_fluxes_err : np.ndarray
        1D array of the rank-based standard deviation of the binned fluxes
        (sigmaG from section 4.7.4 of [1]_). Unit is same as `hist_fluxes`.
    times_phased : ndarray
        The phases of the time coordinates for the observed data.
        Unit is same as `hist_phases`.
    fluxes : numpy.ndarray
        1D array of fluxes corresponding to `times_phased`.
        Unit is same as `hist_fluxes`.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `hist_fluxes_err`.
    flux_unit : {'relative'}, string, optional
        String describing flux units for labeling the plot.
        Example: flux_unit='relative' will label the y-axis
        with "Flux (relative)".
    return_ax : {False, True}, bool
        If `False` (default), show the periodogram plot. Return `None`.
        If `True`, return a `matplotlib.axes` instance for
        additional modification.

    Returns
    -------
    ax : matplotlib.axes
        Returned only if `return_ax` is `True`. Otherwise returns `None`.
    
    See Also
    --------
    plot_phased_light_curve, calc_phased_histogram
    
    Notes
    -----
    - The phased light curve is plotted through two complete cycles to
    illustrate the primary minimum.

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x=np.append(times_phased, np.add(times_phased, 1.0)), y=np.append(fluxes, fluxes),
                yerr=np.append(fluxes_err, fluxes_err), fmt='.k', ecolor='gray')
    plt_hist_phases = np.append(hist_phases, np.add(hist_phases, 1.0))
    plt_hist_fluxes = np.append(hist_fluxes, hist_fluxes)
    plt_hist_fluxes_upr = np.add(plt_hist_fluxes,
                                 np.append(hist_fluxes_err, hist_fluxes_err))
    plt_hist_fluxes_lwr = np.subtract(plt_hist_fluxes,
                                      np.append(hist_fluxes_err, hist_fluxes_err))
    ax.step(x=plt_hist_phases, y=plt_hist_fluxes, color='blue', linewidth=2)
    ax.step(x=plt_hist_phases, y=plt_hist_fluxes_upr, color='blue', linewidth=3, linestyle=':')
    ax.step(x=plt_hist_phases, y=plt_hist_fluxes_lwr, color='blue', linewidth=3, linestyle=':')
    ax.set_title("Phased light curve\n" +
                 "with Bayesian block histogram")
    ax.set_xlabel("Orbital phase (decimal)")
    ax.set_ylabel("Flux ({unit})".format(unit=flux_unit))
    if return_ax:
        return_obj = ax
    else:
        plt.show()
        return_obj = None
    return return_obj


def calc_phased_histogram(
    times_phased, fluxes, fluxes_err, flux_unit='relative', show_plot=True):
    """Calcluate a Bayesian blocks histogram for a phased light curve.
    Assumes that phased lightcurve is symmetrical about phase=0.5.
    Convenience function for methods from [1]_.

    Parameters
    ----------
    times_phased : numpy.ndarray
        1D array of the phases of the time coordinates for the observed data.
        Unit is decimal orbital phase.
    fluxes : numpy.ndarray
        1D array of fluxes corresponding to `times_phased`.
        Unit is integrated flux, e.g. relative flux or magnitudes.
    fluxes_err : numpy.ndarray
        1D array of errors for fluxes. Unit is same as `fluxes`.
        Errors are used only for determining bin widths, not for determining flux
    flux_unit : {'relative'}, string, optional
        String describing flux units for labeling the plot.
        Example: flux_unit='relative' will label the y-axis
        with "Flux (relative)".
    show_plot : {True, False}, bool, optional
        If `True`, display plot of phased light curve with histogram.

    Returns
    -------
    hist_phases : numpy.ndarray
        1D array of the phased times of the right edge of each Bayesian block bin.
        Unit is same as `times_phased`.
    hist_fluxes : numpy.ndarray
        1D array of the median fluxes corresponding to the Bayesian block bins
        with right edges of `hist_phases`. Unit is same as `fluxes`.
    hist_fluxes_err : np.ndarray
        1D array of the rank-based standard deviation of the binned fluxes
        (sigmaG from section 4.7.4 of [1]_). Unit is same as `fluxes`.
    
    See Also
    --------
    plot_phased_histogram
    
    Notes
    -----
    - The phased light curve is assumed to be symmetrical about phase=0.5
      since the phased lightcurve is folded at phase=0.5. This allows fitting
      unevenly sampled data.
    - Phased times for each bin are defined by the right edge since
      `matplotlib.pyplot.step` constructs plots expecting coordinates for the
      right edges of bins.
    - Because Bayesian blocks have variable bin width, the histogram is computed
      from three complete cycles to prevent the edges of the data domain from
      affecting the computed bin widths.
    - Binned fluxes are combined using the median rather than a weighted average.
      Errors in binned flux are the rank-based standard deviation of the binned
      fluxes (sigmaG from section 4.7.4 of [1]_).

    References
    ----------
    .. [1] Ivezic et al, 2014, "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    # Fold the phased light curve at phase=0.5 to enforce symmetry to
    # accommodate irregular data sampling.
    tfmask_lt05 = times_phased < 0.5
    tfmask_gt05 = np.logical_not(tfmask_lt05)
    phases_folded = np.append(times_phased[tfmask_lt05], np.subtract(1.0, times_phased[tfmask_gt05]))
    phases_mirrored = np.append(phases_folded, np.subtract(1.0, phases_folded))
    fluxes_folded = np.append(fluxes[tfmask_lt05], fluxes[tfmask_gt05])
    fluxes_mirrored = np.append(fluxes_folded, fluxes_folded)
    fluxes_err_folded = np.append(fluxes_err[tfmask_lt05], fluxes_err[tfmask_gt05])
    fluxes_err_mirrored = np.append(fluxes_err_folded, fluxes_err_folded)
    # Append the data to itself (tesselate) 2 times for total of 3 cycles to compute
    # histogram without edge effects.
    tess_phases = phases_mirrored
    tess_fluxes = fluxes_mirrored
    tess_fluxes_err = fluxes_err_mirrored
    for begin_phase in xrange(0, 2):
        tess_phases = np.append(tess_phases, np.add(phases_mirrored, begin_phase + 1.0))
        tess_fluxes = np.append(tess_fluxes, fluxes_mirrored)
        tess_fluxes_err = np.append(tess_fluxes_err, fluxes_err_mirrored)
    # Compute edges of Bayesian blocks histogram.
    # Note: number of edges = number of bins + 1
    tess_bin_edges = \
        astroML_dens.bayesian_blocks(t=tess_phases, x=tess_fluxes,
                                     sigma=tess_fluxes_err, fitness='measures')
    # Determine the median flux and sigmaG within each Bayesian block bin.
    tess_bin_fluxes = []
    tess_bin_fluxes_err = []
    for idx_start in xrange(len(tess_bin_edges) - 1):
        bin_phase_start = tess_bin_edges[idx_start]
        bin_phase_end = tess_bin_edges[idx_start + 1]
        tfmask_bin = np.logical_and(bin_phase_start <= tess_phases,
                                    tess_phases <= bin_phase_end)
        if tfmask_bin.any():
            (bin_flux, bin_flux_err) = astroML_stats.median_sigmaG(tess_fluxes[tfmask_bin])
        else:
            (bin_flux, bin_flux_err) = (np.nan, np.nan)
        tess_bin_fluxes.append(bin_flux)
        tess_bin_fluxes_err.append(bin_flux_err)
    tess_bin_fluxes = np.asarray(tess_bin_fluxes)
    tess_bin_fluxes_err = np.asarray(tess_bin_fluxes_err)
    # Fix number of edges = number of bins. Values are for are right edge of each bin.
    tess_bin_edges = tess_bin_edges[1:]
    # Crop 1 complete cycle out of center of of 3-cycle histogram.
    # Plot and return histogram.
    tfmask_hist = np.logical_and(1.0 <= tess_bin_edges, tess_bin_edges <= 2.0)
    hist_phases = np.subtract(tess_bin_edges[tfmask_hist], 1.0)
    hist_fluxes = tess_bin_fluxes[tfmask_hist]
    hist_fluxes_err = tess_bin_fluxes_err[tfmask_hist]
    if show_plot:
        plot_phased_histogram(
            hist_phases=hist_phases, hist_fluxes=hist_fluxes, hist_fluxes_err=hist_fluxes_err,
            times_phased=phases_mirrored, fluxes=fluxes_mirrored, fluxes_err=fluxes_err_mirrored,
            flux_unit=flux_unit, return_ax=False)
    return (hist_phases, hist_fluxes, hist_fluxes_err)
