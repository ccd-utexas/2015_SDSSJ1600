r"""Archived utilities for reproducing Harrold et al 2015 on
SDSS J160036.83+272117.8. Code here is no longer used.

"""


def calc_periodogram(
    times, fluxes, fluxes_err, filts, min_period=None, max_period=None,
    num_periods=None, sigs=(95.0, 99.0), num_shuffles=100,
    show_plot=True, period_unit='seconds', flux_unit='relative'):
    r"""Calculate periods, powers, best period, and significance levels
    using multiband generalized Lomb-Scargle periodogram. Convenience function
    for methods from [1]_, [2]_.
       
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
    filts : list
        1D array of string values for filters.
        Example: filts=['u', 'g', 'r', u', 'g', 'r', ...]
    min_period : {None}, float, optional
        Minimum period to sample. Unit is same as `times`.
        If `None` (default), `min_period` defined by `calc_period_limits`.
    max_period : {None}, float, optional
        Maximum period to sample. Unit is same as `times`.
        If `None` (default), `max_period` defined by `calc_period_limits`.
    num_periods : {None}, int, optional
        Number of periods to sample, including `min_period` and `max_period`.
        If `None` (default), `num_periods` defined by `calc_period_limits`,
        or 1e3, whichever is fewer  to limit computation time.
    sigs : {(95.0, 99.0)}, tuple of floats, optional
        Levels of statistical significance for which to compute corresponding
        Lomb-Scargle relative percentile powers via shuffling `times`.
    num_shuffles : {100}, int, optional
        Number of shuffles to compute significance levels.
    show_periodogram : {True, False}, bool, optional
        If `True` (default), display periodogram plot of Lomb-Scargle
        power spectral density vs period with significance levels.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plot
        with `plot_periodogram`.
    
    Returns
    -------
    periods : numpy.ndarray
        1D array of periods. Unit is same as `times`.
    powers  : numpy.ndarray
        1D array of powers. Unit is relative Lomb-Scargle power spectral density
        from flux and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
    sigs_powers : list
        Relative powers corresponding to levels of statistical significance.
        Example: 
            [(95.0, [0.051, 0.052, 0.049, ...]),
             (99.0, [0.059, 0.062, 0.061, ...])]
            np.shape(relative_powers) == np.shape(times)

    See Also
    --------
    calc_period_limits, plot_periodogram
    
    Notes
    -----
    - Computing periodogram of 1e3 periods with 100 shuffles
        takes ~85 seconds for a single 2.7 GHz core.
        Computation time is approximately linear with number
        of periods and number of shuffles.

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy'
    .. [2] VanderPlas and Ivezic, 2015
           http://adsabs.harvard.edu/abs/2015arXiv150201344V
    
    """
    # Check input.
    (min_period_limit, max_period_limit, num_periods_limit) = \
        calc_period_limits(times=times)
    if min_period is None:
        min_period = min_period_limit
    if max_period is None:
        max_period = max_period_limit
    if num_periods_limit is None:
        num_periods = num_periods_limit
    # Model time series and find best period.
    model = gatspy_per.LombScargleMultiband(Nterms_base=6, Nterms_band=1)
    model.optimizer.period_range = (min_period, max_period)
    model.fit(t=times, y=fluxes, dy=fluxes_err, filts=filts)
    model.best_period
    # Calculate powers for plot.
    min_omega = 2.0*np.pi / max_period
    max_omega = 2.0*np.pi / min_period
    num_omegas = num_periods
    omegas = \
        np.linspace(
            start=min_omega, stop=max_omega, num=num_omegas, endpoint=True)
    periods = 2.0*np.pi / omegas
    powers = model.periodogram(periods=periods)
    # Model noise in time series and find significance levels.
    np.random.seed(seed=0) # for reproducibility
    noise_model = copy.copy(model)
    shuffled_times = copy.copy(times)
    noise_powers_arr = []
    for _ in xrange(num_shuffles):
        np.random.shuffle(shuffled_times)
        noise_model.fit(t=shuffled_times, y=fluxes, dy=fluxes_err, filts=filts)
        noise_powers_arr.append(noise_model.periodogram(periods=periods))
    # TODO: redo below here
    dists = \
        astroML_ts.lomb_scargle_bootstrap(
            t=times, y=fluxes, dy=fluxes_err, omega=omegas, generalized=True,
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
            print(
                ("INFO: At significance = {sig}%, " +
                 "power spectral density = {pwr}").format(
                    sig=sig, pwr=power))
    return (periods, powers, sigs_powers)


def select_sig_periods_powers(
    peak_periods, peak_powers, cutoff_power):
    r"""Select the periods with peak powers above the cutoff power.
           
    Parameters
    ----------
    peak_periods : numpy.ndarray
        1D array of periods. Unit is time, e.g. seconds or days.
    peak_powers  : numpy.ndarray
        1D array of powers. Unit is Lomb-Scargle power spectral density
        from flux and angular frequency, e.g. from relative flux,
        angular frequency 2*pi/seconds.
    cutoff_power : float
        Power corresponding to a level of statistical significance.
        Only periods above this cutoff power level are returned.
    
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
        num_omegas = 1000 # balance fast computation with medium range
        anti_aliasing = 1.0 / 2.56 # remove digital aliasing
        # ensure sampling precision is higher than data precision
        sampling_precision = 0.1 
        range_omega_halfwidth = \
            ((num_omegas/2.0) * omega_resolution * anti_aliasing *
             sampling_precision)
    - Calculating best period from 100 candidate periods takes ~61 seconds for a single 2.7 GHz core.
      Computation time is approx linear with number of candidate periods.
    - Call after `astroML.time_series.search_frequencies`.
    - Call before `calc_num_terms`.

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] http://zone.ni.com/reference/en-XX/help/372416A-01/svtconcepts/
           fft_funda/
    
    """
    # Check input
    # TODO: separate as a function
    # sort to allow combining identical periods
    candidate_periods = sorted(candidate_periods) 
    comp_time = 61 * len(candidate_periods)/100 
    if comp_time > 10.0:
        print("INFO: Estimated computation time: {time:.0f} sec".format(
            time=comp_time))
    # Calculate the multiterm periodograms using a range around each
    # candidate angular frequency based on the angular frequency resolution
    # of the original data.
    acquisition_time = max(times) - min(times)
    omega_resolution = 2.0 * np.pi / acquisition_time
    num_omegas = 1000 # chosen to balance fast computation with medium range
    anti_aliasing = 1.0 / 2.56 # remove digital aliasing
    # ensure sampling precision is higher than data precision
    sampling_precision = 0.1 
    range_omega_halfwidth = \
        ((num_omegas/2.0) * omega_resolution * anti_aliasing *
         sampling_precision)
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
        range_powers = \
            astroML_ts.multiterm_periodogram(
                t=times, y=fluxes, dy=fluxes_err, omega=range_omegas,
                n_terms=n_terms)
        range_bic_max = \
            max(astroML_ts.lomb_scargle_BIC(
                P=range_powers, y=fluxes, dy=fluxes_err, n_harmonics=n_terms))
        range_omega_best = range_omegas[np.argmax(range_powers)]
        range_period_best = 2.0 * np.pi / range_omega_best
        # Combine identical periods, but only keep the larger delta BIC.
        if len(periods_bics) > 0:
            if np.isclose(last_range_omega_best, range_omega_best,
                          atol=omega_resolution):
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
            plot_periodogram(
                periods=range_periods, powers=range_powers, xscale='linear',
                n_terms=n_terms, period_unit=period_unit, flux_unit=flux_unit,
                return_ax=False)
            print("Candidate period: {per} seconds".format(
                per=candidate_period))
            print("Best period within window: {per} seconds".format(
                per=range_period_best))
            print("Relative Bayesian Information Criterion: {bic}".format(
                bic=range_bic_max))
    # Choose the best period from the maximum delta BIC.
    best_idx = np.argmax(zip(*periods_bics)[1])
    (best_period, best_bic) = periods_bics[best_idx]
    if show_summary_plots:
        # Plot delta BICs after all periods have been fit.
        print(80*'-')
        periods_bics_t = zip(*periods_bics)
        fig = plt.figure()
        ax = fig.add_subplot(111, xscale='log')
        ax.plot(
            periods_bics_t[0], periods_bics_t[1], color='black', marker='o')
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
        range_powers = \
            astroML_ts.multiterm_periodogram(
                t=times, y=fluxes, dy=fluxes_err, omega=range_omegas,
                n_terms=n_terms)
        plot_periodogram(
            periods=range_periods, powers=range_powers, xscale='linear',
            n_terms=n_terms, period_unit=period_unit, flux_unit=flux_unit,
            return_ax=False)
        print("Best period: {per} seconds".format(per=best_period))
        print("Relative Bayesian Information Criterion: {bic}".format(
            bic=best_bic))
    return best_period


def calc_num_terms(
    times, fluxes, fluxes_err, best_period, max_n_terms=20,
    show_periodograms=False, show_summary_plots=True, period_unit='seconds',
    flux_unit='relative'):
    r"""Calculate the number of Fourier terms that best represent the data's
    underlying variability for representation by a multi-term generalized
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
        If `True` (default), display summary plots of delta BIC vs number of
        terms, periodogram and phased light curve for best fit number of terms.
    period_unit : {'seconds'}, string, optional
    flux_unit : {'relative'}, string, optional
        Strings describing period and flux units for labeling the plots.
        Example: period_unit='seconds', flux_unit='relative' will label the
        x-axis with "Period (seconds)" and label the y-axis with
        "Lomb-Scargle Power Spectral Density\n" +
        "(from flux in relative, ang. freq. in 2*pi/seconds)".
    
    Returns
    -------
    best_n_terms : int
        Number of Fourier terms that best fit the light curve. The number of
        terms is determined by the maximum relative
        Bayesian Information Criterion, from section 10.3.3 of [1]_.
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
    -  Range around the best period is based on the angular frequency
       resolution of the original data. Adopted from [2]_.
       acquisition_time = max(times) - min(times)
       omega_resolution = 2.0 * np.pi / acquisition_time
       num_omegas = 1000 # chosen to balance fast computation with medium range
       anti_aliasing = 1.0 / 2.56 # remove digital aliasing
       # ensure sampling precision is higher than data precision
       sampling_precision = 0.1 
       range_omega_halfwidth = \
           ((num_omegas/2.0) * omega_resolution * anti_aliasing *
            sampling_precision)
    - Call after `calc_best_period`.
    - Call before `refine_best_period`.

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    
    """
    # TODO: separate as own function.
    # Calculate the multiterm periodograms and choose the best number of
    # terms from the maximum relative BIC.
    acquisition_time = max(times) - min(times)
    omega_resolution = 2.0 * np.pi / acquisition_time
    num_omegas = 1000 # chosen to balance fast computation with medium range
    anti_aliasing = 1.0 / 2.56 # remove digital aliasing
    # ensure sampling precision is higher than data precisionR
    sampling_precision = 0.1 
    range_omega_halfwidth = \
        ((num_omegas/2.0) * omega_resolution * anti_aliasing *
         sampling_precision)
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
        range_powers = \
            astroML_ts.multiterm_periodogram(
                t=times, y=fluxes, dy=fluxes_err, omega=range_omegas,
                n_terms=n_terms)
        range_bic_max = \
            max(astroML_ts.lomb_scargle_BIC(
                P=range_powers, y=fluxes, dy=fluxes_err, n_harmonics=n_terms))
        nterms_bics.append((n_terms, range_bic_max))
        if show_periodograms:
            print(80*'-')
            plot_periodogram(
                periods=range_periods, powers=range_powers, xscale='linear',
                n_terms=n_terms, period_unit=period_unit, flux_unit=flux_unit,
                return_ax=False)
            print("Number of Fourier terms: {num}".format(num=n_terms))
            print("Relative Bayesian Information Criterion: {bic}".format(
                bic=range_bic_max))
    # Choose the best number of Fourier terms from the maximum delta BIC.
    best_idx = np.argmax(zip(*nterms_bics)[1])
    (best_n_terms, best_bic) = nterms_bics[best_idx]
    mtf = astroML_ts.MultiTermFit(omega=best_omega, n_terms=best_n_terms)
    mtf.fit(t=times, y=fluxes, dy=fluxes_err)
    (phases, fits_phased, times_phased) = \
        mtf.predict(Nphase=1000, return_phased_times=True, adjust_offset=True)
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
        range_powers = \
            astroML_ts.multiterm_periodogram(
                t=times, y=fluxes, dy=fluxes_err, omega=range_omegas,
                n_terms=best_n_terms)
        plot_periodogram(
            periods=range_periods, powers=range_powers, xscale='linear',
            n_terms=best_n_terms, period_unit=period_unit,
            flux_unit=flux_unit, return_ax=False)
        print("Best number of Fourier terms: {num}".format(num=best_n_terms))
        print("Relative Bayesian Information Criterion: {bic}".format(
            bic=best_bic))
    return (best_n_terms, phases, fits_phased, times_phased)


def refine_best_period(
    times, fluxes, fluxes_err, best_period, n_terms=6, show_plots=True,
    period_unit='seconds', flux_unit='relative'):
    r"""Refine the best period to a higher precision from a multi-term
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
        # ensure sampling precision very high relative to data precision
        sampling_precision = 0.01 
        range_omega_halfwidth = \
            ((num_omegas/2.0) * omega_resolution * anti_aliasing *
             sampling_precision)
    - Call after `calc_num_terms`.

    References
    ----------
    .. [1] Ivezic et al, 2014,
           "Statistics, Data Mining, and Machine Learning in Astronomy"
    .. [2] http://zone.ni.com/reference/en-XX/help/372416A-01/svtconcepts/
           fft_funda/
    
    """
    # Calculate the multiterm periodogram and choose the best period
    # from the maximal power.
    acquisition_time = max(times) - min(times)
    omega_resolution = 2.0 * np.pi / acquisition_time
    num_omegas = 2000 # chosen to balance fast computation with medium range
    anti_aliasing = 1.0 / 2.56 # remove digital aliasing
    # ensure sampling precision very high relative to data precision
    sampling_precision = 0.01 
    range_omega_halfwidth = \
      ((num_omegas/2.0) * omega_resolution * anti_aliasing *
       sampling_precision)
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
    range_powers = \
        astroML_ts.multiterm_periodogram(
            t=times, y=fluxes, dy=fluxes_err, omega=range_omegas,
            n_terms=n_terms)
    refined_omega = range_omegas[np.argmax(range_powers)]
    refined_period = 2.0 * np.pi / refined_omega
    mtf = astroML_ts.MultiTermFit(omega=refined_omega, n_terms=n_terms)
    mtf.fit(t=times, y=fluxes, dy=fluxes_err)
    (phases, fits_phased, times_phased) = \
        mtf.predict(Nphase=1000, return_phased_times=True, adjust_offset=True)
    if show_plots:
        plot_periodogram(
            periods=range_periods, powers=range_powers, xscale='linear',
            n_terms=n_terms, period_unit=period_unit, flux_unit=flux_unit,
            return_ax=False)
        plot_phased_light_curve(
            phases=phases, fits_phased=fits_phased, times_phased=times_phased,
            fluxes=fluxes, fluxes_err=fluxes_err, n_terms=n_terms,
            flux_unit=flux_unit, return_ax=False)
        print("Refined period: {per} seconds".format(per=refined_period))
    return (refined_period, phases, fits_phased, times_phased, mtf)

