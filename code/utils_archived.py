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

