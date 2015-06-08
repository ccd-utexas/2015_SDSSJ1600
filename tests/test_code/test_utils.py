#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for code/utils.py.

Notes
-----
Tests are executed using pytest.

"""


# Import standard packages.
from __future__ import absolute_import, division, print_function
import collections
import StringIO
import sys
sys.path.insert(0, '.') # Test the code in this repository.
# Import installed packages.
import astroML.time_series as astroML_ts
import gatspy.datasets as gatspy_data
import gatspy.periodic as gatspy_per
import matplotlib.pyplot as plt
import numpy as np
# Import local packages.
import code


def test_calc_period_limits(
    times=xrange(10), ref_min_period=2.0, ref_max_period=4.5,
    ref_num_periods=23):
    r"""Pytest for code/utils.py:
    calc_period_limits
    
    """
    (test_min_period, test_max_period, test_num_periods) = \
        code.utils.calc_period_limits(times=times)
    assert np.isclose(ref_min_period, test_min_period)
    assert np.isclose(ref_max_period, test_max_period)
    assert np.isclose(ref_num_periods, test_num_periods)
    return None


def test_calc_sig_levels_case():
    r"""Pytest cases for code/utils.py:
    calc_sig_levels

    """
    # Define function for testing cases.
    def test_calc_sig_levels(
        model, ref_sig_periods, ref_sig_powers, sigs=(95.0, 99.0), num_periods=10,
        num_shuffles=100):
        r"""Pytest for code/utils.py:
        calc_sig_levels

        """
        (test_sig_periods, test_sig_powers) = \
            code.utils.calc_sig_levels(
                model=model, sigs=sigs, num_periods=num_periods,
                num_shuffles=num_shuffles)
        assert np.all(np.isclose(ref_sig_periods, test_sig_periods))
        for (ref_sig_power, test_sig_power) in zip(ref_sig_powers, test_sig_powers):
            assert np.all(np.isclose(ref_sig_power, test_sig_power))
        return None
    # Test adapted from
    # https://github.com/astroML/gatspy/blob/master/examples/MultiBand.ipynb
    rrlyrae = gatspy_data.fetch_rrlyrae()
    lcid = rrlyrae.ids[0]
    (times, mags, mags_err, filts) = rrlyrae.get_lightcurve(lcid)
    model = gatspy_per.LombScargleMultiband(Nterms_base=6, Nterms_band=1)
    model.fit(t=times, y=mags, dy=mags_err, filts=filts)
    (min_period, max_period, _) = code.utils.calc_period_limits(times=times)
    model.optimizer.period_range = (min_period, max_period)
    sigs = (95.0, 99.0)
    num_periods = 10
    num_shuffles = 100
    ref_sig_periods = \
        [1.66051856e+03, 2.99875187e-02, 1.49938947e-02, 9.99595991e-03,
         7.49698121e-03, 5.99759039e-03, 4.99799500e-03, 4.28399755e-03,
         3.74849907e-03, 3.33200001e-03]
    ref_sig_powers = \
        {95.0: \
            [0.25057282, 0.2677001 , 0.25552392, 0.27637847, 0.26304325,
             0.26439929, 0.24606125, 0.24080878, 0.24847659, 0.2455511 ],
         99.0: \
            [0.32243259, 0.31733991, 0.29113095, 0.33229025, 0.29343131,
             0.29407153, 0.26594016, 0.26332914, 0.29706267, 0.2731428 ]}
    test_calc_sig_levels(
        model=model, sigs=sigs, num_periods=10, num_shuffles=100,
        ref_sig_periods=ref_sig_periods, ref_sig_powers=ref_sig_powers)
    # TODO: insert additional test cases here.
    return None


# TODO: REDO BELOW HERE

def test_plot_periodogram(
    periods=[1,2,4,8,16], powers=[1,2,4,2,1], xscale='log',
    n_terms=1, period_unit='seconds', flux_unit='relative', return_ax=True):
    r"""Pytest for code/utils.py:
    plot_periodogram
    
    """
    ax = \
      code.utils.plot_periodogram(
          periods=periods, powers=powers, xscale=xscale, n_terms=n_terms,
          period_unit=period_unit, flux_unit=flux_unit, return_ax=return_ax)
    assert isinstance(ax, plt.Axes)
    return None


def test_calc_periodogram(
    times=[1,2,3,4,5,6,7,8],
    fluxes=[0,1,0,1,0,1,0,1],
    fluxes_err=[1,1,1,1,1,1,1,1],
    min_period=None, max_period=None, num_periods=None, sigs=(95.0, 99.0),
    num_bootstraps=100, show_periodogram=False, period_unit='seconds',
    flux_unit='relative',
    ref_periods=[3.5, 3.11111111, 2.8, 2.54545455,
                 2.33333333, 2.15384615, 2.0],
    ref_powers=[0.05615773, 0.09897668, 0.02777778, 0.06395519,
                0.68555962, 0.98455515, 1.45655044],
    ref_sigs_powers=[(95.0, 0.92877340237699546),
                     (99.0, 1.0210555725004953)]):
    r"""Pytest for code/utils.py:
    calc_periodogram

    """
    (test_periods, test_powers, test_sigs_powers) = \
        code.utils.calc_periodogram(
            times=times, fluxes=fluxes, fluxes_err=fluxes_err,
            min_period=min_period, max_period=max_period,
            num_periods=num_periods, sigs=sigs, num_bootstraps=num_bootstraps,
            show_periodogram=show_periodogram, period_unit=period_unit,
            flux_unit=flux_unit)
    assert np.all(np.isclose(ref_periods, test_periods))
    assert np.all(np.isclose(ref_powers, test_powers))
    assert np.all(np.isclose(ref_sigs_powers, test_sigs_powers))
    return None


def test_select_sig_periods_powers(
    peak_periods=[1, 2, 3], peak_powers=[0.1, 0.2, 0.3], cutoff_power=0.15,
    ref_sig_periods=[2, 3], ref_sig_powers=[0.2, 0.3]):
    r"""pytest style test for code.select_sig_periods_powers

    """
    (test_sig_periods, test_sig_powers) = \
        code.utils.select_sig_periods_powers(
            peak_periods=peak_periods, peak_powers=peak_powers,
            cutoff_power=cutoff_power)
    assert np.all(np.isclose(ref_sig_periods, test_sig_periods))
    assert np.all(np.isclose(ref_sig_powers, test_sig_powers))
    return None


def test_calc_best_period(
    times=[1,2,3,4,5,6,7,8],
    fluxes=[0,1,0,1,0,1,0,1],
    fluxes_err=[1,1,1,1,1,1,1,1],
    candidate_periods=[2.15384615, 2.], n_terms=1, show_periodograms=False,
    show_summary_plots=False, period_unit='seconds', flux_unit='relative',
    ref_best_period=2.0):
    r"""pytest style test for code.utils.calc_best_period

    """
    test_best_period = \
        code.utils.calc_best_period(
            times=times, fluxes=fluxes, fluxes_err=fluxes_err,
            candidate_periods=candidate_periods, n_terms=n_terms,
            show_periodograms=show_periodograms,
            show_summary_plots=show_summary_plots,
            period_unit=period_unit, flux_unit=flux_unit)
    assert np.isclose(test_best_period, ref_best_period)
    return None


def test_calc_num_terms(
    times=range(2**7), fluxes=[0,1]*2**6, fluxes_err=[1]*2**7,
    best_period=2.0, max_n_terms=2, show_periodograms=False,
    show_summary_plots=False, period_unit='seconds', flux_unit='relative',
    ref_best_n_terms=1,
    ref_phases=np.linspace(start=0, stop=1, num=1000, endpoint=False),
    ref_fits_phased=None, ref_times_phased=[0.004, 0.504]*2**6):
    r"""pytest style test for code.utils.calc_num_terms

    """
    (test_best_n_terms, test_phases, test_fits_phased, test_times_phased) = \
        code.utils.calc_num_terms(
            times=times, fluxes=fluxes, fluxes_err=fluxes_err,
            best_period=best_period, max_n_terms=max_n_terms,
            show_periodograms=show_periodograms,
            show_summary_plots=show_summary_plots, period_unit=period_unit,
            flux_unit=flux_unit)
    assert ref_best_n_terms == test_best_n_terms
    assert np.all(np.isclose(ref_phases, test_phases))
    if ref_fits_phased is None:
        assert len(ref_phases) == len(test_fits_phased)
    else:
        assert np.all(np.isclose(ref_fits_phased, test_fits_phased))
    assert np.all(np.isclose(ref_times_phased, test_times_phased))
    return None


def test_plot_phased_light_curve(
    phases=np.linspace(start=0, stop=1, num=1000, endpoint=False),
    fits_phased=[1]*1000, times_phased=[0.004, 0.504]*2**6, fluxes=[0,1]*2**6,
    fluxes_err=[1]*2**7, n_terms=1, flux_unit='relative', return_ax=True):
    r"""pytest style test for code.utils.plot_phased_light_curve

    """
    ax = \
        code.utils.plot_phased_light_curve(
            phases=phases, fits_phased=fits_phased, times_phased=times_phased,
            fluxes=fluxes, fluxes_err=fluxes_err, n_terms=n_terms,
            flux_unit=flux_unit, return_ax=return_ax)
    assert isinstance(ax, plt.Axes)
    return None


def test_refine_best_period(
    times=range(2**7), fluxes=[0,1]*2**6, fluxes_err=[1]*2**7, best_period=2.0,
    n_terms=1, show_plots=False, period_unit='seconds', flux_unit='relative',
    ref_best_period=2.0,
    ref_phases=np.linspace(start=0, stop=1, num=1000, endpoint=False),
    ref_fits_phased=None, ref_times_phased=[0.004, 0.504]*2**6):
    r"""pytest style test for code.utils.refine_best_period

    """
    (test_best_period, test_phases, test_fits_phased,
     test_times_phased, test_multi_term_fit) = \
        code.utils.refine_best_period(
            times=range(2**7), fluxes=[0,1]*2**6,
            fluxes_err=[1]*2**7, best_period=2.0, n_terms=1,
            show_plots=False, period_unit='seconds', flux_unit='relative')
    assert ref_best_period == test_best_period
    assert np.all(np.isclose(ref_phases, test_phases))
    if ref_fits_phased is None:
        assert len(ref_phases) == len(test_fits_phased)
    else:
        assert np.all(np.isclose(ref_fits_phased, test_fits_phased))
    assert np.all(np.isclose(ref_times_phased, test_times_phased))
    assert isinstance(test_multi_term_fit, astroML_ts.MultiTermFit)
    return None


def test_calc_flux_fits_residuals(
    phases=np.linspace(start=0, stop=1, num=1000, endpoint=False),
    fits_phased=[1]*1000, times_phased=[0.004, 0.504]*2**6, fluxes=[0,1]*2**6,
    ref_fit_fluxes=[1]*2**7, ref_residuals=[-1,0]*2**6):
    r"""pytest style test for code.utils.calc_flux_fits_residuals

    """
    (test_fit_fluxes, test_residuals) = \
        code.utils.calc_flux_fits_residuals(
            phases=phases, fits_phased=fits_phased,
            times_phased=times_phased, fluxes=fluxes)
    assert np.all(np.isclose(ref_fit_fluxes, test_fit_fluxes))
    assert np.all(np.isclose(ref_residuals, test_residuals))
    return None


# Seed random number generator for reproducibility.
np.random.seed(0)
def test_calc_z1_z2(
    dist=np.random.normal(loc=0, scale=1, size=1000),
    ref_z1=0.53192162282074262, ref_z2=0.6959521800983498):
    r"""pytest style test for code.utils.calc_z1_z2

    """
    (test_z1, test_z2) = code.utils.calc_z1_z2(dist=dist)
    assert np.isclose(ref_z1, test_z1)
    assert np.isclose(ref_z2, test_z2)
    return None


def test_plot_phased_histogram(
    hist_phases=[0.1, 0.4, 0.6, 0.9], hist_fluxes=[0.5, 1.0, 0.75, 1.0],
    hist_fluxes_err=[0.05]*4,
    times_phased=np.linspace(start=0, stop=1, num=10, endpoint=False),
    fluxes=[0.5, 0.5, 1.0, 1.0, 0.75, 0.75, 0.75, 1.0, 1.0, 0.5],
    fluxes_err=[0.05]*10, flux_unit='relative', return_ax=False):
    r"""pytest style test for code.utils.plot_phased_histogram

    """
    ax = \
        code.utils.plot_phased_histogram(
            hist_phases=hist_phases, hist_fluxes=hist_fluxes,
            hist_fluxes_err=hist_fluxes_err,
            times_phased=times_phased, fluxes=fluxes, fluxes_err=fluxes_err,
            flux_unit='relative', return_ax=True)
    assert isinstance(ax, plt.Axes)
    return None

def test_calc_phased_histogram(
    times_phased=np.linspace(start=0, stop=1, num=10, endpoint=False),
    fluxes=[0.5, 0.5, 1.0, 1.0, 0.75, 0.75, 0.75, 1.0, 1.0, 0.5],
    fluxes_err=[0.05]*10, flux_unit='relative', show_plot=False,
    ref_hist_phases=[0.15, 0.35, 0.65, 0.85],
    ref_hist_fluxes=[ 0.5, 1., 0.75, 1.], ref_hist_fluxes_err=[0., 0., 0., 0.]):
    r"""pytest style test for code.utils.calc_phased_histogram

    """
    (test_hist_phases, test_hist_fluxes, test_hist_fluxes_err) = \
        code.utils.calc_phased_histogram(
            times_phased=times_phased, fluxes=fluxes, fluxes_err=fluxes_err,
            flux_unit=flux_unit, show_plot=show_plot)
    assert np.all(np.isclose(ref_hist_phases, test_hist_phases))
    assert np.all(np.isclose(ref_hist_fluxes, test_hist_fluxes))
    assert np.all(np.isclose(ref_hist_fluxes_err, test_hist_fluxes_err))
    return None


def test_read_quants_gianninas(
    fobj=StringIO.StringIO("Name         SpT    Teff log L/Lo  t_cool \n" +
                           "==========  ===== ======= ====== =========\n" +
                           "J1600+2721  DA6.0   8353. -1.002 1.107 Gyr"),
    dobj=collections.OrderedDict(
        [('Name', 'J1600+2721'), ('SpT', 'DA6.0'), ('Teff', 8353.0),
         ('log L/Lo', -1.002), ('t_cool', 1.107)])):
    r"""Test that parameters from Gianninas are read correctly.
    
    """
    assert dobj == code.utils.read_quants_gianninas(fobj=fobj)
    return None


def test_has_nans(
    obj={'a': None, 'b': {'b1': True, 'b2': [False, 1, np.nan, 'asdf']}},
    found_nan=True):
    r"""Test that nans are found correctly.
    
    """
    assert code.utils.has_nans(obj) == found_nan
    return None


# Additional cases for test_has_nans
test_has_nans(
    obj={'a': None, 'b': {'b1': True, 'b2': [False, 1, 'nan', ('asdf', 2.0)]}},
    found_nan=False)


# TODO: update test when light curve parameters include phase, period.
def test_model_geometry_from_light_curve(
    params=(np.deg2rad(3.5), np.deg2rad(12.3), 0.898, 1.0, 0.739, 0.001),
    show_plots=False,
    ref_geoms=(0.102, 0.898, 0.539115831462, np.deg2rad(88.8888888889),
               0.0749139580237, 0.138957275514)):
    r"""Pytest style test for code/utils.py:
    model_geometry_from_light_curve

    """
    test_geoms = \
        code.utils.model_geometry_from_light_curve(
            params=params, show_plots=show_plots)
    assert np.all(np.isclose(ref_geoms, test_geoms))
    return None


# Additional cases for test_model_geometry_from_light_curve
test_model_geometry_from_light_curve(
    params=(0.164135455619, 0.165111260919, 0.0478630092323, 1.0,
            0.758577575029, 0.01),
    show_plots=False,
    ref_geoms=(0.952136990768, 0.0478630092323, 2.2458916679, np.deg2rad(90.0),
               0.000481306260183, 0.163880773527))


# TODO: update test when light curve parameters include phase, period.
def test_model_quantities_from_lc_velr_stellar(
    phase0=np.nan,
    period=271209600.0,
    lc_params=(0.164135455619, 0.165111260919,
               0.0478630092323, 1.0, 0.758577575029, np.nan),
    velr_b=33e3,
    stellar_b=(2.61291629258e+30, 760266000.0, 1.40922538433),
    ref_quants=(np.nan, 271209600.0, np.deg2rad(90.0), 1.55823297919e+12,
                2.324294844333284e+31,
                33e3, 1.42442349898e+12, 2.61291629258e+30, 760266000.0,
                1.40922538433,
                3.1e3, 1.33809480207e+11, 2.78149153727e+31, 258864241950.22577,
                1.0)):
    r"""Pytest style test for code/utils.py:
    model_quantities_from_light_curve_model

    """
    test_quants = \
        code.utils.model_quantities_from_lc_velr_stellar(
            phase0=phase0, period=period, lc_params=lc_params, velr_b=velr_b,
            stellar_b=stellar_b)
    # NOTE: remove equal_nan when phase0 is computed from light curve
    assert np.all(np.isclose(ref_quants, test_quants, equal_nan=True))
    return None
