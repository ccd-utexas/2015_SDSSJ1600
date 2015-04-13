#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for code/utils.py.

Notes
-----
Tests are executed using pytest.

"""


# Import standard packages.
from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(0, '.')
# Import installed packages.
import matplotlib.pyplot as plt
import numpy as np
# Import local packages.
import code


def test_plot_periodogram(
    periods=[1,2,4,8,16], powers=[1,2,4,2,1], xscale='log',
    n_terms=1, period_unit='seconds', flux_unit='relative', return_ax=True):
    """pytest style test for code.plot_periodogram
    
    """
    ax = \
      code.utils.plot_periodogram(
          periods=periods, powers=powers, xscale=xscale, n_terms=n_terms,
          period_unit=period_unit, flux_unit=flux_unit, return_ax=return_ax)
    assert isinstance(ax, plt.Axes)
    return None


def test_calc_periodogram(
    times=[1,2,3,4,5,6,7,8], fluxes=[0,1,0,1,0,1,0,1], fluxes_err=[1,1,1,1,1,1,1,1],
    min_period=None, max_period=None, num_periods=None, sigs=(95.0, 99.0), num_bootstraps=100,
    show_periodogram=False, period_unit='seconds', flux_unit='relative',
    ref_periods=[3.5, 3.11111111, 2.8, 2.54545455, 2.33333333, 2.15384615, 2.0],
    ref_powers=[0.05615773, 0.09897668, 0.02777778, 0.06395519, 0.68555962, 0.98455515, 1.45655044],
    ref_sigs_powers=[(95.0, 0.92877340237699546), (99.0, 1.0210555725004953)]):
    """pytest style test for code.calc_periodogram

    """
    (test_periods, test_powers, test_sigs_powers) = \
        code.utils.calc_periodogram(
            times=times, fluxes=fluxes, fluxes_err=fluxes_err, min_period=min_period,
            max_period=max_period, num_periods=num_periods, sigs=sigs, num_bootstraps=num_bootstraps,
            show_periodogram=show_periodogram, period_unit=period_unit, flux_unit=flux_unit)
    assert np.all(np.isclose(ref_periods, test_periods))
    assert np.all(np.isclose(ref_powers, test_powers))
    assert np.all(np.isclose(ref_sigs_powers, test_sigs_powers))
    return None


def test_select_sig_periods_powers(
    peak_periods=[1, 2, 3], peak_powers=[0.1, 0.2, 0.3], cutoff_power=0.15,
    ref_sig_periods=[2, 3], ref_sig_powers=[0.2, 0.3]):
    """pytest style test for code.select_sig_periods_powers

    """
    (test_sig_periods, test_sig_powers) = \
        code.utils.select_sig_periods_powers(
            peak_periods=peak_periods, peak_powers=peak_powers, cutoff_power=cutoff_power)
    assert np.all(np.isclose(ref_sig_periods, test_sig_periods))
    assert np.all(np.isclose(ref_sig_powers, test_sig_powers))
    return None


def test_calc_best_period(
    times=[1,2,3,4,5,6,7,8], fluxes=[0,1,0,1,0,1,0,1], fluxes_err=[1,1,1,1,1,1,1,1],
    candidate_periods=[2.15384615, 2.], n_terms=1, show_periodograms=False,
    show_summary_plots=False, period_unit='seconds', flux_unit='relative',
    ref_best_period=2.0):
    """pytest style test for code.utils.calc_best_period

    """
    test_best_period = \
        code.utils.calc_best_period(
            times=times, fluxes=fluxes, fluxes_err=fluxes_err, candidate_periods=candidate_periods,
            n_terms=n_terms, show_periodograms=show_periodograms, show_summary_plots=show_summary_plots,
            period_unit=period_unit, flux_unit=flux_unit)
    assert np.isclose(test_best_period, ref_best_period)
    return None


def test_calc_num_terms():
    pass

def test_plot_phased_light_curve():
    pass

def test_refine_best_period():
    pass

def test_calc_flux_fits_residuals():
    pass

def test_calc_z1_z2():
    pass

def test_plot_phased_histogram():
    pass

def test_calc_phased_histogram():
    pass

