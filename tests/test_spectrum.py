import math

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c

from jwst_ifu_analysis import Spectrum

__author__ = "bdrechsler"
__copyright__ = "bdrechsler"
__license__ = "MIT"

REST_WVL = 5.0
LINE_WIDTH = 0.06
SIGMA_WVL = LINE_WIDTH / 3.0
AMP = 3.0


@pytest.fixture
def continuum_plus_line_spectrum():
    """A spectrum with a linear continuum plus a Gaussian line, spanning a
    window much wider than the line itself."""
    wvl_axis = np.linspace(4.9, 5.1, 2001)
    cont_true = 2.0 + 0.5 * (wvl_axis - REST_WVL)
    line_profile = AMP * np.exp(-0.5 * ((wvl_axis - REST_WVL) / SIGMA_WVL) ** 2)
    flux = cont_true + line_profile
    return Spectrum(wvl_axis=wvl_axis, flux=flux), cont_true


@pytest.fixture
def isolated_gaussian_line():
    """A spectrum narrowed to just the line (as if extracted via
    Cube.spectral_region), with no continuum, for clean Gaussian-fit checks."""
    wvl_axis = np.linspace(REST_WVL - 5 * SIGMA_WVL, REST_WVL + 5 * SIGMA_WVL, 501)
    flux = AMP * np.exp(-0.5 * ((wvl_axis - REST_WVL) / SIGMA_WVL) ** 2)
    return Spectrum(wvl_axis=wvl_axis, flux=flux)


def test_fit_continuum_recovers_linear_continuum(continuum_plus_line_spectrum):
    spectrum, cont_true = continuum_plus_line_spectrum
    line = {"rest_wvl": REST_WVL, "line_width": LINE_WIDTH}

    cont_fit = spectrum.fit_continuum([line], med_kernel=3, fit_order=1)

    assert np.max(np.abs(cont_fit.flux - cont_true)) < 0.1 * AMP


def test_fit_gaussian_recovers_injected_parameters(isolated_gaussian_line):
    line = {"rest_wvl": REST_WVL, "line_width": LINE_WIDTH}

    fit_result = isolated_gaussian_line.fit_gaussian(line)

    amp, _ = fit_result["amp"]
    mean, _ = fit_result["mean"]
    std, _ = fit_result["std"]

    assert amp == pytest.approx(AMP, rel=1e-3)
    assert mean == pytest.approx(REST_WVL, rel=1e-3)
    assert std == pytest.approx(SIGMA_WVL, rel=1e-3)


def test_line_flux_fit_matches_analytic_gaussian_area(isolated_gaussian_line):
    line = {"rest_wvl": REST_WVL, "line_width": LINE_WIDTH}

    flux_fit, flux_fit_err = isolated_gaussian_line.line_flux(line, method="fit")

    sigma_nu = (SIGMA_WVL * u.um * c / (REST_WVL * u.um) ** 2).to(u.Hz)
    expected = (AMP * u.Jy * sigma_nu * np.sqrt(2 * np.pi)).to(u.erg / u.s / u.cm**2).value

    assert flux_fit == pytest.approx(expected, rel=1e-3)
    assert flux_fit_err > 0


def test_line_flux_integral_is_smaller_due_to_window_truncation(isolated_gaussian_line):
    line = {"rest_wvl": REST_WVL, "line_width": LINE_WIDTH}

    flux_fit, _ = isolated_gaussian_line.line_flux(line, method="fit")
    flux_integral = isolated_gaussian_line.line_flux(line, method="integral")

    # the integral method only sums within +/- line_width/2 (= 1.5 sigma),
    # so it should recover that fraction of the full analytic Gaussian area
    enclosed_fraction = math.erf(1.5 / math.sqrt(2))

    assert flux_integral < flux_fit
    assert flux_integral == pytest.approx(flux_fit * enclosed_fraction, rel=0.01)


def test_add_sums_matching_spectra():
    wvl_axis = np.linspace(4.9, 5.1, 11)
    s1 = Spectrum(wvl_axis=wvl_axis, flux=np.ones(11))
    s2 = Spectrum(wvl_axis=wvl_axis, flux=2 * np.ones(11))

    result = s1 + s2

    assert np.allclose(result.flux, 3 * np.ones(11))


def test_subtract_matching_spectra():
    wvl_axis = np.linspace(4.9, 5.1, 11)
    s1 = Spectrum(wvl_axis=wvl_axis, flux=3 * np.ones(11))
    s2 = Spectrum(wvl_axis=wvl_axis, flux=np.ones(11))

    result = s1 - s2

    assert np.allclose(result.flux, 2 * np.ones(11))


def test_add_mismatched_axes_returns_none(capsys):
    wvl_axis = np.linspace(4.9, 5.1, 11)
    s1 = Spectrum(wvl_axis=wvl_axis, flux=np.ones(11))
    s2 = Spectrum(wvl_axis=wvl_axis[:-1], flux=np.ones(10))

    result = s1 + s2

    assert result is None
    assert "not matched" in capsys.readouterr().out


def test_write_read_round_trip(tmp_path):
    wvl_axis = np.linspace(4.9, 5.1, 11)
    flux = np.arange(11, dtype=float)
    spectrum = Spectrum(wvl_axis=wvl_axis, flux=flux)

    fname = tmp_path / "spec.fits"
    spectrum.write(str(fname))
    round_tripped = Spectrum.read(str(fname))

    assert np.allclose(round_tripped.wvl_axis, wvl_axis)
    assert np.allclose(round_tripped.flux, flux)
