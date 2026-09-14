import numpy as np
import pytest
from astropy.constants import c

from jwst_ifu_analysis import LineCube

__author__ = "bdrechsler"
__copyright__ = "bdrechsler"
__license__ = "MIT"

NCHAN, NY, NX = 41, 5, 5
REST_WVL = 5.0
LINE_WIDTH = 0.06
SIGMA_WVL = LINE_WIDTH / 3.0
AMP = 5.0


@pytest.fixture
def line_and_list():
    line = {"rest_wvl": REST_WVL, "line_width": LINE_WIDTH}
    return line, [line]


@pytest.fixture
def gaussian_line_cube(cube_header_factory, line_and_list):
    line, line_list = line_and_list
    header = cube_header_factory(NCHAN, NY, NX, crval3=4.9, cdelt3=0.005)
    wvl_axis = (np.arange(NCHAN) * header["CDELT3"]) + header["CRVAL3"]
    profile = AMP * np.exp(-0.5 * ((wvl_axis - REST_WVL) / SIGMA_WVL) ** 2)

    data = np.zeros((NCHAN, NY, NX))
    for i in range(NCHAN):
        data[i, :, :] = profile[i]

    line_cube = LineCube(data=data, line=line, line_list=line_list, header=header)
    return line_cube, wvl_axis, profile


def test_vel_axis_matches_doppler_formula(gaussian_line_cube):
    line_cube, wvl_axis, _ = gaussian_line_cube

    dlam_um = wvl_axis - REST_WVL
    expected_vel = (c.to("km/s").value) * (dlam_um / REST_WVL)

    assert np.allclose(line_cube.vel_axis, expected_vel, rtol=1e-6)


def test_moment0_matches_analytic_integral(gaussian_line_cube):
    line_cube, _, profile = gaussian_line_cube

    dv = line_cube.vel_axis[1] - line_cube.vel_axis[0]
    expected_mom0 = np.nansum(profile) * dv

    mom0 = line_cube.moment0()

    assert mom0.shape == (NY, NX)
    assert mom0[NY // 2, NX // 2] == pytest.approx(expected_mom0, rel=1e-6)


def test_moment1_is_zero_for_line_centered_at_rest_wavelength(gaussian_line_cube):
    line_cube, _, _ = gaussian_line_cube

    mom1 = line_cube.moment1()

    assert mom1[NY // 2, NX // 2] == pytest.approx(0.0, abs=1e-6)


def test_spectral_region_preserves_line_and_list(gaussian_line_cube, line_and_list):
    line, line_list = line_and_list
    line_cube, _, _ = gaussian_line_cube

    sub_cube = line_cube.spectral_region(center_wvl=REST_WVL, region_width=10 * SIGMA_WVL)

    assert isinstance(sub_cube, LineCube)
    assert sub_cube.line == line
    assert sub_cube.line_list == line_list


def test_bm_collapse_zeroth_moment_smoke(gaussian_line_cube):
    line_cube, _, _ = gaussian_line_cube

    mom0 = line_cube.bm_collapse(order=0, smooth=None, clip=None)

    assert np.shape(mom0) == (2, NY, NX)
