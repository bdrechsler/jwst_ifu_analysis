import numpy as np
import astropy.units as u
import photutils.aperture as ap
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import CircleSkyRegion

from linescape import Cube

__author__ = "bdrechsler"
__copyright__ = "bdrechsler"
__license__ = "MIT"


def test_mjy_per_sr_converted_to_jy(cube_header_factory):
    header = cube_header_factory(3, 4, 4, bunit="MJy/sr", pixar_sr=1e-13)
    data = np.full((3, 4, 4), 1.0)

    cube = Cube(data=data, header=header)

    assert cube.header["BUNIT"] == "Jy"
    assert cube.data[0, 0, 0] == pytest.approx(1.0e6 * 1e-13)


def test_jy_left_unchanged(small_cube):
    data, header = small_cube
    cube = Cube(data=data.copy(), header=header)

    assert cube.header["BUNIT"] == "Jy"
    assert np.array_equal(cube.data, data)


def test_load_from_fits_file(small_cube, small_cube_fits_file):
    data, header = small_cube
    cube = Cube(small_cube_fits_file)

    assert cube.shape == data.shape
    assert cube.data[0, 0, 0] == pytest.approx(data[0, 0, 0])


def test_spectral_region_slices_and_updates_header(small_cube):
    data, header = small_cube
    cube = Cube(data=data.copy(), header=header)

    center_wvl = header["CRVAL3"] + 2 * header["CDELT3"]
    region_width = 2 * header["CDELT3"]
    sub_cube = cube.spectral_region(center_wvl=center_wvl, region_width=region_width)

    wvl_min = center_wvl - region_width / 2.0
    expected_nchan = np.sum((cube.wvl_axis > wvl_min) & (cube.wvl_axis < wvl_min + region_width))

    assert sub_cube.shape == (expected_nchan, data.shape[1], data.shape[2])
    assert sub_cube.header["NAXIS3"] == expected_nchan
    assert sub_cube.header["CRVAL3"] == pytest.approx(wvl_min)


def test_extract_spectrum_matches_analytic_aperture_area(small_cube):
    data, header = small_cube
    flux_value = data[0, 0, 0]
    cube = Cube(data=data.copy(), header=header)

    radius = 3 * u.arcsec
    region = CircleSkyRegion(
        center=SkyCoord(ra=header["CRVAL1"], dec=header["CRVAL2"], unit="deg"),
        radius=radius,
    )

    spectrum = cube.extract_spectrum(region)

    pix_ap = ap.region_to_aperture(region).to_pixel(cube.wcs.celestial)
    expected_area = pix_ap.to_mask(method="exact").get_values(np.ones(data.shape[1:])).sum()
    expected_flux = flux_value * expected_area

    assert spectrum.wvl_axis.shape == cube.wvl_axis.shape
    assert np.allclose(spectrum.flux, expected_flux, rtol=1e-6)


def test_write_round_trips(tmp_path, small_cube):
    data, header = small_cube
    cube = Cube(data=data.copy(), header=header)

    fname = tmp_path / "out.fits"
    cube.write(str(fname))

    written_data, written_header = fits.getdata(str(fname), header=True)
    assert np.array_equal(written_data, cube.data)
    assert written_header["BUNIT"] == "Jy"
