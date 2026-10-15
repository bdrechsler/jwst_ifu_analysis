"""Shared fixtures for linescape tests.

Builds small synthetic IFU-cube headers/data in memory so tests don't need
real JWST pipeline products.
"""

import numpy as np
import pytest
from astropy.io import fits

__author__ = "bdrechsler"
__copyright__ = "bdrechsler"
__license__ = "MIT"


def make_cube_header(
    nchan, ny, nx, crval3=5.0, cdelt3=0.01, bunit="Jy", pixar_sr=None
):
    """Build a minimal but WCS-valid 3D IFU cube header."""
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["NAXIS3"] = nchan
    header["WCSAXES"] = 3
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CRPIX3"] = 1.0
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = 20.0
    header["CRVAL3"] = crval3
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["CDELT3"] = cdelt3
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CTYPE3"] = "WAVE"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "um"
    header["PC1_1"] = 1.0
    header["PC1_2"] = 0.0
    header["PC1_3"] = 0.0
    header["PC2_1"] = 0.0
    header["PC2_2"] = 1.0
    header["PC2_3"] = 0.0
    header["PC3_1"] = 0.0
    header["PC3_2"] = 0.0
    header["PC3_3"] = 1.0
    header["DISPAXIS"] = 3
    header["VELOSYS"] = 0.0
    header["SPECSYS"] = "BARYCENT"
    header["BUNIT"] = bunit
    if pixar_sr is not None:
        header["PIXAR_SR"] = pixar_sr
    return header


@pytest.fixture
def cube_header_factory():
    """Factory fixture: call with (nchan, ny, nx, **kwargs) to get a header."""
    return make_cube_header


@pytest.fixture
def small_cube(cube_header_factory):
    """A small uniform-flux cube (5 channels, 21x21 spaxels) for aperture tests."""
    nchan, ny, nx = 5, 21, 21
    header = cube_header_factory(nchan, ny, nx)
    data = np.full((nchan, ny, nx), 2.0)
    return data, header


@pytest.fixture
def small_cube_fits_file(tmp_path, small_cube):
    """Write `small_cube` to a FITS file with a SCI extension, as JWST pipeline
    products are structured, to exercise Cube's file-loading path."""
    data, header = small_cube
    fname = tmp_path / "cube.fits"
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=header, name="SCI")]
    )
    hdul.writeto(fname)
    return str(fname)
