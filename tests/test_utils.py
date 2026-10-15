import numpy as np

from linescape.utils import (
    get_chan,
    get_line_mask,
    get_wvl_axis,
    make_header_2d,
)

__author__ = "bdrechsler"
__copyright__ = "bdrechsler"
__license__ = "MIT"


def test_get_wvl_axis(cube_header_factory):
    header = cube_header_factory(5, 4, 4, crval3=1.0, cdelt3=0.1)

    wvl_axis = get_wvl_axis(header)

    assert np.allclose(wvl_axis, [1.0, 1.1, 1.2, 1.3, 1.4])


def test_get_line_mask_excludes_nearby_lines_but_not_central_line():
    wvl_axis = np.linspace(4.9, 5.1, 201)
    central_line = {"rest_wvl": 5.0, "line_width": 0.02}
    nearby_line = {"rest_wvl": 4.95, "line_width": 0.02}

    mask = get_line_mask([central_line, nearby_line], wvl_axis, central_line=central_line)

    in_nearby_window = (wvl_axis > 4.94) & (wvl_axis < 4.96)
    in_central_window = (wvl_axis > 4.99) & (wvl_axis < 5.01)

    assert not mask[in_nearby_window].any()
    assert mask[in_central_window].all()


def test_get_line_mask_no_nearby_lines_masks_nothing():
    wvl_axis = np.linspace(4.9, 5.1, 51)
    far_line = {"rest_wvl": 20.0, "line_width": 0.02}

    mask = get_line_mask([far_line], wvl_axis)

    assert mask.all()


def test_get_chan_boundaries():
    assert get_chan(5.0) == "nirspec"
    assert get_chan(7.0) == "ch1-long"
    assert get_chan(1.0) == 0


def test_make_header_2d_strips_spectral_axis(cube_header_factory):
    header = cube_header_factory(5, 4, 4)

    header_2d = make_header_2d(header)

    assert header_2d["NAXIS"] == 2
    assert header_2d["WCSAXES"] == 2
    for key in ("NAXIS3", "CRPIX3", "CDELT3", "CRVAL3", "CTYPE3", "CUNIT3", "DISPAXIS", "VELOSYS", "SPECSYS"):
        assert key not in header_2d
    assert header_2d["NAXIS1"] == 4
    assert header_2d["NAXIS2"] == 4
