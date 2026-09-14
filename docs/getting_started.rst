===============
Getting Started
===============

Installation
=============

.. code-block:: bash

    pip install jwst_ifu_analysis

Loading a cube
==============

A :class:`~jwst_ifu_analysis.Cube.Cube` wraps a JWST pipeline spectral cube
(a Stage 3 ``*_s3d.fits`` product). Pass it a filename and it reads the
``SCI`` extension's data and header:

.. code-block:: python

    from jwst_ifu_analysis import Cube

    cube = Cube("stage3_s3d.fits")

If the header's ``BUNIT`` is ``MJy/sr``, the data is automatically converted
to ``Jy`` on load using the ``PIXAR_SR`` keyword, so downstream flux
calculations are always in Jy.

Extracting a 1D spectrum
=========================

:meth:`~jwst_ifu_analysis.Cube.Cube.extract_spectrum` sums the flux within an
aperture, at every wavelength channel, to produce a
:class:`~jwst_ifu_analysis.Spectrum.Spectrum`. The aperture is defined by a
region — either a path to a DS9/CRTF region file, or a
:mod:`regions` region object:

.. code-block:: python

    spectrum = cube.extract_spectrum("aperture.reg")

Fitting and subtracting the continuum
======================================

Spectral lines are described throughout the package as plain ``dict``\ s
with ``rest_wvl`` and ``line_width`` keys (both in microns), and a "line
list" is just a list of such dicts — used to mask out line emission when
fitting the continuum:

.. code-block:: python

    line = {"rest_wvl": 12.81, "line_width": 0.05}
    line_list = [line]  # add any other nearby lines here so they get masked too

    # narrow down to a window around the line of interest
    region = spectrum.spectral_region(center_wvl=line["rest_wvl"], region_width=0.3)

    continuum = region.fit_continuum(line_list)
    line_spectrum = region - continuum

Measuring a line flux
======================

:meth:`~jwst_ifu_analysis.Spectrum.Spectrum.line_flux` supports two methods:
fitting a Gaussian and taking its analytic area (``method="fit"``), or
numerically integrating the flux within the line window
(``method="integral"``, which only captures the flux within
``rest_wvl`` :math:`\pm` ``line_width / 2``). Fit the Gaussian to a spectrum
that's already narrowed to just the line, so the continuum doesn't bias the
fit:

.. code-block:: python

    line_flux, line_flux_err = line_spectrum.line_flux(line, method="fit")

Moment maps with a LineCube
=============================

A :class:`~jwst_ifu_analysis.LineCube.LineCube` is a
:class:`~jwst_ifu_analysis.Cube.Cube` scoped to a single line: it blanks out
channels covered by other nearby lines in ``line_list``, and computes a
velocity axis relative to ``rest_wvl``. It takes the same ``data``/file
argument as ``Cube``, plus the line and line list:

.. code-block:: python

    from jwst_ifu_analysis import LineCube

    line_cube = LineCube("stage3_s3d.fits", line=line, line_list=line_list)

Moment-0 (integrated intensity) and moment-1 (intensity-weighted velocity)
maps can be computed directly, optionally writing the result to a FITS file:

.. code-block:: python

    mom0 = line_cube.moment0(fname="mom0.fits")
    mom1 = line_cube.moment1(fname="mom1.fits")

For noisier data, :meth:`~jwst_ifu_analysis.LineCube.LineCube.bm_collapse`
delegates to the `bettermoments <https://github.com/richteague/bettermoments>`_
package instead, which smooths and sigma-clips the cube before collapsing it
and also returns an uncertainty map:

.. code-block:: python

    mom0, mom0_err = line_cube.bm_collapse(order=0, smooth=3, clip=3.0)

See the :doc:`Module Reference <api/modules>` for the full API, including
mosaicking overlapping cubes with
:func:`~jwst_ifu_analysis.Cube.combine_cubes`.
