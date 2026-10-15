.. image:: https://img.shields.io/pypi/v/linescape.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/linescape/

|

=================
linescape
=================

Tools for extracting, fitting, and visualizing JWST IFU (NIRSpec and MIRI MRS)
spectral data cubes.

``linescape`` provides a small class hierarchy built on top of
``astropy`` and ``photutils`` for common IFU analysis tasks:

- **Cube** — load a spectral data cube from a pipeline FITS file (or a raw
  array + header), extract a 1D spectrum through an aperture/region, slice
  out a spectral window, continuum-fit every spaxel, and mosaic two
  overlapping cubes together.
- **LineCube** — a ``Cube`` scoped to a single spectral line, with an
  associated velocity axis and moment-0/moment-1 map generation (either a
  manual calculation or via the `bettermoments
  <https://github.com/richteague/bettermoments>`_ package).
- **Spectrum** — a 1D wavelength/flux pair supporting continuum fitting,
  Gaussian line fitting, and line flux measurement (by Gaussian area or by
  numerical integration).

Installation
=============

.. code-block:: bash

    pip install linescape

Usage
=====

.. code-block:: python

    from linescape import Cube

    cube = Cube("stage3_s3d.fits")
    spectrum = cube.extract_spectrum("aperture.reg")

    line = {"rest_wvl": 12.81, "line_width": 0.02}
    line_flux, line_flux_err = spectrum.line_flux(line, method="fit")

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
