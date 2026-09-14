=========
Changelog
=========

Version 0.1.0
=============

- ``Cube``: load spectral cubes from FITS or raw arrays, extract 1D spectra
  through an aperture/region, slice spectral windows, fit per-spaxel
  continua, and mosaic overlapping cubes (``combine_cubes``).
- ``LineCube``: ``Cube`` subclass scoped to a single spectral line, with a
  velocity axis and moment-0/moment-1 map generation (manual or via
  ``bettermoments``).
- ``Spectrum``: 1D wavelength/flux container with continuum fitting,
  Gaussian line fitting, and line flux measurement (fit or integral method).
- ``utils``/``plotting``: shared helpers for wavelength axes, line masking,
  2D header generation, and WCS-aware plot annotations.
