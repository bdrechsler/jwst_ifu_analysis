import numpy as np
from astropy.table import Table
from pathlib import Path

from .Line import Line

_REQUIRED_FIELDS = ("name", "rest_wvl", "line_width")
_OPTIONAL_FIELDS = ("species", "transition")
_ALL_FIELDS = _REQUIRED_FIELDS + _OPTIONAL_FIELDS


class LineList:
    """A collection of spectral lines

    Lines are sorted by rest wavelength. Every query method returns a
    new LineList (or a Line) and leaves the original untouched. All wavelengths
    are in microns.
    """

    def __init__(self, lines):
        """Create a LineList

        Args:
            lines (iterable of Line): the lines in the list. Names must be unique.

        Raises:
            TypeError: if any element is not a Line
            ValueError: if two lines share a name
        """
        lines = list(lines)
        for l in lines:
            if not isinstance(l, Line):
                raise TypeError(f"LineList elements must be Line objects, got {type(l).__name__}")

        names = [l.name for l in lines]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate line names: {duplicates}")

        self._lines = sorted(lines, key=lambda l: l.rest_wvl)
        self._by_name = {l.name: l for l in self._lines}

    # ------------------------------------------------------------------
    # container protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._lines)

    def __iter__(self):
        # makes LineList iteratble (iterate over lines)
        return iter(self._lines)

    def __getitem__(self, key):
        """Index by position (int), name (str), or slice

        Returns:
            Line for an int or str key, LineList for a slice
        """
        if isinstance(key, str):
            try:
                return self._by_name[key]
            except KeyError:
                raise KeyError(f"No line named '{key}'") from None
        if isinstance(key, slice):
            return LineList(self._lines[key])
        if isinstance(key, (int, np.integer)):
            return self._lines[key]
        raise TypeError(f"LineList indices must be int, str, or slice, not {type(key).__name__}")

    def __contains__(self, item):
        """Check for a Line (matched by name) or a line name"""
        if isinstance(item, Line):
            return item.name in self._by_name
        return item in self._by_name

    # check if two linelists are the same
    def __eq__(self, other):
        if not isinstance(other, LineList):
            return NotImplemented
        return self._lines == other._lines

    # how a line list get printed
    def __repr__(self):
        if not self._lines:
            return "LineList([])"
        rows = "\n".join(
            f"  {l.name}: rest_wvl={l.rest_wvl}, line_width={l.line_width}" for l in self._lines
        )
        return f"LineList([\n{rows}\n])"

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def in_range(self, wvl_min, wvl_max):
        """Get the lines with rest wavelengths strictly between two wavelengths

        Args:
            wvl_min (float): lower wavelength bound in microns
            wvl_max (float): upper wavelength bound in microns

        Returns:
            LineList: lines with wvl_min < rest_wvl < wvl_max
        """
        return LineList(l for l in self._lines if wvl_min < l.rest_wvl < wvl_max)

    def exclude(self, line):
        """Get the list without a given line

        Args:
            line (Line or str): the line, or its name, to remove. If it is not
                                in the list, an equivalent copy is returned.

        Returns:
            LineList: the list without the line
        """
        if isinstance(line, Line):
            name = line.name
        else:
            name = line

        return LineList(l for l in self._lines if l.name != name)

    def nearest(self, wvl):
        """Get the line closest to a wavelength

        Args:
            wvl (float): wavelength in microns

        Returns:
            Line: the line with rest wavelength closest to wvl

        Raises:
            ValueError: if the list is empty
        """
        if not self._lines:
            raise ValueError("Cannot find the nearest line in an empty LineList")
        return min(self._lines, key=lambda l: abs(l.rest_wvl - wvl))

    def mask(self, wvl_axis, exclude=None):
        """Get a mask that is False at channels covered by lines

        Only lines with rest wavelengths inside the range of wvl_axis are
        considered, and a channel is masked if it lies strictly inside a line's
        region (rest_wvl +/- line_width / 2).

        Args:
            wvl_axis (numpy.ndarray): wavelength axis in microns
            exclude (Line or str, optional): line, or line name, to leave unmasked
                                             (e.g., the central line of a cube)

        Returns:
            numpy.ndarray: boolean mask, True for line-free (continuum) channels
        """
        wvl_axis = np.asarray(wvl_axis)
        lines = self.in_range(np.min(wvl_axis), np.max(wvl_axis))
        if exclude is not None:
            lines = lines.exclude(exclude)

        line_mask = np.zeros(len(wvl_axis), dtype=bool)
        for l in lines:
            line_mask |= (wvl_axis > l.lo) & (wvl_axis < l.hi)
        return ~line_mask

    # ------------------------------------------------------------------
    # constructors and conversion
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, line_dict):
        """Create a LineList from a dictionary of dictionaries

        Args:
            line_dict (dict): maps line name to a dict of Line fields, e.g.
                              {"H2 S(1)": {"rest_wvl": 17.035, "line_width": 0.05}}

        Raises:
            ValueError: if an inner dict has unknown or missing fields
        """
        lines = []
        for name, info in line_dict.items():
            info = dict(info)
            info.setdefault("name", name)
            _check_fields(info.keys(), f"line '{name}'")
            lines.append(Line(**info))
        return cls(lines)

    def to_dict(self):
        """Convert to a dictionary of dictionaries keyed by line name"""
        out = {}
        for l in self._lines:
            out[l.name] = {
                "rest_wvl": l.rest_wvl,
                "line_width": l.line_width,
                "species": l.species,
                "transition": l.transition,
            }
        return out

    @classmethod
    def from_table(cls, table):
        """Create a LineList from an astropy Table with one row per line

        The table must have name, rest_wvl, and line_width columns and may have
        species and transition columns. Any other column raises an error.
        """
        _check_fields(table.colnames, "table")
        lines = []
        for row in table:
            info = {"name": str(row["name"]),
                    "rest_wvl": float(row["rest_wvl"]),
                    "line_width": float(row["line_width"])}
            for f in _OPTIONAL_FIELDS:
                if f in table.colnames:
                    # masked entries (e.g., empty strings read from FITS) become ""
                    info[f] = "" if row[f] is np.ma.masked else str(row[f])
            lines.append(Line(**info))
        return cls(lines)

    def to_table(self):
        """Convert to an astropy Table with one row per line"""
        return Table(
            {
                "name": [l.name for l in self._lines],
                "rest_wvl": [l.rest_wvl for l in self._lines],
                "line_width": [l.line_width for l in self._lines],
                "species": [l.species for l in self._lines],
                "transition": [l.transition for l in self._lines],
            },
            # keep column order stable (and valid when the list is empty)
            names=list(_ALL_FIELDS),
            dtype=[str, float, float, str, str],
        )

    @classmethod
    def read(cls, path, hdu=1):
        """Read a LineList from a FITS, CSV, or plain text file, chosen by file extension

        Args:
            path (str or Path): file to read (.fits, .fit, .fts, .csv, .txt, or .dat)
            hdu (int): HDU holding the table, only used for FITS files

        Raises:
            ValueError: if the file extension is not a supported type
        """
        suffix = Path(path).suffix.lower()
        if suffix in (".fits", ".fit", ".fts"):
            return cls.from_fits(path, hdu)
        if suffix == ".csv":
            return cls.from_csv(path)
        if suffix in (".txt", ".dat"):
            return cls.from_txt(path)
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path}. "
            "Supported: .fits, .fit, .fts, .csv, .txt, .dat"
        )

    def write(self, path, overwrite=False):
        """Write a LineList to a FITS, CSV, or plain text file, chosen by file extension

        Args:
            path (str or Path): file to write (.fits, .fit, .fts, .csv, .txt, or .dat)
            overwrite (bool): if True, will overwrite an existing file

        Raises:
            ValueError: if the file extension is not a supported type
        """
        suffix = Path(path).suffix.lower()
        if suffix in (".fits", ".fit", ".fts"):
            self.to_fits(path, overwrite=overwrite)
        elif suffix == ".csv":
            self.to_csv(path, overwrite=overwrite)
        elif suffix in (".txt", ".dat"):
            self.to_txt(path, overwrite=overwrite)
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}' for {path}. "
                "Supported: .fits, .fit, .fts, .csv, .txt, .dat"
            )

    @classmethod
    def from_fits(cls, path, hdu=1):
        """Read a LineList from a FITS table (see from_table for the columns)"""
        return cls.from_table(Table.read(path, hdu=hdu))

    def to_fits(self, path, overwrite=False):
        """Write the LineList to a FITS table"""
        self.to_table().write(path, format="fits", overwrite=overwrite)

    @classmethod
    def from_csv(cls, path):
        """Read a LineList from a CSV file with a header row (see from_table for the columns)"""
        # fill_values=[] keeps empty optional fields as "" instead of masking them
        return cls.from_table(Table.read(path, format="ascii.csv", fill_values=[]))

    @classmethod
    def from_txt(cls, path):
        """Read a LineList from a whitespace-delimited text file with a header row
        (see from_table for the columns)"""
        # fill_values=[] keeps empty optional fields as "" instead of masking them
        return cls.from_table(Table.read(path, format="ascii", fill_values=[]))

    def to_csv(self, path, overwrite=False):
        """Write the LineList to a CSV file with a header row"""
        self.to_table().write(path, format="ascii.csv", overwrite=overwrite)

    def to_txt(self, path, overwrite=False):
        """Write the LineList to a whitespace-delimited text file with a header row"""
        self.to_table().write(path, format="ascii", overwrite=overwrite)

    @classmethod
    def coerce(cls, obj):
        """Convert a LineList, dict of dicts, or iterable of Lines to a LineList

        A LineList is returned unchanged. Intended for functions that should
        accept any of these forms.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        return cls(obj)


def _check_fields(fields, where):
    """Raise ValueError if fields has unknown names or lacks required ones"""
    fields = set(fields)
    unknown = fields - set(_ALL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown field(s) in {where}: {sorted(unknown)}. Allowed: {list(_ALL_FIELDS)}")
    missing = set(_REQUIRED_FIELDS) - fields
    if missing:
        raise ValueError(f"Missing required field(s) in {where}: {sorted(missing)}")
