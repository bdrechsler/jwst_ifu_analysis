from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    """A single spectral line

    All wavelengths are in microns.

    Attributes:
        name (str): unique identifier for the line (e.g., "H2 S(1)")
        rest_wvl (float): rest wavelength of the line in microns
        line_width (float): full width of the line region in microns, used for
                            masking the line and sizing line cubes
        species (str, optional): species producing the line (e.g., "H2")
        transition (str, optional): transition label (e.g., "S(1)")
    """

    name: str
    rest_wvl: float
    line_width: float
    species: str = ""
    transition: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("Line name must be a non-empty string")
        if self.rest_wvl <= 0:
            raise ValueError(f"{self.name}: rest_wvl must be positive, got {self.rest_wvl}")
        if self.line_width <= 0:
            raise ValueError(f"{self.name}: line_width must be positive, got {self.line_width}")

    @property
    def lo(self):
        """float: lower wavelength bound of the line region (microns)"""
        return self.rest_wvl - 0.5 * self.line_width

    @property
    def hi(self):
        """float: upper wavelength bound of the line region (microns)"""
        return self.rest_wvl + 0.5 * self.line_width

    def contains(self, wvl):
        """Check whether a wavelength falls within the line region

        Args:
            wvl (float or np.ndarray): wavelength(s) in microns

        Returns:
            bool or np.ndarray: True where lo <= wvl <= hi
        """
        return (wvl >= self.lo) & (wvl <= self.hi)
