"""Dopplium Parser - Parse Dopplium radar data formats."""

from .parse_dopplium import parse_dopplium
from .parse_dopplium_raw import (
    parse_dopplium_raw,
    BodyHeader,
    FrameHeader,
)
from .parse_dopplium_rdch import (
    parse_dopplium_rdch,
    RDChBodyHeader,
    CPIHeader,
)
from .parse_dopplium_radarcube import (
    parse_dopplium_radarcube,
    RadarCubeBodyHeader,
    get_range_axis,
    get_velocity_axis,
    get_azimuth_axis,
    get_elevation_axis,
    get_processing_info,
    has_known_angles,
    uses_fft_for_angles,
)
from .parse_dopplium_header import FileHeader

__version__ = "1.1.0"
__all__ = [
    "parse_dopplium",
    "parse_dopplium_raw",
    "parse_dopplium_rdch",
    "parse_dopplium_radarcube",
    "FileHeader",
    "BodyHeader",
    "FrameHeader",
    "RDChBodyHeader",
    "RadarCubeBodyHeader",
    "CPIHeader",
    "get_range_axis",
    "get_velocity_axis",
    "get_azimuth_axis",
    "get_elevation_axis",
    "get_processing_info",
    "has_known_angles",
    "uses_fft_for_angles",
]

