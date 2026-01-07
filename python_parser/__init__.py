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
from .parse_dopplium_detections import (
    parse_dopplium_detections,
    DetectionsBodyHeader,
    DetectionsBatchHeader,
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude,
    get_detection_statistics,
)
from .parse_dopplium_header import FileHeader

__version__ = "1.2.0"
__all__ = [
    "parse_dopplium",
    "parse_dopplium_raw",
    "parse_dopplium_rdch",
    "parse_dopplium_radarcube",
    "parse_dopplium_detections",
    "FileHeader",
    "BodyHeader",
    "FrameHeader",
    "RDChBodyHeader",
    "RadarCubeBodyHeader",
    "CPIHeader",
    "DetectionsBodyHeader",
    "DetectionsBatchHeader",
    "get_range_axis",
    "get_velocity_axis",
    "get_azimuth_axis",
    "get_elevation_axis",
    "get_processing_info",
    "has_known_angles",
    "uses_fft_for_angles",
    "filter_detections_by_range",
    "filter_detections_by_velocity",
    "filter_detections_by_amplitude",
    "get_detection_statistics",
]

