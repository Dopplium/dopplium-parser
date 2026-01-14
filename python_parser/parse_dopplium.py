"""
Main dispatcher for parsing Dopplium binary files.
Automatically detects the file version and message type, then routes to the appropriate parser.

Supported message types:
  Version 2:
    0 - Unknown (unsupported)
    1 - Detections (not yet implemented)
    2 - Tracks (not yet implemented)
    3 - RawData/ADC (supported)
    4 - Aggregated (not yet implemented)
  Version 3:
    0 - Unknown (unsupported)
    1 - ADCData (supported via RawData parser)
    2 - RDCMaps/RDCh (supported)
    3 - RadarCube (supported)
    4 - Detections (supported)
    5 - Blobs (not yet implemented)
    6 - Tracks (supported)
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional

import numpy as np

from .parse_dopplium_header import parse_file_header
from .parse_dopplium_raw import parse_dopplium_raw
from .parse_dopplium_rdch import parse_dopplium_rdch


def parse_dopplium(
    filename: str,
    *,
    max_cpis_or_frames: Optional[int] = None,
    cast: str = "float32",
    return_complex: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium binary file, automatically detecting the format.
    
    This function reads the file header to determine the file version and message type,
    then dispatches to the appropriate parser using version-specific message type mappings.
    
    Supported combinations:
    - Version 2, message_type 3: RawData/ADC -> parse_dopplium_raw
    - Version 3, message_type 1: ADCData -> parse_dopplium_raw
    - Version 3, message_type 2: RDCMaps/RDCh -> parse_dopplium_rdch
    - Version 3, message_type 3: RadarCube -> parse_dopplium_radarcube
    - Version 3, message_type 4: Detections -> parse_dopplium_detections
    - Version 3, message_type 6: Tracks -> parse_dopplium_tracks
    
    Parameters:
    -----------
    filename : str
        Path to the Dopplium binary file
    max_cpis_or_frames : int, optional
        Maximum number of CPIs/frames to read (None = all)
    cast : str
        Data type for output ('float32', 'float64', 'int16')
        Only used for RawData parsing
    return_complex : bool
        Whether to return complex data
        Only used for RawData parsing
    verbose : bool
        Print parsing information
    
    Returns:
    --------
    tuple : (data, headers)
        data : np.ndarray
            Parsed data array (shape depends on message type)
            - RawData: [samples, chirpsPerTx, channels, frames]
            - RDCh: [range_bins, doppler_bins, channels, cpis]
        headers : dict
            Dictionary containing parsed headers
    
    Raises:
    -------
    ValueError
        If the version or message type is not supported
    """
    # Parse file header to determine version and message type
    file_header, endian_prefix = parse_file_header(filename)
    
    if verbose:
        print(f"Detected file version: {file_header.version}, message_type: {file_header.message_type}")
    
    # Dispatch based on version and message type
    version = file_header.version
    msg_type = file_header.message_type
    
    # Version 2 message type mappings
    if version == 2:
        if msg_type == 0:
            raise ValueError("File has unknown message_type (0). Cannot parse.")
        elif msg_type == 1:
            raise NotImplementedError("Version 2 Detections (message_type=1) not yet implemented.")
        elif msg_type == 2:
            raise NotImplementedError("Version 2 Tracks (message_type=2) not yet implemented.")
        elif msg_type == 3:
            # RawData/ADC
            if verbose:
                print("Routing to RawData/ADC parser...")
            return parse_dopplium_raw(
                filename,
                max_frames=max_cpis_or_frames,
                cast=cast,
                return_complex=return_complex,
                verbose=verbose,
                _file_header=file_header,
                _endian_prefix=endian_prefix
            )
        elif msg_type == 4:
            raise NotImplementedError("Version 2 Aggregated (message_type=4) not yet implemented.")
        else:
            raise ValueError(f"Unsupported Version 2 message_type: {msg_type}")
    
    # Version 3 message type mappings
    elif version == 3:
        if msg_type == 0:
            raise ValueError("File has unknown message_type (0). Cannot parse.")
        elif msg_type == 1:
            # ADCData
            if verbose:
                print("Routing to ADCData parser...")
            return parse_dopplium_raw(
                filename,
                max_frames=max_cpis_or_frames,
                cast=cast,
                return_complex=return_complex,
                verbose=verbose,
                _file_header=file_header,
                _endian_prefix=endian_prefix
            )
        elif msg_type == 2:
            # RDCMaps/RDCh
            if verbose:
                print("Routing to RDCMaps/RDCh parser...")
            return parse_dopplium_rdch(
                filename,
                max_cpis=max_cpis_or_frames,
                verbose=verbose,
                _file_header=file_header,
                _endian_prefix=endian_prefix
            )
        elif msg_type == 3:
            # RadarCube
            from .parse_dopplium_radarcube import parse_dopplium_radarcube
            return parse_dopplium_radarcube(
                filename,
                max_cpis=max_cpis_or_frames,
                verbose=verbose,
                _file_header=file_header,
                _endian_prefix=endian_prefix
            )
        elif msg_type == 4:
            # Detections
            if verbose:
                print("Routing to Detections parser...")
            from .parse_dopplium_detections import parse_dopplium_detections
            return parse_dopplium_detections(
                filename,
                max_batches=max_cpis_or_frames,
                verbose=verbose,
                _file_header=file_header,
                _endian_prefix=endian_prefix
            )
        elif msg_type == 5:
            raise NotImplementedError("Version 3 Blobs (message_type=5) not yet implemented.")
        elif msg_type == 6:
            # Tracks
            if verbose:
                print("Routing to Tracks parser...")
            from .parse_dopplium_tracks import parse_dopplium_tracks
            return parse_dopplium_tracks(
                filename,
                max_frames=max_cpis_or_frames,
                verbose=verbose,
                _file_header=file_header,
                _endian_prefix=endian_prefix
            )
        else:
            raise ValueError(f"Unsupported Version 3 message_type: {msg_type}")
    
    else:
        raise ValueError(f"Unsupported file version: {version}. Supported versions: 2, 3")


# Re-export individual parsers for direct use
__all__ = [
    'parse_dopplium',
    'parse_dopplium_raw',
    'parse_dopplium_rdch',
    'parse_dopplium_radarcube',
    'parse_dopplium_detections',
    'parse_dopplium_tracks',
]

