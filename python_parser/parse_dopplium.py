"""
Main dispatcher for parsing Dopplium binary files.
Automatically detects the message type and routes to the appropriate parser.
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
    
    This function reads the file header to determine the message type,
    then dispatches to the appropriate parser:
    - message_type == 3: RawData -> parse_dopplium_raw
    - message_type == 5: RDCh -> parse_dopplium_rdch
    
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
        If the message type is not supported
    """
    # Parse file header to determine message type
    file_header, endian_prefix = parse_file_header(filename)
    
    if verbose:
        print(f"Detected message_type: {file_header.message_type}")
    
    # Dispatch to appropriate parser based on message type
    if file_header.message_type == 3:
        # RawData
        if verbose:
            print("Routing to RawData parser...")
        return parse_dopplium_raw(
            filename,
            max_frames=max_cpis_or_frames,
            cast=cast,
            return_complex=return_complex,
            verbose=verbose,
            _file_header=file_header,
            _endian_prefix=endian_prefix
        )
    
    elif file_header.message_type == 5:
        # RDCh
        if verbose:
            print("Routing to RDCh parser...")
        return parse_dopplium_rdch(
            filename,
            max_cpis=max_cpis_or_frames,
            verbose=verbose,
            _file_header=file_header,
            _endian_prefix=endian_prefix
        )
    
    else:
        raise ValueError(
            f"Unsupported message_type: {file_header.message_type}. "
            f"Supported types: 3 (RawData), 5 (RDCh)"
        )


# Re-export individual parsers for direct use
__all__ = [
    'parse_dopplium',
    'parse_dopplium_raw',
    'parse_dopplium_rdch',
]

