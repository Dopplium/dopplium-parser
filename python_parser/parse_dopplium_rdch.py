"""
Parser for Dopplium RDCh (Range-Doppler-Channel) binary format.
Reads processed radar data files written by RDChBinaryWriter.

Returns numpy arrays shaped [range, doppler, channels, chunks] and all headers.
"""

from __future__ import annotations
import io
import struct
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np

from .parse_dopplium_header import FileHeader, parse_file_header


# ==============================
# Dataclasses for headers
# ==============================


@dataclass
class RDChBodyHeader:
    """RDCh body header (256 bytes) containing configuration and processing parameters."""
    config_magic: str
    config_version: int
    body_header_size: int
    chunk_header_size: int
    reserved1: int
    # Dimensions
    n_range_bins: int
    n_doppler_bins: int
    n_channels: int
    # Range parameters
    range_min_m: float
    range_max_m: float
    range_resolution_m: float
    # Velocity parameters
    velocity_min_mps: float
    velocity_max_mps: float
    velocity_resolution_mps: float
    # Radar parameters
    start_freq_ghz: float
    bandwidth_ghz: float
    sample_rate_ksps: float
    frame_period_ms: float
    # Data type
    data_type: int  # 0=complex64, 1=complex128, 2=float32, 3=float64, 4=int16, 5=int32
    # Processing parameters
    nfft_range: int
    nfft_doppler: int
    n_samples_original: int
    n_chirps_original: int
    range_window_type: int  # 0=none, 1=hann, 2=hamming, 3=blackman, 4=kaiser
    doppler_window_type: int
    fftshift_range: int  # 0=no, 1=yes
    fftshift_doppler: int
    range_half_spectrum: int
    n_accumulated_frames: int
    _reserved3: bytes


@dataclass
class ChunkHeader:
    """RDCh chunk header (22 bytes)."""
    chunk_magic: str
    chunk_header_size: int
    chunk_timestamp_utc_ticks: int
    chunk_number: int
    chunk_payload_size: int


# ==============================
# Public API
# ==============================

def parse_dopplium_rdch(
    filename: str,
    *,
    max_chunks: Optional[int] = None,
    verbose: bool = True,
    _file_header: Optional[FileHeader] = None,
    _endian_prefix: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parse a Dopplium RDCh file into a numpy array shaped [range, doppler, channels, chunks].
    
    Parameters:
    -----------
    filename : str
        Path to the RDCh binary file
    max_chunks : int, optional
        Maximum number of chunks to read (None = all chunks)
    verbose : bool
        Print parsing information
    _file_header : FileHeader, optional
        Pre-parsed file header (internal use)
    _endian_prefix : str, optional
        Endianness prefix (internal use)
    
    Returns:
    --------
    tuple : (data, headers)
        data : np.ndarray
            Processed radar data [range_bins, doppler_bins, channels, chunks]
        headers : dict
            Dictionary containing 'file', 'body', and 'chunk' headers
    """
    with open(filename, "rb") as f:
        # Use provided header or parse it
        if _file_header is None or _endian_prefix is None:
            from .parse_dopplium_header import parse_file_header as parse_fh
            FH, endian_prefix = parse_fh(filename)
        else:
            FH = _file_header
            endian_prefix = _endian_prefix
            # Seek to start of file to read headers
            f.seek(0, io.SEEK_SET)
            # Skip the file header
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        if FH.message_type != 5:
            raise ValueError(f"This file is not RDCh (message_type={FH.message_type}, expected 5).")
        
        # Position at body header if we parsed the file header ourselves
        if _file_header is None:
            f.seek(FH.file_header_size, io.SEEK_SET)
        
        BH = _read_rdch_body_header(f, endian_prefix)
        
        if verbose:
            _print_header_summary(FH, BH)
        
        # Extract dimensions
        n_range = int(BH.n_range_bins)
        n_doppler = int(BH.n_doppler_bins)
        n_channels = int(BH.n_channels)
        
        # Map data type code to numpy dtype
        dtype = _map_data_type(BH.data_type)
        
        # Determine number of chunks from file size
        file_size = _file_size(f)
        bytes_after_headers = file_size - FH.file_header_size - BH.body_header_size
        chunk_unit = FH.frame_header_size + (n_range * n_doppler * n_channels * dtype.itemsize)
        
        # Calculate expected payload size per chunk
        expected_payload_size = n_range * n_doppler * n_channels * dtype.itemsize
        
        if verbose:
            print(f"\nExpected payload size per chunk: {expected_payload_size} bytes")
            print(f"Bytes after headers: {bytes_after_headers}")
        
        # Estimate number of chunks
        # We'll read chunk by chunk since payload sizes might vary
        n_chunks_estimate = max(0, bytes_after_headers // chunk_unit)
        
        if verbose:
            print(f"Estimated chunks in file: {n_chunks_estimate}")
        
        n_chunks = n_chunks_estimate if max_chunks is None else min(n_chunks_estimate, max_chunks)
        
        # Allocate output array
        # We'll start with estimated size and potentially resize if needed
        data = np.zeros((n_range, n_doppler, n_channels, n_chunks), dtype=dtype)
        
        # Read chunks
        f.seek(FH.file_header_size + BH.body_header_size, io.SEEK_SET)
        chunk_headers = []
        
        chunks_read = 0
        while chunks_read < n_chunks:
            try:
                # Check if we have enough bytes left
                current_pos = f.tell()
                if current_pos >= file_size:
                    break
                
                CH = _read_chunk_header(f, endian_prefix)
                chunk_headers.append(CH)
                
                if verbose and (chunks_read == 0 or (chunks_read + 1) % 10 == 0):
                    print(f"  Reading chunk {chunks_read + 1}/{n_chunks}: "
                          f"number={CH.chunk_number}, size={CH.chunk_payload_size} bytes")
                
                # Validate chunk payload size
                if CH.chunk_payload_size != expected_payload_size:
                    if verbose:
                        print(f"Warning: Chunk {chunks_read} payload size mismatch: "
                              f"expected={expected_payload_size}, got={CH.chunk_payload_size}")
                
                # Read payload
                payload_bytes = f.read(CH.chunk_payload_size)
                if len(payload_bytes) != CH.chunk_payload_size:
                    raise EOFError(f"Unexpected EOF while reading chunk {chunks_read} payload.")
                
                # Reshape payload to [range, doppler, channels]
                chunk_data = np.frombuffer(payload_bytes, dtype=dtype)
                
                # Reshape from flat array to 3D
                try:
                    chunk_data = chunk_data.reshape((n_range, n_doppler, n_channels), order='C')
                    data[:, :, :, chunks_read] = chunk_data
                except ValueError as e:
                    raise ValueError(f"Chunk {chunks_read}: Cannot reshape payload. "
                                   f"Expected {n_range}×{n_doppler}×{n_channels} = "
                                   f"{n_range * n_doppler * n_channels} elements, "
                                   f"got {chunk_data.size}. Error: {e}")
                
                chunks_read += 1
                
            except EOFError:
                if verbose:
                    print(f"Reached end of file after reading {chunks_read} chunks.")
                break
        
        # Trim data array if we read fewer chunks than estimated
        if chunks_read < n_chunks:
            data = data[:, :, :, :chunks_read]
        
        headers = {
            "file": FH,
            "body": BH,
            "chunk": chunk_headers,
        }
        
        if verbose:
            print(f"\nParsed data shape: {tuple(data.shape)}  "
                  f"[range_bins, doppler_bins, channels, chunks]")
            print(f"Total chunks read: {chunks_read}")
        
        return data, headers


# ==============================
# Helpers
# ==============================

def _file_size(fh: io.BufferedReader) -> int:
    """Get file size without changing current position."""
    cur = fh.tell()
    fh.seek(0, io.SEEK_END)
    size = fh.tell()
    fh.seek(cur, io.SEEK_SET)
    return size


def _read_rdch_body_header(f: io.BufferedReader, ep: str) -> RDChBodyHeader:
    """Read RDCh body header (256 bytes)."""
    fmt = (
        f"{ep}"
        "4s"   # config_magic (RDCH)
        "H"    # config_version
        "H"    # body_header_size
        "H"    # chunk_header_size
        "H"    # reserved1
        "I"    # n_range_bins
        "I"    # n_doppler_bins
        "I"    # n_channels
        "f"    # range_min_m
        "f"    # range_max_m
        "f"    # range_resolution_m
        "f"    # velocity_min_mps
        "f"    # velocity_max_mps
        "f"    # velocity_resolution_mps
        "f"    # start_freq_ghz
        "f"    # bandwidth_ghz
        "f"    # sample_rate_ksps
        "f"    # frame_period_ms
        "B"    # data_type
        "I"    # nfft_range
        "I"    # nfft_doppler
        "I"    # n_samples_original
        "I"    # n_chirps_original
        "B"    # range_window_type
        "B"    # doppler_window_type
        "B"    # fftshift_range
        "B"    # fftshift_doppler
        "B"    # range_half_spectrum
        "I"    # n_accumulated_frames
        "166s" # reserved3
    )
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError(f"Failed to read body header. Expected {size} bytes, got {len(raw)}.")
    
    unpacked = struct.unpack(fmt, raw)
    
    return RDChBodyHeader(
        config_magic=unpacked[0].decode("ascii"),
        config_version=unpacked[1],
        body_header_size=unpacked[2],
        chunk_header_size=unpacked[3],
        reserved1=unpacked[4],
        n_range_bins=unpacked[5],
        n_doppler_bins=unpacked[6],
        n_channels=unpacked[7],
        range_min_m=unpacked[8],
        range_max_m=unpacked[9],
        range_resolution_m=unpacked[10],
        velocity_min_mps=unpacked[11],
        velocity_max_mps=unpacked[12],
        velocity_resolution_mps=unpacked[13],
        start_freq_ghz=unpacked[14],
        bandwidth_ghz=unpacked[15],
        sample_rate_ksps=unpacked[16],
        frame_period_ms=unpacked[17],
        data_type=unpacked[18],
        nfft_range=unpacked[19],
        nfft_doppler=unpacked[20],
        n_samples_original=unpacked[21],
        n_chirps_original=unpacked[22],
        range_window_type=unpacked[23],
        doppler_window_type=unpacked[24],
        fftshift_range=unpacked[25],
        fftshift_doppler=unpacked[26],
        range_half_spectrum=unpacked[27],
        n_accumulated_frames=unpacked[28],
        _reserved3=unpacked[29],
    )


def _read_chunk_header(f: io.BufferedReader, ep: str) -> ChunkHeader:
    """Read RDCh chunk header (22 bytes)."""
    fmt = f"{ep}4sHqII"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read chunk header.")
    
    (chunk_magic_b, chunk_header_size, chunk_timestamp_utc_ticks,
     chunk_number, chunk_payload_size) = struct.unpack(fmt, raw)
    
    chunk_magic = chunk_magic_b.decode("ascii")
    if chunk_magic != "CHUN":
        raise ValueError(f"Invalid chunk magic: expected 'CHUN', got '{chunk_magic}'")
    
    return ChunkHeader(
        chunk_magic=chunk_magic,
        chunk_header_size=chunk_header_size,
        chunk_timestamp_utc_ticks=chunk_timestamp_utc_ticks,
        chunk_number=chunk_number,
        chunk_payload_size=chunk_payload_size,
    )


def _map_data_type(data_type_code: int) -> np.dtype:
    """Map data type code to numpy dtype."""
    type_map = {
        0: np.complex64,
        1: np.complex128,
        2: np.float32,
        3: np.float64,
        4: np.int16,
        5: np.int32
    }
    return np.dtype(type_map.get(data_type_code, np.complex64))


def _map_window_type(window_code: int) -> str:
    """Map window type code to string."""
    window_map = {
        0: 'none',
        1: 'hann',
        2: 'hamming',
        3: 'blackman',
        4: 'kaiser'
    }
    return window_map.get(window_code, 'unknown')


def _print_header_summary(FH: FileHeader, BH: RDChBodyHeader) -> None:
    """Print summary of file and body headers."""
    print("--- Dopplium RDCh Data ---")
    print(f"Magic={FH.magic}  Version={FH.version}  "
          f"Endianness={'LE' if FH.endianness==1 else 'BE'}  MessageType={FH.message_type}")
    print(f"FileHdr={FH.file_header_size}  BodyHdr={BH.body_header_size}  "
          f"ChunkHdr={BH.chunk_header_size}  TotalChunksWritten={FH.total_frames_written}")
    print(f"NodeId=\"{FH.node_id}\"")
    
    print("\n-- RDCh Configuration --")
    dtype_str = ['complex64', 'complex128', 'float32', 'float64', 'int16', 'int32'][BH.data_type] \
                if BH.data_type < 6 else 'unknown'
    print(f"Dimensions: Range={BH.n_range_bins}, Doppler={BH.n_doppler_bins}, Channels={BH.n_channels}")
    print(f"Data type: {dtype_str}")
    
    print("\n-- Range/Velocity Axes --")
    print(f"Range: {BH.range_min_m:.2f} to {BH.range_max_m:.2f} m, "
          f"resolution={BH.range_resolution_m:.3f} m")
    print(f"Velocity: {BH.velocity_min_mps:.2f} to {BH.velocity_max_mps:.2f} m/s, "
          f"resolution={BH.velocity_resolution_mps:.3f} m/s")
    
    print("\n-- Radar Parameters --")
    print(f"StartFreq={BH.start_freq_ghz:.3f} GHz  BW={BH.bandwidth_ghz:.3f} GHz  "
          f"Fs={BH.sample_rate_ksps:.1f} ksps")
    print(f"FramePeriod={BH.frame_period_ms:.3f} ms")
    
    print("\n-- Processing Parameters --")
    print(f"FFT sizes: Range={BH.nfft_range}, Doppler={BH.nfft_doppler}")
    print(f"Original dimensions: Samples={BH.n_samples_original}, Chirps={BH.n_chirps_original}")
    print(f"Window types: Range={_map_window_type(BH.range_window_type)}, "
          f"Doppler={_map_window_type(BH.doppler_window_type)}")
    print(f"FFT shifts: Range={'Yes' if BH.fftshift_range else 'No'}, "
          f"Doppler={'Yes' if BH.fftshift_doppler else 'No'}")
    print(f"Half spectrum: {'Yes' if BH.range_half_spectrum else 'No'}")
    print(f"Accumulated frames: {BH.n_accumulated_frames}")


# ==============================
# Convenience function
# ==============================

def get_range_axis(headers: Dict[str, Any]) -> np.ndarray:
    """
    Extract range axis from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_rdch
    
    Returns:
    --------
    np.ndarray : Range axis in meters
    """
    BH = headers['body']
    return np.linspace(BH.range_min_m, BH.range_max_m, BH.n_range_bins)


def get_velocity_axis(headers: Dict[str, Any]) -> np.ndarray:
    """
    Extract velocity axis from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_rdch
    
    Returns:
    --------
    np.ndarray : Velocity axis in m/s
    """
    BH = headers['body']
    return np.linspace(BH.velocity_min_mps, BH.velocity_max_mps, BH.n_doppler_bins)


def get_processing_info(headers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract processing information from headers.
    
    Parameters:
    -----------
    headers : dict
        Headers dictionary returned by parse_dopplium_rdch
    
    Returns:
    --------
    dict : Processing parameters
    """
    BH = headers['body']
    
    return {
        'nfft_range': BH.nfft_range,
        'nfft_doppler': BH.nfft_doppler,
        'n_samples_original': BH.n_samples_original,
        'n_chirps_original': BH.n_chirps_original,
        'range_window': _map_window_type(BH.range_window_type),
        'doppler_window': _map_window_type(BH.doppler_window_type),
        'fftshift_range': bool(BH.fftshift_range),
        'fftshift_doppler': bool(BH.fftshift_doppler),
        'range_half_spectrum': bool(BH.range_half_spectrum),
        'n_accumulated_frames': BH.n_accumulated_frames,
    }

