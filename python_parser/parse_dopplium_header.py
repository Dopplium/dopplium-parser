"""
Common header parsing for Dopplium binary files.
Handles initial file header parsing, version detection, and message type detection.

Supported file versions: 2, 3
File header format is the same for both versions (80 bytes minimum).
"""

from __future__ import annotations
import io
import struct
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FileHeader:
    """Standard Dopplium file header (80 bytes)."""
    magic: str
    version: int
    endianness: int
    compression: int
    product_id: int
    message_type: int
    file_header_size: int
    body_header_size: int
    frame_header_size: int
    file_created_utc_ticks: int
    last_written_utc_ticks: int
    total_frames_written: int
    total_payload_bytes: int
    reserved1: int
    node_id: str


def detect_endianness(f: io.BufferedReader) -> Tuple[int, str]:
    """
    Detect file endianness from the header.
    
    Parameters:
    -----------
    f : io.BufferedReader
        File handle positioned at start of file
    
    Returns:
    --------
    tuple : (endianness_byte, endian_prefix)
        endianness_byte : int (0=big-endian, 1=little-endian)
        endian_prefix : str ('<' for little-endian, '>' for big-endian)
    """
    # Read magic first to verify file format
    magic = f.read(4).decode("ascii")
    if magic != "DOPP":
        raise ValueError("Invalid magic; not a Dopplium file.")
    
    # Seek to endianness byte at offset 6
    f.seek(6, io.SEEK_SET)
    endianness_byte = f.read(1)
    if len(endianness_byte) != 1:
        raise ValueError("Unable to read endianness byte.")
    
    endianness = endianness_byte[0]
    if endianness == 1:
        endian_prefix = "<"  # little-endian
    elif endianness == 0:
        endian_prefix = ">"  # big-endian
    else:
        # Default to little-endian if unknown
        endian_prefix = "<"
    
    return endianness, endian_prefix


def read_file_header(f: io.BufferedReader, endian_prefix: str) -> FileHeader:
    """
    Read standard Dopplium file header (80 bytes).
    
    Parameters:
    -----------
    f : io.BufferedReader
        File handle positioned at start of file header
    endian_prefix : str
        '<' for little-endian, '>' for big-endian
    
    Returns:
    --------
    FileHeader : Parsed file header
    """
    # Layout (80 bytes):
    # 4s magic, H version, B endianness, B compression, B product_id, B message_type,
    # H file_header_size, I body_header_size, I frame_header_size,
    # q file_created_ticks, q last_written_ticks,
    # I total_frames_written, I total_payload_bytes, I reserved1,
    # 32s node_id
    fmt = f"{endian_prefix}4sHBBBBHIIqqIII32s"
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("Failed to read file header.")
    
    (magic, version, endianness, compression, product_id, message_type,
     file_header_size, body_header_size, frame_header_size,
     file_created_ticks, last_written_ticks, total_frames_written,
     total_payload_bytes, reserved1, node_id_bytes) = struct.unpack(fmt, raw)
    
    magic = magic.decode("ascii")
    node_id = node_id_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
    
    return FileHeader(
        magic, version, endianness, compression, product_id, message_type,
        file_header_size, body_header_size, frame_header_size,
        file_created_ticks, last_written_ticks,
        total_frames_written, total_payload_bytes, reserved1, node_id
    )


def validate_version(version: int) -> None:
    """
    Validate that the file version is supported.
    
    Parameters:
    -----------
    version : int
        File version from header
    
    Raises:
    -------
    ValueError
        If version is not supported
    """
    if version not in (2, 3):
        raise ValueError(f"Unsupported file version: {version}. Supported versions: 2, 3")


def parse_file_header(filename: str) -> Tuple[FileHeader, str]:
    """
    Parse the file header from a Dopplium binary file.
    
    Parameters:
    -----------
    filename : str
        Path to the Dopplium binary file
    
    Returns:
    --------
    tuple : (file_header, endian_prefix)
        file_header : FileHeader
            Parsed file header (versions 2 and 3 use same format)
        endian_prefix : str
            '<' for little-endian, '>' for big-endian
    
    Raises:
    -------
    ValueError
        If file format is invalid or version is unsupported
    """
    with open(filename, "rb") as f:
        # Detect endianness
        endianness, endian_prefix = detect_endianness(f)
        
        # Rewind to beginning and read full file header
        f.seek(0, io.SEEK_SET)
        file_header = read_file_header(f, endian_prefix)
        
        # Validate version
        validate_version(file_header.version)
        
        return file_header, endian_prefix

