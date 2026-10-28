"""
Synthetic ADCData files packed with struct, for parser tests.

Layouts follow DOP-BIN 1/0 (file versions 2-5, 24-byte payload headers) and
DOP-BIN 2.0 (file version 6, config_version 4, multiplexing_mode 2, 32-byte
payload headers carrying tx_map).
"""

import struct

import numpy as np


def sample_code(frame, chirp, rx, sample):
    """Distinct int16 value for every (frame, chirp, receiver, sample) used in the tests."""
    return frame * 1024 + chirp * 32 + rx * 8 + sample


def frame_samples(frame, n_chirps, n_rx, n_samples):
    """Frame samples shaped (chirp, rx, sample), chirps in transmission order."""
    c, r, s = np.meshgrid(np.arange(n_chirps), np.arange(n_rx), np.arange(n_samples), indexing="ij")
    return sample_code(frame, c, r, s).astype(np.int16)


def file_header(*, version, message_type, payload_header_size, n_frames, total_payload_bytes,
                file_header_size=None, body_header_size=192, ep="<"):
    if file_header_size is None:
        file_header_size = 128 if version >= 4 else 80
    endianness = 1 if ep == "<" else 0
    node_id = b"test-node".ljust(32, b"\x00")
    common = (b"DOPP", version, endianness, 0, 7, message_type, file_header_size,
              body_header_size, payload_header_size, 638000000000000000, 638000000000000000, n_frames)
    if version >= 5:
        raw = struct.pack(f"{ep}4sHBBBBHIIqqIQ32s", *common, total_payload_bytes, node_id)
    else:
        raw = struct.pack(f"{ep}4sHBBBBHIIqqIII32s", *common, total_payload_bytes, 0, node_id)
    if file_header_size >= 128:
        raw += struct.pack(f"{ep}iii3h30s", 415000000, -37000000, 1200, 1, 2, 3, bytes(30))
    return raw.ljust(file_header_size, b"\x00")


def adc_body_header(*, config_version, multiplexing_mode, frame_header_size, n_samples,
                    n_chirps_per_frame, n_rx, n_tx, bytes_per_frame, sample_type=0, data_order=0,
                    iq_order=0, channel_order=None, ep="<"):
    bytes_per_sample = 2 if sample_type == 0 else 4
    if channel_order is None:
        channel_order = list(range(1, n_rx + 1))
    channel_order = list(channel_order) + [0] * (30 - len(channel_order))
    raw = struct.pack(
        f"{ep}4sHHHHIIIHHHBBBBBBdddddddddIIIIffff30b22s",
        b"ADCD", config_version, 192, frame_header_size, 0,
        frame_header_size + bytes_per_frame,
        n_samples, n_chirps_per_frame, 16, n_rx, n_tx,
        sample_type, data_order, iq_order, 0, multiplexing_mode, 0,
        77.0, 1.0, 5.0, 0.0, 6.0, 60.0, 10000.0, 20.0, 9.92,
        2, bytes_per_sample, bytes_per_frame // bytes_per_sample, bytes_per_frame,
        100.0, 10.0, 0.15, 0.1,
        *channel_order, bytes(22),
    )
    assert len(raw) == 192
    return raw


def frame_header(*, frame_number, payload_size, frame_header_size, tx_map=None, ep="<"):
    raw = struct.pack(f"{ep}4sHHqII", b"FRME", 0, frame_header_size,
                      638000000000000000 + frame_number * 99200, frame_number, payload_size)
    if tx_map is not None:
        raw += bytes(tx_map) + bytes(4)
    return raw.ljust(frame_header_size, b"\x00")


def encode_payload(samples, *, data_order=0, sample_type=0, ep="<"):
    """
    Encode (chirp, rx, sample) data as stored bytes.

    data_order 0 (ByChannel): chirp -> rx -> int16 element. data_order 1: chirp ->
    int16 element -> rx. Complex samples (sample_type 1) are written as interleaved
    I, Q elements (iq_order 0) with Q = -I, so the decoded value is I - jI.
    """
    n_chirps, n_rx, n_samples = samples.shape
    if sample_type == 1:
        samples = np.stack([samples, -samples], axis=-1).reshape(n_chirps, n_rx, 2 * n_samples)
    if data_order == 1:
        samples = np.transpose(samples, (0, 2, 1))
    return np.ascontiguousarray(samples).astype(f"{ep}i2").tobytes()


def write_adc_file(path, *, frames, n_chirps_on_wire, n_rx=4, n_samples=8, version=6,
                   config_version=4, multiplexing_mode=2, n_chirps_field=None, n_tx=2,
                   payload_header_size=None, sample_type=0, data_order=0, message_type=None,
                   ep="<"):
    """
    Write an ADCData file. frames is a list of tx_map entries (4 ints) for mode-2
    style 32-byte headers, or None per frame for classic 24-byte headers.
    Frame f holds frame_samples(f, ...). Returns the expected samples per frame.
    """
    if payload_header_size is None:
        payload_header_size = 32 if multiplexing_mode == 2 else 24
    if n_chirps_field is None:
        n_chirps_field = n_chirps_on_wire
    if message_type is None:
        message_type = 3 if version == 2 else 1
    bytes_per_frame = n_chirps_on_wire * n_rx * n_samples * (2 if sample_type == 0 else 4)

    blob = file_header(version=version, message_type=message_type,
                       payload_header_size=payload_header_size, n_frames=len(frames),
                       total_payload_bytes=len(frames) * bytes_per_frame, ep=ep)
    blob += adc_body_header(config_version=config_version, multiplexing_mode=multiplexing_mode,
                            frame_header_size=payload_header_size, n_samples=n_samples,
                            n_chirps_per_frame=n_chirps_field, n_rx=n_rx, n_tx=n_tx,
                            bytes_per_frame=bytes_per_frame, sample_type=sample_type,
                            data_order=data_order, ep=ep)
    expected = []
    for index, tx_map in enumerate(frames):
        samples = frame_samples(index, n_chirps_on_wire, n_rx, n_samples)
        expected.append(samples)
        blob += frame_header(frame_number=100 + index, payload_size=bytes_per_frame,
                             frame_header_size=payload_header_size, tx_map=tx_map, ep=ep)
        blob += encode_payload(samples, data_order=data_order, sample_type=sample_type, ep=ep)
    with open(path, "wb") as fh:
        fh.write(blob)
    return expected
