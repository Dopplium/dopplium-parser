# Dopplium Parser

Parsers for Dopplium radar data formats in Python and MATLAB.

## Installation

Python (editable install from this folder):

```bash
pip install -e .
```

MATLAB:
- Add `matlab_parser/` to your MATLAB path.

## Supported Message Types

### File versions 3, 4, 5 and 6

- `1`: ADCData
- `2`: RDCMaps (RDCh)
- `3`: RadarCube
- `4`: Detections
- `5`: Blobs
- `6`: Tracks

### Parser version 2

- `3`: RawData/ADC supported
- `1` (Detections), `2` (Tracks), `4` (Aggregated) are not implemented in the Python dispatcher

## DOP-BIN 2.0 ADC files (file version 6)

ADC files with file version 6, `config_version` 4 and `multiplexing_mode` 2
can change their transmit pattern from frame to frame (for example TDMA, TX1 only,
TDMA, TX2 only). Each frame's 32-byte payload header carries a `tx_map`, the
frame's repeating transmitter pattern, and `n_chirps_per_frame` counts every chirp.

- Data comes back shaped `[samples, chirps, receivers, frames]`, with the chirps
  in transmission order. `n_transmitters` does not change the decoding.
- Each frame's pattern is in its frame header: `headers["frame"][i].tx_map`
  (Python tuple, e.g. `(1, 2)`, `(2,)`, or `()` for no transmitter information) or
  `headers.frame(i).tx_map` (MATLAB row vector). Chirp `c` (0-based) was sent by
  `tx_map[c % L]`, with `L = len(tx_map)`.
- Process a frame per pattern slot, not by collecting all chirps of one
  transmitter: `split_pattern_slots(frame, tx_map)` (Python) and
  `splitPatternSlots(frame, txMap)` (MATLAB) return
  `[samples, chirps / L, L, receivers]`, where slot `k` holds chirps `k, k + L, ...`.
- A frame whose `tx_map` has a non-zero entry after the terminating 0 (for example
  `[1, 0, 2, 0]`), or whose pattern length does not divide the chirp count, is
  rejected with an error. Read around it with `start_frame` / `max_frames`.
- Older files (versions 2-5, 24-byte payload headers) decode exactly as before.

## Python Usage

### Automatic format detection (recommended)

```python
from python_parser import parse_dopplium

data, headers = parse_dopplium("file.bin", verbose=True)
print(headers["file"].message_type)
```

For ADC, RDCh, and RadarCube files, pass `start_frame` to begin at a
zero-based frame/CPI offset. With a limit, `parse_dopplium(...,
start_frame=100, max_cpis_or_frames=50)` returns the half-open window
`[100, 150)`. `start_frame` is not supported for Detections, Blobs, or Tracks.

### Direct parser calls

```python
from python_parser import (
    parse_dopplium_raw,
    parse_dopplium_rdch,
    parse_dopplium_radarcube,
    parse_dopplium_detections,
    parse_dopplium_blobs,
    parse_dopplium_tracks,
)

raw, raw_headers = parse_dopplium_raw("raw.bin", start_frame=100, max_frames=50)
rdch, rdch_headers = parse_dopplium_rdch("rdch.bin", start_frame=100, max_cpis=50)
cube, cube_headers = parse_dopplium_radarcube("radarcube.bin", start_frame=100, max_cpis=50)
detections, det_headers = parse_dopplium_detections("detections.bin")
blobs, blob_headers = parse_dopplium_blobs("blobs.bin")
tracks, track_headers = parse_dopplium_tracks("tracks.bin")
```

## MATLAB Usage

Use the MATLAB dispatcher:

```matlab
[data, headers] = doppliumParser('file.bin');
```

For ADC, RDCMaps, and RadarCube files, MATLAB also accepts a zero-based
`startFrame` option:

```matlab
opts.startFrame = 100;
opts.maxFrames = 50;
[data, headers] = doppliumParser('file.bin', opts);
```

Or call format-specific functions in `matlab_parser/`:
- `parseADCData`
- `parseRDCMaps`
- `parseRadarCube`
- `parseDetections`
- `parseBlobs`
- `parseTracks`

## Data Shapes (Python)

- ADCData/RawData: `[samples, chirpsPerTx, channels, frames]`; `multiplexing_mode` 2:
  `[samples, chirps, receivers, frames]`
- RDCh: `[range_bins, doppler_bins, channels, cpis]`
- RadarCube: `[range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]`
- Detections/Blobs/Tracks: numpy structured arrays

## Examples

- Python: `python_parser/example_parse.py`, `python_parser/example_parse_detections.py`, `python_parser/example_parse_tracks.py`
- MATLAB: `matlab_parser/exampleADCDataParse.m`, `matlab_parser/exampleRDCMapsDataParse.m`

For full Python parser details, see `python_parser/README.md`.

## Tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

```bash
matlab -batch "results = runtests('tests/matlab'); assertSuccess(results)"
```

The tests build synthetic ADC files with known samples, including DOP-BIN 2.0
files with mixed `tx_map` patterns and an invalid one.
