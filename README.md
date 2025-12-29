# Dopplium Parser

Parsers for Dopplium radar data formats (MATLAB & Python).

## Installation

**MATLAB**: Add `matlab_parser/` to your path.

**Python**: 
```bash
pip install -e .
```

## Usage

### Automatic Format Detection (Recommended)

**Python**:
```python
from python_parser import parse_dopplium

# Automatically detects format (RawData or RDCh) and parses accordingly
data, headers = parse_dopplium('file.bin', verbose=True)
```

The dispatcher automatically detects the message type and routes to the appropriate parser:
- **message_type = 1**: ADCData (raw radar data) calls `parse_dopplium_raw`
- **message_type = 2**: RDCMaps (Range-Doppler-Channel) calls `parse_dopplium_rdch`
- **message_type = 3**: RadarCube (Range-Doppler-Azimuth-Elevation) calls `parse_dopplium_radarcube`

**Full Message Type List (Parser Version 3):**
- 0: Unknown (unsupported)
- 1: ADCData
- 2: RDCMaps
- 3: RadarCube
- 4: Detections (not yet implemented)
- 5: Blobs (not yet implemented)
- 6: Tracks (not yet implemented)

### Direct Parser Calls

For direct access to specific parsers:

**Python - RawData**:
```python
from python_parser import parse_dopplium_raw
data, headers = parse_dopplium_raw('raw_file.bin')
```

**Python - RDCh**:
```python
from python_parser import parse_dopplium_rdch
data, headers = parse_dopplium_rdch('rdch_file.bin')
```

**Python - RadarCube**:
```python
from python_parser import parse_dopplium_radarcube, get_azimuth_axis, get_elevation_axis
data, headers = parse_dopplium_radarcube('radarcube_file.bin')

# Get angular axes
azimuth_axis = get_azimuth_axis(headers)
elevation_axis = get_elevation_axis(headers)
```

**MATLAB**:
```matlab
[data, headers] = parseDoppliumRaw('file.bin');
```

## Data Formats

### RawData Format
Returns data shaped `[samples, chirpsPerTx, channels, frames]`:
- **Samples**: ADC samples per chirp
- **ChirpsPerTx**: Chirps per transmitter per frame  
- **Channels**: Receiver index (single TX) or TX-RX combinations (multi TX)
- **Frames**: Number of radar frames

### RDCh Format
Returns data shaped `[range_bins, doppler_bins, channels, cpis]`:
- **Range bins**: Processed range dimension
- **Doppler bins**: Processed velocity/Doppler dimension
- **Channels**: Number of receiver channels
- **CPIs**: Number of Coherent Processing Intervals

### RadarCube Format
Returns data shaped `[range_bins, doppler_bins, azimuth_bins, elevation_bins, cpis]`:
- **Range bins**: Processed range dimension
- **Doppler bins**: Processed velocity/Doppler dimension
- **Azimuth bins**: Angular resolution in azimuth (horizontal)
- **Elevation bins**: Angular resolution in elevation (vertical)
- **CPIs**: Number of Coherent Processing Intervals
- **Algorithms**: Supports FFT, CAPON, MUSIC for angle estimation

## Options

**Common options** (`parse_dopplium`, `parse_dopplium_raw`, `parse_dopplium_rdch`):
- `max_cpis_or_frames` / `max_frames` / `max_cpis`: Limit CPIs/frames to read
- `verbose`: Show parsing info

**RawData-specific options**:
- `cast`: Output type ('float32', 'float64', 'int16')
- `return_complex`: Return complex numbers for IQ data

## Examples

See `matlab_parser/exampleParse.m` and `python_parser/example_parse.py` for complete examples with 2D FFT processing and visualization.