# Python Parsers for Dopplium Binary Formats

Binary data parsers for Dopplium radar data formats, implemented in Python using `struct` and `numpy`.

## Overview

This package provides four binary parsers for different radar data formats:

| Parser | Format | Message Type | Data Shape | Use Case |
|--------|--------|--------------|------------|----------|
| **parse_dopplium_raw** | RawData/ADCData | 3 (v2) / 1 (v3) | [samples, chirpsPerTx, channels, frames] | Raw ADC data from radar |
| **parse_dopplium_rdch** | RDCMaps (RDCh) | 2 | [range, doppler, channels, cpis] | Range-Doppler-Channel processed data |
| **parse_dopplium_radarcube** | RadarCube | 3 | [range, doppler, azimuth, elevation, cpis] | Full 4D radar cubes with angles |
| **parse_dopplium_detections** | Detections | 4 | Structured array of detections | Processed target detections |

All parsers support automatic format detection via the main `parse_dopplium()` dispatcher.

## Installation

```bash
# From the repo root
pip install -e .

# Or install dependencies only
pip install numpy matplotlib
```

## Quick Start

### Automatic Format Detection (Recommended)

The easiest way to parse any Dopplium file:

```python
from python_parser import parse_dopplium

# Automatically detects format and routes to appropriate parser
data, headers = parse_dopplium("file.bin", verbose=True)

# Check what format was parsed
print(f"Message type: {headers['file'].message_type}")
print(f"Data shape: {data.shape}")
```

### Direct Parser Usage

For when you know the format ahead of time:

```python
from python_parser import (
    parse_dopplium_raw,
    parse_dopplium_rdch,
    parse_dopplium_radarcube,
    parse_dopplium_detections
)

# Parse specific formats
raw_data, headers = parse_dopplium_raw("raw.bin")
rdch_data, headers = parse_dopplium_rdch("rdch.bin")
cube_data, headers = parse_dopplium_radarcube("radarcube.bin")
detections, headers = parse_dopplium_detections("detections.bin")
```

## Parser Details

### 1. parse_dopplium_raw - Raw ADC Data

For unprocessed radar ADC samples.

```python
from python_parser import parse_dopplium_raw

# Parse raw ADC data
data, headers = parse_dopplium_raw(
    "raw_data.bin",
    max_frames=100,          # Limit number of frames (None = all)
    cast='float32',          # Output data type
    return_complex=True,     # Return complex data (I+jQ)
    verbose=True
)

# Data shape: [samples, chirpsPerTx, channels, frames]
print(f"Shape: {data.shape}")
print(f"Samples per chirp: {data.shape[0]}")
print(f"Chirps per TX: {data.shape[1]}")
print(f"Channels: {data.shape[2]}")
print(f"Frames: {data.shape[3]}")

# Access headers
body_header = headers['body']
print(f"Sample rate: {body_header.sample_rate_ksps} kHz")
print(f"Chirp bandwidth: {body_header.bandwidth_ghz} GHz")
```

**Key Features:**
- Raw I+Q samples from radar ADC
- Configurable output type (float32, float64, int16)
- Complex or separate I/Q channels
- Frame-based organization

**Common Use Case:** Input to custom processing pipelines

### 2. parse_dopplium_rdch - Range-Doppler-Channel

For processed 3D radar data cubes.

```python
from python_parser import (
    parse_dopplium_rdch,
    get_range_axis,
    get_velocity_axis,
    get_processing_info
)

# Parse RDCh file
data, headers = parse_dopplium_rdch(
    "rdch.bin",
    max_cpis=50,    # Limit CPIs (None = all)
    verbose=True
)

# Data shape: [range_bins, doppler_bins, channels, cpis]
print(f"Shape: {data.shape}")

# Get axis information
range_axis = get_range_axis(headers)    # Range in meters
velocity_axis = get_velocity_axis(headers)  # Velocity in m/s

# Get processing parameters
proc_info = get_processing_info(headers)
print(f"FFT sizes: range={proc_info['nfft_range']}, doppler={proc_info['nfft_doppler']}")
print(f"Windows: range={proc_info['range_window']}, doppler={proc_info['doppler_window']}")
print(f"Data format: {proc_info['data_format']}")  # complex, amplitude, or power
print(f"Scale: {'dB' if proc_info['is_db_scale'] else 'linear'}")

# Process single CPI
cpi_data = data[:, :, :, 0]  # First CPI
print(f"CPI shape: {cpi_data.shape}")
```

**Key Features:**
- 3D data cubes (range × doppler × channels)
- Processing metadata (FFT sizes, windows, shifts)
- Physical and FFT resolutions
- Support for complex, amplitude, or power data
- Linear or dB scale

**Common Use Case:** Input to detection algorithms, visualization

### 3. parse_dopplium_radarcube - Full 4D Radar Cubes

For radar data with angle estimation (azimuth and elevation).

```python
from python_parser import (
    parse_dopplium_radarcube,
    get_range_axis,
    get_velocity_axis,
    get_azimuth_axis,
    get_elevation_axis,
    has_known_angles,
    uses_fft_for_angles
)

# Parse RadarCube file
data, headers = parse_dopplium_radarcube(
    "radarcube.bin",
    max_cpis=50,
    verbose=True
)

# Data shape: [range, doppler, azimuth, elevation, cpis]
print(f"Shape: {data.shape}")

# Get all axes
range_axis = get_range_axis(headers)          # meters
velocity_axis = get_velocity_axis(headers)    # m/s
azimuth_axis = get_azimuth_axis(headers)      # degrees
elevation_axis = get_elevation_axis(headers)  # degrees

# Check angle estimation method
if uses_fft_for_angles(headers):
    print("Using FFT-based angle estimation")
else:
    algo = headers['body'].angle_estimation_algorithm
    algo_map = {1: 'CAPON', 2: 'MUSIC', 3: 'other'}
    print(f"Using {algo_map.get(algo, 'unknown')} angle estimation")

# Access single CPI
cpi_data = data[:, :, :, :, 0]  # First CPI: [range, doppler, az, el]
```

**Key Features:**
- 4D data cubes (range × doppler × azimuth × elevation)
- Multiple angle estimation algorithms (FFT, CAPON, MUSIC)
- Full angular information
- All RDCh features plus angular processing

**Common Use Case:** Advanced detection with angle information, 3D visualization

### 4. parse_dopplium_detections - Target Detections

For processed detection lists with full target parameters.

```python
from python_parser import (
    parse_dopplium_detections,
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude,
    get_detection_statistics
)

# Parse detections file
detections, headers = parse_dopplium_detections(
    "detections.bin",
    max_batches=None,  # Read all batches
    verbose=True
)

# Detections is a numpy structured array
print(f"Total detections: {len(detections)}")
print(f"Fields: {list(detections.dtype.names)}")

# Access detection fields
ranges = detections['range']              # meters
velocities = detections['velocity']       # m/s
azimuths = detections['azimuth']         # degrees
elevations = detections['elevation']     # degrees
amplitudes = detections['amplitude']     # detection strength
monopulse_az = detections['monopulse_ratio_az']  # angle quality
monopulse_el = detections['monopulse_ratio_el']

# Cell indices (link back to radar cube)
range_cells = detections['range_cell']
doppler_cells = detections['doppler_cell']
azimuth_cells = detections['azimuth_cell']
elevation_cells = detections['elevation_cell']

# Filter detections
close_targets = filter_detections_by_range(detections, 0.0, 50.0)
moving_targets = filter_detections_by_velocity(detections, 1.0, 100.0)
strong_targets = filter_detections_by_amplitude(detections, 20.0)

# Combined filtering
close_strong_moving = filter_detections_by_amplitude(
    filter_detections_by_velocity(
        filter_detections_by_range(detections, 10.0, 50.0),
        2.0, 50.0
    ),
    25.0
)

# Get statistics
stats = get_detection_statistics(detections)
print(f"Range: {stats['range']['min']:.2f} to {stats['range']['max']:.2f} m")
print(f"Velocity: mean={stats['velocity']['mean']:.2f} m/s")
print(f"Total batches: {stats['batches']['unique_batches']}")

# Custom filtering with numpy
mask = (
    (detections['range'] > 10) & 
    (detections['range'] < 50) &
    (np.abs(detections['velocity']) > 1.0)
)
filtered = detections[mask]

# Group by batch
for batch_idx in np.unique(detections['batch_index']):
    batch_dets = detections[detections['batch_index'] == batch_idx]
    print(f"Batch {batch_idx}: {len(batch_dets)} detections")
```

**Detection Record Fields:**
- `range` (float64): Range in meters
- `velocity` (float64): Radial velocity in m/s
- `azimuth` (float64): Azimuth angle in degrees
- `elevation` (float64): Elevation angle in degrees
- `amplitude` (float64): Detection amplitude
- `monopulse_ratio_az` (float32): Azimuth monopulse ratio (angle quality)
- `monopulse_ratio_el` (float32): Elevation monopulse ratio
- `range_cell` (int16): Range bin index in processing grid
- `doppler_cell` (int16): Doppler bin index
- `azimuth_cell` (int16): Azimuth bin index
- `elevation_cell` (int16): Elevation bin index
- `batch_index` (uint32): Which batch this detection belongs to
- `sequence_number` (uint32): Sequence number of the batch

**Key Features:**
- Structured numpy arrays for efficient access
- Built-in filtering functions
- Statistics calculation
- Batch organization (detections grouped by time)
- Cell indices link back to radar cube
- Empty batches supported

**Common Use Case:** Target tracking, visualization, data export

## Example Scripts

Each parser has example code:

```bash
python example_parse.py               # Raw/RDCh/RadarCube examples
python example_parse_detections.py    # Detections with visualization
```

## Common Features

All parsers share these capabilities:

| Feature | Description |
|---------|-------------|
| **Automatic Dispatch** | Use `parse_dopplium()` to auto-detect and parse any format |
| **Header Access** | Full access to file, body, and CPI/batch headers |
| **Verbose Mode** | Optional detailed output for debugging |
| **Partial Reading** | Limit number of CPIs/frames/batches read |
| **Endianness Support** | Handles both little-endian and big-endian files |
| **Version Support** | Supports both Version 2 and Version 3 formats |
| **Error Handling** | Comprehensive validation and error messages |

## Accessing Headers

All parsers return headers in a consistent dictionary structure:

```python
data, headers = parse_dopplium_*("file.bin")

# File header (common to all formats)
file_header = headers['file']
print(f"Version: {file_header.version}")
print(f"Message type: {file_header.message_type}")
print(f"Node ID: {file_header.node_id}")
print(f"Total CPIs/frames written: {file_header.total_frames_written}")
print(f"Endianness: {'little' if file_header.endianness == 1 else 'big'}")

# Body header (format-specific)
body_header = headers['body']
# Fields vary by format

# CPI/Frame/Batch headers (list of per-CPI metadata)
for i, cpi_header in enumerate(headers['cpi'][:5]):  # or 'frame', 'batch'
    print(f"CPI {i}: timestamp={cpi_header.cpi_timestamp_utc_ticks}")
```

## Message Type Reference

| Message Type | Version 2 | Version 3 | Parser Function |
|--------------|-----------|-----------|-----------------|
| 0 | Unknown | Unknown | Not supported |
| 1 | Detections | ADCData | `parse_dopplium_raw` |
| 2 | Tracks | RDCMaps (RDCh) | `parse_dopplium_rdch` |
| 3 | RawData/ADC | RadarCube | `parse_dopplium_raw` / `parse_dopplium_radarcube` |
| 4 | Aggregated | Detections | `parse_dopplium_detections` |
| 5 | - | Blobs | Not yet implemented |
| 6 | - | Tracks | Not yet implemented |

## Helper Functions

### For RDCh and RadarCube

```python
from python_parser import (
    get_range_axis,
    get_velocity_axis,
    get_azimuth_axis,        # RadarCube only
    get_elevation_axis,      # RadarCube only
    get_processing_info,
    has_known_angles,        # RadarCube only
    uses_fft_for_angles      # RadarCube only
)

# Extract axis vectors
range_m = get_range_axis(headers)
velocity_mps = get_velocity_axis(headers)

# RadarCube specific
azimuth_deg = get_azimuth_axis(headers)
elevation_deg = get_elevation_axis(headers)

# Check angle estimation
if has_known_angles(headers):
    if uses_fft_for_angles(headers):
        print("Using FFT-based angles")
    else:
        print("Using advanced angle estimation (CAPON/MUSIC)")

# Get all processing parameters
proc_info = get_processing_info(headers)
```

### For Detections

```python
from python_parser import (
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude,
    get_detection_statistics
)

# Filter detections
filtered = filter_detections_by_range(detections, min_range, max_range)
filtered = filter_detections_by_velocity(detections, min_vel, max_vel)
filtered = filter_detections_by_amplitude(detections, min_amplitude)

# Get statistics
stats = get_detection_statistics(detections)
```

## Advanced Usage

### Working with Complex Data

```python
# RDCh/RadarCube often contain complex data
data, headers = parse_dopplium_rdch("file.bin")

# Convert to magnitude
magnitude = np.abs(data)

# Convert to dB
magnitude_db = 20 * np.log10(np.abs(data) + 1e-12)

# Extract phase
phase = np.angle(data)

# Separate real and imaginary
real_part = np.real(data)
imag_part = np.imag(data)
```

### Processing Single CPI

```python
# RDCh data
data, headers = parse_dopplium_rdch("file.bin")
# Shape: [range, doppler, channels, cpis]

# Process first CPI
cpi_0 = data[:, :, :, 0]  # [range, doppler, channels]

# Average across channels
cpi_avg = np.mean(cpi_0, axis=2)  # [range, doppler]

# Find peak
peak_idx = np.unravel_index(np.argmax(np.abs(cpi_avg)), cpi_avg.shape)
range_bin, doppler_bin = peak_idx

# Convert to physical units
range_axis = get_range_axis(headers)
velocity_axis = get_velocity_axis(headers)
peak_range = range_axis[range_bin]
peak_velocity = velocity_axis[doppler_bin]
```

### Memory-Efficient Partial Reading

```python
# Read only first 10 CPIs (useful for large files)
data, headers = parse_dopplium_rdch(
    "large_file.bin",
    max_cpis=10,
    verbose=True
)

# Read specific number of frames
raw_data, headers = parse_dopplium_raw(
    "raw_data.bin",
    max_frames=50
)

# Read limited detections
detections, headers = parse_dopplium_detections(
    "detections.bin",
    max_batches=100
)
```

## Data Types

All parsers handle these numpy data types:

| Code | NumPy Type | Bytes | Common Use |
|------|------------|-------|------------|
| 0 | `np.complex64` | 8 | Complex radar data (default) |
| 1 | `np.complex128` | 16 | High-precision complex |
| 2 | `np.float32` | 4 | Real-valued (magnitude/power) |
| 3 | `np.float64` | 8 | High-precision real |
| 4 | `np.int16` | 2 | Integer data (rare) |
| 5 | `np.int32` | 4 | Integer data (rare) |

## Notes

- All files use **little-endian** byte order by default
- Data arrays stored in **Fortran-order** (column-major) with range varying fastest
- Timestamps stored as .NET ticks (100ns intervals since 0001-01-01)
- String fields (node_id) are ASCII with null-termination
- All parsers validate magic numbers and header consistency

## Best Practices

1. **Use automatic dispatch for flexibility**:
   ```python
   data, headers = parse_dopplium("unknown_format.bin")
   ```

2. **Check message type after parsing**:
   ```python
   msg_type = headers['file'].message_type
   if msg_type == 2:  # RDCh
       range_axis = get_range_axis(headers)
   ```

3. **Limit data for testing**:
   ```python
   # Test with small amount of data first
   data, headers = parse_dopplium("file.bin", max_cpis_or_frames=1)
   ```

4. **Use verbose mode for debugging**:
   ```python
   data, headers = parse_dopplium("file.bin", verbose=True)
   ```

5. **Access metadata through helper functions**:
   ```python
   # Preferred
   range_axis = get_range_axis(headers)
   
   # Instead of manual calculation
   # range_axis = np.linspace(...)
   ```

## See Also

- **Writers**: `../dopplium_writer/python_writer/` - Write these binary formats
- **Main Dispatcher**: `parse_dopplium.py` - Automatic format detection
- **Format Specs**: See repository documentation for detailed binary format specifications

