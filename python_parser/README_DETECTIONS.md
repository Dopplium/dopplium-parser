# Dopplium Detections Parser

Python parser for Dopplium Detections binary format (Version 3, message_type 4).

## Overview

The Detections format stores processed radar detections with complete target information including:
- Range, velocity, azimuth, elevation
- Detection amplitude
- Monopulse ratios (for angle estimation quality)
- Grid cell indices (range, doppler, azimuth, elevation bins)
- Batch/sequence information

## File Format

### Structure
```
[File Header (80 bytes)]
[Body Header (64 bytes)]
[Batch Header (30 bytes)]
[Detection Records (56 bytes each)]
[Batch Header (30 bytes)]
[Detection Records (56 bytes each)]
...
```

### Headers
- **File Header (80 bytes)**: Standard Dopplium format (version=3, message_type=4)
- **Body Header (64 bytes)**: Algorithm ID, version, record sizes
- **Batch Header (30 bytes)**: Per-batch metadata (timestamp, detection count, sequence number)

### Detection Record (56 bytes)
Each detection contains:
- `range` (float64): Range in meters
- `velocity` (float64): Radial velocity in m/s
- `azimuth` (float64): Azimuth angle in degrees
- `elevation` (float64): Elevation angle in degrees
- `amplitude` (float64): Detection amplitude (linear or dB)
- `monopulse_ratio_az` (float32): Azimuth monopulse ratio
- `monopulse_ratio_el` (float32): Elevation monopulse ratio
- `range_cell` (int16): Range bin index
- `doppler_cell` (int16): Doppler bin index
- `azimuth_cell` (int16): Azimuth bin index
- `elevation_cell` (int16): Elevation bin index

## Installation

```bash
cd dopplium_parser
pip install -e .
```

Or install dependencies directly:
```bash
pip install numpy matplotlib
```

## Usage

### Basic Parsing

```python
from python_parser import parse_dopplium_detections

# Parse detections file
detections, headers = parse_dopplium_detections(
    "path/to/detections.bin",
    max_batches=None,  # Read all batches
    verbose=True
)

# Access detection data
print(f"Total detections: {len(detections)}")
print(f"Range: {detections['range']}")
print(f"Velocity: {detections['velocity']}")
print(f"Azimuth: {detections['azimuth']}")
```

### Filtering Detections

```python
from python_parser import (
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude
)

# Filter by range (10m to 100m)
close_detections = filter_detections_by_range(detections, 10.0, 100.0)

# Filter by velocity (moving targets: |v| > 1 m/s)
moving_detections = filter_detections_by_velocity(detections, 1.0, 100.0)

# Filter by amplitude threshold
strong_detections = filter_detections_by_amplitude(detections, 20.0)

# Combine filters
close_strong = filter_detections_by_amplitude(
    filter_detections_by_range(detections, 0.0, 50.0),
    15.0
)
```

### Getting Statistics

```python
from python_parser import get_detection_statistics

stats = get_detection_statistics(detections)
print(f"Total detections: {stats['count']}")
print(f"Range: min={stats['range']['min']:.2f}, max={stats['range']['max']:.2f}")
print(f"Velocity: mean={stats['velocity']['mean']:.2f}")
print(f"Unique batches: {stats['batches']['unique_batches']}")
```

### Automatic Format Detection

The main dispatcher automatically routes to the detections parser:

```python
from python_parser import parse_dopplium

# Automatically detects and parses detections format
detections, headers = parse_dopplium("path/to/detections.bin", verbose=True)
```

### Accessing Headers

```python
# File header
file_header = headers['file']
print(f"Version: {file_header.version}")
print(f"Node ID: {file_header.node_id}")
print(f"Total batches: {file_header.total_frames_written}")

# Body header
body_header = headers['body']
print(f"Algorithm ID: {body_header.algorithm_id}")
print(f"Algorithm version: {body_header.algorithm_version}")

# Batch headers
for i, batch in enumerate(headers['batch'][:5]):
    print(f"Batch {i}: seq={batch.sequence_number}, "
          f"detections={batch.detection_count}")
```

## Example Script

See `example_parse_detections.py` for a complete example with visualization.

```bash
python example_parse_detections.py
```

## Data Structure

The parser returns a numpy structured array with the following dtype:

```python
dtype = np.dtype([
    ('range', np.float64),
    ('velocity', np.float64),
    ('azimuth', np.float64),
    ('elevation', np.float64),
    ('amplitude', np.float64),
    ('monopulse_ratio_az', np.float32),
    ('monopulse_ratio_el', np.float32),
    ('range_cell', np.int16),
    ('doppler_cell', np.int16),
    ('azimuth_cell', np.int16),
    ('elevation_cell', np.int16),
    ('batch_index', np.uint32),
    ('sequence_number', np.uint32),
])
```

## Advanced Usage

### Custom Filtering

```python
# Custom filtering using numpy boolean indexing
mask = (
    (detections['range'] > 10) & 
    (detections['range'] < 50) &
    (np.abs(detections['velocity']) > 1.0) &
    (detections['amplitude'] > 20.0)
)
filtered = detections[mask]
```

### Sorting Detections

```python
# Sort by range
sorted_detections = np.sort(detections, order='range')

# Sort by amplitude (descending)
sorted_by_amp = detections[np.argsort(-detections['amplitude'])]
```

### Grouping by Batch

```python
# Get detections from specific batch
batch_idx = 5
batch_detections = detections[detections['batch_index'] == batch_idx]

# Group by sequence number
for seq_num in np.unique(detections['sequence_number']):
    seq_detections = detections[detections['sequence_number'] == seq_num]
    print(f"Sequence {seq_num}: {len(seq_detections)} detections")
```

## API Reference

### Functions

#### `parse_dopplium_detections(filename, *, max_batches=None, verbose=True, _file_header=None, _endian_prefix=None)`
Parse a Dopplium Detections file.

**Parameters:**
- `filename` (str): Path to the detections binary file
- `max_batches` (int, optional): Maximum number of batches to read
- `verbose` (bool): Print parsing information

**Returns:**
- `detections` (np.ndarray): Structured array with detection fields
- `headers` (dict): Dictionary with 'file', 'body', and 'batch' headers

#### `filter_detections_by_range(detections, min_range, max_range)`
Filter detections by range.

#### `filter_detections_by_velocity(detections, min_velocity, max_velocity)`
Filter detections by velocity.

#### `filter_detections_by_amplitude(detections, min_amplitude)`
Filter detections by amplitude threshold.

#### `get_detection_statistics(detections)`
Get statistics about detections (min, max, mean, std for each field).

### Classes

#### `DetectionsBodyHeader`
Dataclass containing body header fields.

#### `DetectionsBatchHeader`
Dataclass containing batch/payload header fields.

## Notes

- Detections are stored in batches, which may have varying numbers of detections
- Empty batches (zero detections) are supported
- Batch headers include timestamps for temporal tracking
- The parser adds `batch_index` and `sequence_number` fields to help track detection provenance
- Cell indices allow correlation back to the processing grid (range-doppler-angle cube)

## See Also

- `parse_dopplium_rdch.py` - Parser for Range-Doppler-Channel data
- `parse_dopplium_radarcube.py` - Parser for RadarCube data
- `parse_dopplium_raw.py` - Parser for raw ADC data
- `../python_writer/DetectionsBinaryWriter.py` - Writer for Detections format

