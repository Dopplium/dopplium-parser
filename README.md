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
- **message_type = 3**: RawData → `parse_dopplium_raw`
- **message_type = 5**: RDCh (Range-Doppler-Channel) → `parse_dopplium_rdch`

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
Returns data shaped `[range_bins, doppler_bins, channels, chunks]`:
- **Range bins**: Processed range dimension
- **Doppler bins**: Processed velocity/Doppler dimension
- **Channels**: Number of receiver channels
- **Chunks**: Number of processed data chunks

## Options

**Common options** (`parse_dopplium`, `parse_dopplium_raw`, `parse_dopplium_rdch`):
- `max_chunks_or_frames` / `max_frames` / `max_chunks`: Limit chunks/frames to read
- `verbose`: Show parsing info

**RawData-specific options**:
- `cast`: Output type ('float32', 'float64', 'int16')
- `return_complex`: Return complex numbers for IQ data

## Examples

See `matlab_parser/exampleParse.m` and `python_parser/example_parse.py` for complete examples with 2D FFT processing and visualization.