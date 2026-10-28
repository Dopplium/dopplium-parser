function [data, headers] = parseADCData(fid, FH, machinefmt, filename, opts)
% PARSEADCDATA Parse Dopplium ADC/Raw Data (message_type=3)
%   [data, headers] = parseADCData(fid, FH, machinefmt, filename, opts)
%
%   INPUTS
%     fid        : file identifier (positioned after file header)
%     FH         : file header struct (already parsed)
%     machinefmt : endianness ('ieee-le' or 'ieee-be')
%     filename   : file path (for size calculation)
%     opts       : options struct with fields:
%                  .maxFrames, .startFrame, .cast, .returnComplex, .verbose
%
%   OUTPUTS
%     data    : array shaped [samples, chirpsPerTx, channel, frame]
%     headers : struct with file/body/frame headers and derived info
%
%   DOP-BIN 2.0 (file version 6, config_version 4, multiplexing_mode 2):
%   every frame carries a tx_map, its repeating transmitter pattern, and
%   n_chirps_per_frame counts every chirp. data is then shaped
%   [samples, chirps, receivers, frame] with the chirps in transmission
%   order; chirp c (1-based) of frame i was sent by
%   headers.frame(i).tx_map(mod(c-1, L) + 1), L = numel(tx_map). An empty
%   tx_map means the frame carries no transmitter information. A frame with
%   an invalid tx_map raises an error. Use splitPatternSlots to group a
%   frame by pattern slot.

if nargin < 5 || isempty(opts)
    opts = struct;
end
if isfield(opts, 'start_frame') && ~isfield(opts, 'startFrame')
    opts.startFrame = opts.start_frame;
end
if ~isfield(opts, 'maxFrames'), opts.maxFrames = Inf; end
if ~isfield(opts, 'startFrame'), opts.startFrame = 0; end
if ~isfield(opts, 'cast'), opts.cast = 'single'; end
if ~isfield(opts, 'returnComplex'), opts.returnComplex = true; end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
maxFrames = coerceOptionalNonnegativeInt(opts.maxFrames, 'maxFrames', Inf, true);
startFrame = coerceOptionalNonnegativeInt(opts.startFrame, 'startFrame', 0, false);

% -------------------------------------------------------------------------
% Read body header (ADC/RawData format)
% -------------------------------------------------------------------------
% Honor file_header_size for forward compatibility with possible file-header extensions.
fseek(fid, FH.file_header_size, 'bof');
BH = readBodyHeader(fid, machinefmt);
assert(BH.body_header_size >= 192, 'Unexpected body_header_size (BH).');
assert(BH.frame_header_size >= 24, 'Unexpected frame_header_size (BH).');

if opts.verbose
    printHeaderSummary(FH, BH);
end

% -------------------------------------------------------------------------
% Dimensions & types (trust header bytes for sizing)
% -------------------------------------------------------------------------
S      = double(BH.n_samples_per_chirp);
CptxHdr = double(BH.n_chirps_per_frame); % spec: chirps per frame (legacy files may store per-TX)
nRx    = double(BH.n_receivers);
nTx    = double(BH.n_transmitters);
bytesPerFrame = double(BH.bytes_per_frame);

if ~(BH.data_order == 0 || BH.data_order == 1)
    error('Unsupported data_order=%d. Only 0 (ByChannel) and 1 (BySample) are supported.', BH.data_order);
end
assert(BH.sample_format == 0, 'Only 16-bit aligned samples supported (sample_format==0).');
assert(ismember(double(BH.bits_per_sample), [12, 14, 16]), ...
    'Unsupported bits_per_sample=%d (expected one of 12, 14, 16).', BH.bits_per_sample);

txMapMode = checkMultiplexingMode(FH, BH);
if txMapMode
    % n_transmitters is information only in mode 2: the chirps stay in
    % transmission order and each frame's tx_map names their transmitters.
    nTx = 1;
end

% Header-guided sizing
bytesPerElement   = double(BH.bytes_per_element);     % should be 2 if int16 containers
bytesPerSample    = double(BH.bytes_per_sample);      % bytes per (real or complex) sample
intsPerElement    = bytesPerElement / 2;              % normally 1
elementsPerSample = bytesPerSample / bytesPerElement; % 1 for real, 2 for complex IQ
if abs(elementsPerSample - round(elementsPerSample)) > 1e-9
    warning('Non-integer elementsPerSample detected from header (%.3f). Rounding.', elementsPerSample);
end
elementsPerSample = round(elementsPerSample);

% Expected ints per (channel,chirp) block, from header bytes
blockLenInts = S * elementsPerSample * intsPerElement;

% Infer total chirps on wire from payload bytes.
nInt16_hdr = bytesPerFrame / 2;
denom = blockLenInts * nRx;
if denom <= 0
    error('Invalid header values: cannot infer chirp count from payload size.');
end
CtotFloat = nInt16_hdr / denom;
Ctot = round(CtotFloat);
if abs(CtotFloat - Ctot) > 1e-9
    warning('Non-integer total chirp count inferred (%.6f). Rounding to %d.', CtotFloat, Ctot);
end
if Ctot <= 0
    error('Invalid inferred total chirp count: %d.', Ctot);
end
if txMapMode && CptxHdr * denom * 2 ~= bytesPerFrame
    error('dopplium:txMapRequirements', ...
        'multiplexing_mode 2: bytes_per_frame=%d does not hold n_chirps_per_frame=%d chirps x %d receivers x %d samples.', ...
        bytesPerFrame, CptxHdr, nRx, S);
end

% Build TX mapping from multiplexing mode and channel order.
[txSeq, cTxSeq, chirpsPerTx, chirpInterpretation] = ...
    buildTxMapping(BH, Ctot, nTx, CptxHdr, opts.verbose);
if txMapMode
    chirpInterpretation = 'tx_map (all chirps, transmission order)';
end

% Determine number of frames from file size
fileInfo = dir(filename);
bytesAfterHeaders = fileInfo.bytes - FH.file_header_size - BH.body_header_size;
bytesPerUnit = BH.frame_header_size + bytesPerFrame;
nFramesTotal = floor(bytesAfterHeaders / bytesPerUnit);
nFramesAvailable = max(0, nFramesTotal - startFrame);
nFrames = min(nFramesAvailable, maxFrames);
if isfinite(maxFrames) && maxFrames > nFramesAvailable
    warning('Requested maxFrames exceeds available content from startFrame. Reading %d frames.', nFramesAvailable);
end

% Output dimensions
if nTx > 1
    nChanOut = nTx * nRx;
else
    nChanOut = nRx;
end

% Allocate output
outClass = mapOutClass(BH.sample_type, opts.cast, opts.returnComplex);
if BH.sample_type == 0 % real
    data = zeros(S, chirpsPerTx, nChanOut, nFrames, outClass);
else % complex
    if opts.returnComplex
        t = mapFloat(opts.cast);
        data = complex(zeros(S, chirpsPerTx, nChanOut, nFrames, t), ...
                       zeros(S, chirpsPerTx, nChanOut, nFrames, t));
    else
        data = zeros(S, chirpsPerTx, nChanOut, nFrames, 'int16'); % uncommon path
    end
end

% -------------------------------------------------------------------------
% Read frames
% -------------------------------------------------------------------------
fseek(fid, FH.file_header_size + BH.body_header_size + startFrame * bytesPerUnit, 'bof');
frames = repmat(emptyFrameHeader(), nFrames, 1);

for f = 1:nFrames
    frameIndex = startFrame + f - 1;
    frameOrdinal = frameIndex + 1;
    FR = readFrameHeader(fid, machinefmt);

    if FR.frame_payload_size ~= bytesPerFrame
        error('Frame %d payload size mismatch: header=%d, expected=%d', ...
              frameOrdinal, FR.frame_payload_size, bytesPerFrame);
    end

    % Honor frame_header_size for forward compatibility.
    extraFrameHeaderBytes = double(FR.frame_header_size) - 24;
    if extraFrameHeaderBytes < 0
        error('Frame %d has invalid frame_header_size=%d (<24).', frameOrdinal, FR.frame_header_size);
    elseif txMapMode
        FR = readFrameTxMap(fid, machinefmt, FR, FH, Ctot, frameOrdinal);
    elseif extraFrameHeaderBytes > 0
        fseek(fid, extraFrameHeaderBytes, 'cof');
    end
    frames(f) = FR;

    % Ground truth count from header:
    nInt16_hdr = bytesPerFrame / 2;
    raw = fread(fid, nInt16_hdr, '*int16', 0, machinefmt);
    if numel(raw) ~= nInt16_hdr
        error('Unexpected EOF while reading frame %d payload.', frameOrdinal);
    end

    % Theoretical count (for info only)
    nInt16_theo = blockLenInts * nRx * Ctot;
    if nInt16_theo ~= nInt16_hdr
        warning(['Frame %d: header bytes imply %d int16, but theoretical calc suggests %d. ' ...
                 'Proceeding with header-derived sizing.'], frameOrdinal, nInt16_hdr, nInt16_theo);
    end

    % ---- Normalize block ordering to (blockLenInts, rx, c) ----
    nBlocks = nRx * Ctot;
    assert(numel(raw) == nBlocks * blockLenInts, ...
        'Frame %d: payload size does not match expected nBlocks*blockLenInts.', frameOrdinal);

    switch BH.data_order
        case 0 % ByChannel: on-wire grouping is [for c=1..Ctot, for rx=1..nRx] contiguous blocks
            buf = reshape(raw, blockLenInts, nRx, Ctot); % (ints, rx, c)

        case 1 % BySample: on-wire grouping is [for rx=1..nRx, for c=1..Ctot] contiguous blocks
            buf = reshape(raw, nRx, blockLenInts, Ctot); % (ints, c, rx)
            buf = permute(buf, [2, 1, 3]); % reorder to (ints, rx, c) for uniform processing

        otherwise
            error('Unsupported data_order=%d (should have been caught earlier).', BH.data_order);
    end

    % Populate output from buf(:, rx, c)
    if BH.sample_type == 0
        % ------------------------ REAL ------------------------
        % elementsPerSample should be 1 -> blockLenInts = S
        for c = 1:Ctot
            tx   = txSeq(c);
            c_tx = cTxSeq(c);
            for rx = 1:nRx
                seg = buf(:, rx, c); % int16 column
                if nTx == 1
                    data(:, c_tx, rx, f) = cast(seg, outClass);
                else
                    chOut = (tx-1)*nRx + rx;
                    data(:, c_tx, chOut, f) = cast(seg, outClass);
                end
            end
        end

    else
        % ------------------------ COMPLEX ------------------------
        % elementsPerSample should be 2 -> blockLenInts = 2*S
        for c = 1:Ctot
            tx   = txSeq(c);
            c_tx = cTxSeq(c);
            for rx = 1:nRx
                seg = buf(:, rx, c); % int16 column of length elementsPerSample*S
                if elementsPerSample == 2
                    % Decode based on body header config version
                    if BH.config_version == 1
                        z = decodeIQV1(seg, S, BH.iq_order, opts);
                    elseif ismember(double(BH.config_version), [2, 3, 4])
                        z = decodeIQV2(seg, S, BH.iq_order, opts);
                    else
                        error('Unsupported body header config_version: %d', BH.config_version);
                    end
                else
                    % Fallback: treat first half as I, second half as Q
                    half = numel(seg)/2;
                    I = seg(1:half);
                    Q = seg(half+1:end);
                    z = complex(cast(I, mapFloat(opts.cast)), cast(Q, mapFloat(opts.cast)));
                end

                if nTx == 1
                    if opts.returnComplex
                        data(:, c_tx, rx, f) = cast(z, mapFloat(opts.cast));
                    else
                        data(:, c_tx, rx, f) = int16(real(z));
                    end
                else
                    chOut = (tx-1)*nRx + rx;
                    if opts.returnComplex
                        data(:, c_tx, chOut, f) = cast(z, mapFloat(opts.cast));
                    else
                        data(:, c_tx, chOut, f) = int16(real(z));
                    end
                end
            end
        end
    end
end

% -------------------------------------------------------------------------
% Outputs
% -------------------------------------------------------------------------
headers.file  = FH;
headers.body  = BH;
headers.frame = frames;

if opts.verbose
    fprintf('Chirp interpretation: %s | total_on_wire=%d | chirpsPerTx(dim-2)=%d | nTx=%d\n', ...
        chirpInterpretation, Ctot, chirpsPerTx, nTx);
    if txMapMode
        fprintf('\nParsed data shape: [samples=%d, chirps=%d, receivers=%d, frames=%d]\n', ...
            size(data,1), size(data,2), size(data,3), size(data,4));
        printTxMapPatterns(frames);
    else
        fprintf('\nParsed data shape: [samples=%d, chirpsPerTx=%d, channels=%d, frames=%d]\n', ...
            size(data,1), size(data,2), size(data,3), size(data,4));
        if nTx > 1
            fprintf('Multi-TX: nTx=%d, total chirps on wire per frame=%d\n', nTx, Ctot);
        else
            fprintf('Single-TX: total chirps per frame=%d\n', Ctot);
        end
    end
end
end

% ====================== Version 2 Helper Functions ======================

function BH = readBodyHeader(fid, machinefmt)
    BH.config_magic            = char(fread(fid, [1,4], '*char'));
    BH.config_version          = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.body_header_size        = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.frame_header_size       = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.reserved1               = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.total_frame_size        = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.n_samples_per_chirp     = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.n_chirps_per_frame      = fread(fid, 1, 'uint32', 0, machinefmt); % spec: per frame
    BH.bits_per_sample         = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.n_receivers             = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.n_transmitters          = fread(fid, 1, 'uint16', 0, machinefmt);
    BH.sample_type             = fread(fid, 1, 'uint8',  0, machinefmt); % 0=real, 1=complex/IQ
    BH.data_order              = fread(fid, 1, 'uint8',  0, machinefmt); % 0=ByChannel, 1=BySample
    BH.iq_order                = fread(fid, 1, 'uint8',  0, machinefmt); % 0..3 (v1), 0..5 (v2/v3)
    BH.sample_format           = fread(fid, 1, 'uint8',  0, machinefmt); % 0=16b aligned
    BH.multiplexing_mode       = fread(fid, 1, 'uint8',  0, machinefmt); % 0=MIMO, 1=Beamforming
    BH.reserved2               = fread(fid, 1, 'uint8',  0, machinefmt);
    BH.start_freq_ghz          = fread(fid, 1, 'double', 0, machinefmt);
    BH.bandwidth_ghz           = fread(fid, 1, 'double', 0, machinefmt);
    BH.idle_time_us            = fread(fid, 1, 'double', 0, machinefmt);
    BH.tx_start_time_us        = fread(fid, 1, 'double', 0, machinefmt);
    BH.adc_start_time_us       = fread(fid, 1, 'double', 0, machinefmt);
    BH.ramp_end_time_us        = fread(fid, 1, 'double', 0, machinefmt);
    BH.sample_rate_ksps        = fread(fid, 1, 'double', 0, machinefmt);
    BH.slope_mhz_per_us        = fread(fid, 1, 'double', 0, machinefmt);
    BH.frame_periodicity_ms    = fread(fid, 1, 'double', 0, machinefmt);
    BH.bytes_per_element       = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.bytes_per_sample        = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.samples_per_frame       = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.bytes_per_frame         = fread(fid, 1, 'uint32', 0, machinefmt);
    BH.max_range_m             = fread(fid, 1, 'single', 0, machinefmt);
    BH.max_velocity_mps        = fread(fid, 1, 'single', 0, machinefmt);
    BH.range_resolution_m      = fread(fid, 1, 'single', 0, machinefmt);
    BH.velocity_resolution_mps = fread(fid, 1, 'single', 0, machinefmt);
    BH.channel_order           = fread(fid, 30, '*uint8', 0, machinefmt);
    BH.reserved3               = fread(fid, 22, '*uint8', 0, machinefmt);
end

function FR = readFrameHeader(fid, machinefmt)
    FR.frame_magic               = char(fread(fid, [1,4], '*char'));
    FR.header_type               = fread(fid, 1, 'uint16', 0, machinefmt);
    FR.frame_header_size         = fread(fid, 1, 'uint16', 0, machinefmt);
    FR.frame_timestamp_utc_ticks = fread(fid, 1, 'int64', 0, machinefmt);
    FR.frame_number              = fread(fid, 1, 'uint32', 0, machinefmt);
    FR.frame_payload_size        = fread(fid, 1, 'uint32', 0, machinefmt);

    if ~strcmp(FR.frame_magic, 'FRME')
        error('Invalid frame magic at frame read position.');
    end
    if FR.frame_header_size < 24
        error('Unexpected frame_header_size in frame header.');
    end
    FR.tx_map = []; % multiplexing_mode 2 only, set by readFrameTxMap
end

function s = emptyFrameHeader()
    s = struct( ...
        'frame_magic',              '', ...
        'header_type',              0, ...
        'frame_header_size',        0, ...
        'frame_timestamp_utc_ticks',0, ...
        'frame_number',             0, ...
        'frame_payload_size',       0, ...
        'tx_map',                   []);
end

% ================== DOP-BIN 2.0 (multiplexing_mode 2) ==================

function txMapMode = checkMultiplexingMode(FH, BH)
% CHECKMULTIPLEXINGMODE Validate multiplexing_mode; true for mode 2 (per-frame tx_map).
    mode = double(BH.multiplexing_mode);
    if mode > 2
        error('dopplium:unsupportedMultiplexingMode', ...
            ['Unsupported multiplexing_mode=%d. Supported: 0 (MIMO), 1 (Beamforming), ' ...
             '2 (TDM with per-frame tx_map).'], mode);
    end
    txMapMode = (mode == 2);
    if ~txMapMode
        return;
    end
    if double(FH.version) < 6
        error('dopplium:txMapRequirements', ...
            'multiplexing_mode 2 requires file version >= 6, got %d.', FH.version);
    end
    if double(BH.config_version) < 4
        error('dopplium:txMapRequirements', ...
            'multiplexing_mode 2 requires config_version >= 4, got %d.', BH.config_version);
    end
    if double(FH.payload_header_size) < 32 || double(BH.frame_header_size) ~= double(FH.payload_header_size)
        error('dopplium:txMapRequirements', ...
            ['multiplexing_mode 2 requires a payload header of at least 32 bytes in both file ' ...
             'and body header, got %d and %d.'], FH.payload_header_size, BH.frame_header_size);
    end
end

function FR = readFrameTxMap(fid, machinefmt, FR, FH, Ctot, frameOrdinal)
% READFRAMETXMAP Read the rest of a 32-byte payload header: tx_map (bytes 24-27), reserved.
    if double(FR.frame_header_size) ~= double(FH.payload_header_size)
        error('dopplium:txMapRequirements', ...
            'Frame %d: frame_header_size=%d differs from the file''s payload_header_size=%d.', ...
            frameOrdinal, FR.frame_header_size, FH.payload_header_size);
    end
    nExtra = double(FR.frame_header_size) - 24;
    extra = fread(fid, nExtra, '*uint8', 0, machinefmt);
    if numel(extra) ~= nExtra
        error('Unexpected EOF while reading frame %d header.', frameOrdinal);
    end
    FR.tx_map = parseTxMap(extra(1:4), frameOrdinal);
    L = numel(FR.tx_map);
    if L > 0 && mod(Ctot, L) ~= 0
        error('dopplium:txMapLength', ...
            'Frame %d: tx_map %s has length %d, which does not divide %d chirps.', ...
            frameOrdinal, mat2str(FR.tx_map), L, Ctot);
    end
end

function txMap = parseTxMap(raw, frameOrdinal)
% PARSETXMAP Pattern from the 4 tx_map bytes. The list ends at the first 0;
% a non-zero entry after it (e.g. [1 0 2 0]) is invalid and is never repaired.
    raw = double(raw(:)');
    L = find(raw == 0, 1, 'first') - 1;
    if isempty(L)
        L = numel(raw);
    end
    if any(raw(L+1:end))
        error('dopplium:invalidTxMap', ...
            'Frame %d: invalid tx_map %s: non-zero entry after the terminating 0.', ...
            frameOrdinal, mat2str(raw));
    end
    txMap = raw(1:L);
end

function printTxMapPatterns(frames)
% PRINTTXMAPPATTERNS List the distinct tx_map patterns and how many frames use each.
    keys = arrayfun(@(fr) ['[' strtrim(sprintf('%d ', fr.tx_map)) ']'], frames, 'UniformOutput', false);
    [patterns, ~, idx] = unique(keys, 'stable');
    counts = accumarray(idx(:), 1);
    parts = cell(1, numel(patterns));
    for k = 1:numel(patterns)
        parts{k} = sprintf('%s x%d', patterns{k}, counts(k));
    end
    fprintf('tx_map patterns (frames): %s\n', strjoin(parts, ', '));
end

function z = decodeIQV1(segInt16, S, iqOrder, opts)
% DECODEIQV1 Decode IQ data for body header v1 (IQ orders 0-3)
    switch iqOrder
        case 0 % IQ
            I = segInt16(1:2:end);
            Q = segInt16(2:2:end);
        case 1 % QI
            Q = segInt16(1:2:end);
            I = segInt16(2:2:end);
        case 2 % NonInterleaved (IIII... QQQQ...)
            I = segInt16(1:S);
            Q = segInt16(S+1:2*S);
        case 3 % BlockInterleaved: [I0 I1 Q0 Q1 I2 I3 Q2 Q3 ...]
            if mod(S,2) ~= 0
                I = zeros(S,1,'int16'); Q = zeros(S,1,'int16');
                ii = 1; qi = 1; k = 1;
                while k <= 2*S
                    take = min(2, S - (ii-1));
                    if take>0, I(ii:ii+take-1) = segInt16(k:k+take-1); k = k+take; ii = ii+take; end
                    take = min(2, S - (qi-1));
                    if take>0, Q(qi:qi+take-1) = segInt16(k:k+take-1); k = k+take; qi = qi+take; end
                end
            else
                g = reshape(segInt16, 4, []);
                I = reshape(g(1:2, :), [], 1);
                Q = reshape(g(3:4, :), [], 1);
            end
        otherwise
            error('Unsupported iq_order value: %d', iqOrder);
    end

    if opts.returnComplex
        Iflt = cast(I, mapFloat(opts.cast));
        Qflt = cast(Q, mapFloat(opts.cast));
        z = complex(Iflt, Qflt);
    else
        z = complex(cast(I, mapFloat(opts.cast)), cast(Q, mapFloat(opts.cast)));
    end
end

function z = decodeIQV2(segInt16, S, iqOrder, opts)
% DECODEIQV2 Decode IQ data for body header v2 (IQ orders 0-5)
    switch iqOrder
        case 0 % IQ: [I0, Q0, I1, Q1, ...]
            I = segInt16(1:2:end);
            Q = segInt16(2:2:end);
        case 1 % QI: [Q0, I0, Q1, I1, ...]
            Q = segInt16(1:2:end);
            I = segInt16(2:2:end);
        case 2 % NonInterleaved: [I0, I1, ..., I_n-1, Q0, Q1, ..., Q_n-1]
            I = segInt16(1:S);
            Q = segInt16(S+1:2*S);
        case 3 % NonInterleavedQ: [Q0, Q1, ..., Q_n-1, I0, I1, ..., I_n-1]
            Q = segInt16(1:S);
            I = segInt16(S+1:2*S);
        case 4 % BlockInterleaved: [I0, I1, Q0, Q1, I2, I3, Q2, Q3, ...]
            if mod(S,2) ~= 0
                I = zeros(S,1,'int16'); Q = zeros(S,1,'int16');
                ii = 1; qi = 1; k = 1;
                while k <= 2*S
                    take = min(2, S - (ii-1));
                    if take>0, I(ii:ii+take-1) = segInt16(k:k+take-1); k = k+take; ii = ii+take; end
                    take = min(2, S - (qi-1));
                    if take>0, Q(qi:qi+take-1) = segInt16(k:k+take-1); k = k+take; qi = qi+take; end
                end
            else
                g = reshape(segInt16, 4, []);
                I = reshape(g(1:2, :), [], 1);
                Q = reshape(g(3:4, :), [], 1);
            end
        case 5 % BlockInterleavedQ: [Q0, Q1, I0, I1, Q2, Q3, I2, I3, ...]
            if mod(S,2) ~= 0
                I = zeros(S,1,'int16'); Q = zeros(S,1,'int16');
                qi = 1; ii = 1; k = 1;
                while k <= 2*S
                    take = min(2, S - (qi-1));
                    if take>0, Q(qi:qi+take-1) = segInt16(k:k+take-1); k = k+take; qi = qi+take; end
                    take = min(2, S - (ii-1));
                    if take>0, I(ii:ii+take-1) = segInt16(k:k+take-1); k = k+take; ii = ii+take; end
                end
            else
                g = reshape(segInt16, 4, []);
                Q = reshape(g(1:2, :), [], 1);
                I = reshape(g(3:4, :), [], 1);
            end
        otherwise
            error('Unsupported iq_order value: %d (valid for v2: 0-5)', iqOrder);
    end

    if opts.returnComplex
        Iflt = cast(I, mapFloat(opts.cast));
        Qflt = cast(Q, mapFloat(opts.cast));
        z = complex(Iflt, Qflt);
    else
        z = complex(cast(I, mapFloat(opts.cast)), cast(Q, mapFloat(opts.cast)));
    end
end

function [txSeq, cTxSeq, chirpsPerTx, interpLabel] = buildTxMapping(BH, Ctot, nTx, cptxHdr, verbose)
% BUILDTXMAPPING Determine TX index and per-TX chirp ordinal for each chirp on wire.
    nTxEff = max(1, nTx);
    if nTxEff == 1
        txSeq = ones(1, Ctot);
    else
        if ~(BH.multiplexing_mode == 0 || BH.multiplexing_mode == 1)
            error('Unsupported multiplexing_mode=%d.', BH.multiplexing_mode);
        end

        if BH.multiplexing_mode == 0
            txOrder = double(BH.channel_order(:)');
            txOrder = txOrder(txOrder > 0 & txOrder <= nTxEff);
            if isempty(txOrder)
                txOrder = 1:nTxEff;
                if verbose
                    warning(['channel_order is empty/invalid for multiplexing_mode=0. ' ...
                             'Falling back to sequential TX order.']);
                end
            end
        else
            % Beamforming mode does not define per-chirp TX cycling in the current parser API.
            txOrder = 1:nTxEff;
            if verbose
                warning(['multiplexing_mode=1 (Beamforming). Using sequential TX mapping ' ...
                         'for compatibility.']);
            end
        end

        txSeq = repmat(txOrder, 1, ceil(Ctot / numel(txOrder)));
        txSeq = txSeq(1:Ctot);
    end

    txCounts = zeros(1, nTxEff);
    cTxSeq = zeros(1, Ctot);
    for c = 1:Ctot
        tx = txSeq(c);
        txCounts(tx) = txCounts(tx) + 1;
        cTxSeq(c) = txCounts(tx);
    end
    chirpsPerTx = max(txCounts);

    if nTxEff > 1 && cptxHdr * nTxEff == Ctot
        interpLabel = 'per-tx';
    elseif cptxHdr == Ctot
        interpLabel = 'per-frame';
    else
        interpLabel = 'inferred';
    end
end
