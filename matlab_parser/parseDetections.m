function [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts)
% PARSEDETECTIONS Parse Dopplium Detections data (v3/v4/v5 message_type=4)
%   [data, headers] = parseDetections(fid, FH, machinefmt, filename, opts)
%
%   Returns:
%     data    : struct of column vectors (fields aligned with Python parser)
%     headers : struct with fields .file, .body, .batch

if FH.version == 2
    error('parseDetections:Version2NotImplemented', ...
        'Version 2 Detections are not implemented in this parser.');
end
if ~(FH.version == 3 || FH.version == 4 || FH.version == 5) || FH.message_type ~= 4
    error('parseDetections:InvalidMessageType', ...
        ['This file is not Detections (version=%d, message_type=%d, ' ...
         'expected version=3/4/5, type=4).'], ...
        FH.version, FH.message_type);
end

if nargin < 6 || isempty(opts)
    opts = struct;
end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'maxFrames'), opts.maxFrames = Inf; end
maxBatches = opts.maxFrames;
if isfield(opts, 'maxBatches')
    maxBatches = opts.maxBatches;
end

% Read body header
fseek(fid, FH.file_header_size, 'bof');
BH = readDetectionsBodyHeader(fid, machinefmt);

if opts.verbose
    printHeaderSummary(FH, BH);
end

fileInfo = dir(filename);
bytesAfterHeaders = double(fileInfo.bytes) - double(FH.file_header_size) - double(BH.body_header_size);
avgBatchSize = double(BH.payload_header_size) + (10 * double(BH.detection_record_size));
if avgBatchSize <= 0
    nBatchesEstimate = 1;
else
    nBatchesEstimate = max(1, floor(bytesAfterHeaders / avgBatchSize));
end

if opts.verbose
    fprintf('\nBytes after headers: %d\n', bytesAfterHeaders);
    fprintf('Estimated batches in file: ~%d\n', nBatchesEstimate);
end

% Read batches
fseek(fid, FH.file_header_size + BH.body_header_size, 'bof');
batchHeaders = repmat(emptyBatchHeader(), 0, 1);
chunks = {};
batchesRead = 0;

while true
    if batchesRead >= maxBatches
        break;
    end
    if ftell(fid) >= fileInfo.bytes
        break;
    end

    try
        BHdr = readBatchHeader(fid, machinefmt);
        batchHeaders(end+1,1) = BHdr; %#ok<AGROW>

        if opts.verbose && (batchesRead == 0 || mod(batchesRead + 1, 100) == 0)
            fprintf('  Reading batch %d: seq=%d, detections=%d, size=%d bytes\n', ...
                batchesRead + 1, BHdr.sequence_number, BHdr.detection_count, BHdr.payload_size_bytes);
        end

        expectedPayloadSize = double(BHdr.detection_count) * double(BH.detection_record_size);
        if double(BHdr.payload_size_bytes) ~= expectedPayloadSize
            if opts.verbose
                warning('parseDetections:PayloadSizeMismatch', ...
                    'Batch %d payload size mismatch: expected=%d, got=%d', ...
                    batchesRead, expectedPayloadSize, BHdr.payload_size_bytes);
            end
        end

        % Honor payload_header_size for forward compatibility.
        extraHeaderBytes = double(BHdr.payload_header_size) - 32;
        if extraHeaderBytes < 0
            error('parseDetections:InvalidHeaderSize', ...
                'Batch %d has invalid payload_header_size=%d (<32).', ...
                batchesRead + 1, BHdr.payload_header_size);
        elseif extraHeaderBytes > 0
            fseek(fid, extraHeaderBytes, 'cof');
        end

        if BHdr.detection_count > 0
            chunk = readDetectionRecords(fid, double(BHdr.detection_count), machinefmt, ...
                uint32(batchesRead), uint32(BHdr.sequence_number));
            chunks{end+1} = chunk; %#ok<AGROW>
        end

        batchesRead = batchesRead + 1;
    catch ME
        if strcmp(ME.identifier, 'parseDetections:UnexpectedEOF')
            if opts.verbose
                fprintf('Reached end of file after reading %d batches.\n', batchesRead);
            end
            break;
        end
        rethrow(ME);
    end
end

if isempty(chunks)
    data = emptyDetections();
else
    data = concatDetectionChunks(chunks);
end

headers.file  = FH;
headers.body  = BH;
headers.batch = batchHeaders;

if opts.verbose
    fprintf('\nTotal batches read: %d\n', batchesRead);
    fprintf('Total detections: %d\n', numel(data.range));
end
end

function BH = readDetectionsBodyHeader(fid, machinefmt)
BH.body_magic           = char(fread(fid, [1,4], '*char'));
BH.body_header_version  = fread(fid, 1, 'uint16', 0, machinefmt);
BH.body_header_size     = fread(fid, 1, 'uint16', 0, machinefmt);
BH.payload_header_size  = fread(fid, 1, 'uint16', 0, machinefmt);
BH.detection_record_size= fread(fid, 1, 'uint16', 0, machinefmt);
BH.algorithm_id         = fread(fid, 1, 'uint32', 0, machinefmt);
BH.algorithm_version    = fread(fid, 1, 'uint32', 0, machinefmt);
BH.reserved             = fread(fid, 44, '*uint8', 0, machinefmt);

if ~strcmp(BH.body_magic, 'DETC')
    warning('parseDetections:MagicMismatch', ...
        'Unexpected detections body magic "%s" (expected "DETC").', BH.body_magic);
end
end

function BHdr = readBatchHeader(fid, machinefmt)
BHdr.payload_magic      = char(fread(fid, [1,4], '*char'));
BHdr.payload_header_size= fread(fid, 1, 'uint16', 0, machinefmt);
BHdr.timestamp_utc_ticks= fread(fid, 1, 'int64',  0, machinefmt);
BHdr.reserved1          = fread(fid, 1, 'uint16', 0, machinefmt);
BHdr.detection_count    = fread(fid, 1, 'uint32', 0, machinefmt);
BHdr.sequence_number    = fread(fid, 1, 'uint32', 0, machinefmt);
BHdr.payload_size_bytes = fread(fid, 1, 'uint32', 0, machinefmt);
BHdr.reserved2          = fread(fid, 1, 'uint32', 0, machinefmt);

if isempty(BHdr.reserved2)
    error('parseDetections:UnexpectedEOF', 'Failed to read detections batch header.');
end
if ~strcmp(BHdr.payload_magic, 'BTCH')
    error('parseDetections:InvalidBatchMagic', ...
        'Invalid batch magic "%s" (expected "BTCH").', BHdr.payload_magic);
end
end

function chunk = readDetectionRecords(fid, count, machinefmt, batchIndex, sequenceNumber)
chunk = emptyDetections();

chunk.range               = zeros(count,1,'double');
chunk.velocity            = zeros(count,1,'double');
chunk.azimuth             = zeros(count,1,'double');
chunk.elevation           = zeros(count,1,'double');
chunk.amplitude           = zeros(count,1,'double');
chunk.monopulse_ratio_az  = zeros(count,1,'single');
chunk.monopulse_ratio_el  = zeros(count,1,'single');
chunk.range_cell          = zeros(count,1,'int16');
chunk.doppler_cell        = zeros(count,1,'int16');
chunk.azimuth_cell        = zeros(count,1,'int16');
chunk.elevation_cell      = zeros(count,1,'int16');
chunk.batch_index         = repmat(batchIndex, count, 1);
chunk.sequence_number     = repmat(sequenceNumber, count, 1);

for i = 1:count
    chunk.range(i)              = readScalar(fid, 'double', machinefmt, 'range');
    chunk.velocity(i)           = readScalar(fid, 'double', machinefmt, 'velocity');
    chunk.azimuth(i)            = readScalar(fid, 'double', machinefmt, 'azimuth');
    chunk.elevation(i)          = readScalar(fid, 'double', machinefmt, 'elevation');
    chunk.amplitude(i)          = readScalar(fid, 'double', machinefmt, 'amplitude');
    chunk.monopulse_ratio_az(i) = readScalar(fid, 'single', machinefmt, 'monopulse_ratio_az');
    chunk.monopulse_ratio_el(i) = readScalar(fid, 'single', machinefmt, 'monopulse_ratio_el');
    chunk.range_cell(i)         = readScalar(fid, 'int16',  machinefmt, 'range_cell');
    chunk.doppler_cell(i)       = readScalar(fid, 'int16',  machinefmt, 'doppler_cell');
    chunk.azimuth_cell(i)       = readScalar(fid, 'int16',  machinefmt, 'azimuth_cell');
    chunk.elevation_cell(i)     = readScalar(fid, 'int16',  machinefmt, 'elevation_cell');
end
end

function v = readScalar(fid, typeName, machinefmt, fieldName)
v = fread(fid, 1, ['*' typeName], 0, machinefmt);
if isempty(v)
    error('parseDetections:UnexpectedEOF', ...
        'Unexpected EOF while reading detection field "%s".', fieldName);
end
end

function out = concatDetectionChunks(chunks)
out = emptyDetections();
out.range              = vertcatField(chunks, 'range');
out.velocity           = vertcatField(chunks, 'velocity');
out.azimuth            = vertcatField(chunks, 'azimuth');
out.elevation          = vertcatField(chunks, 'elevation');
out.amplitude          = vertcatField(chunks, 'amplitude');
out.monopulse_ratio_az = vertcatField(chunks, 'monopulse_ratio_az');
out.monopulse_ratio_el = vertcatField(chunks, 'monopulse_ratio_el');
out.range_cell         = vertcatField(chunks, 'range_cell');
out.doppler_cell       = vertcatField(chunks, 'doppler_cell');
out.azimuth_cell       = vertcatField(chunks, 'azimuth_cell');
out.elevation_cell     = vertcatField(chunks, 'elevation_cell');
out.batch_index        = vertcatField(chunks, 'batch_index');
out.sequence_number    = vertcatField(chunks, 'sequence_number');
end

function v = vertcatField(chunks, fieldName)
vals = cellfun(@(c) c.(fieldName), chunks, 'UniformOutput', false);
v = vertcat(vals{:});
end

function d = emptyDetections()
d = struct( ...
    'range',              zeros(0,1,'double'), ...
    'velocity',           zeros(0,1,'double'), ...
    'azimuth',            zeros(0,1,'double'), ...
    'elevation',          zeros(0,1,'double'), ...
    'amplitude',          zeros(0,1,'double'), ...
    'monopulse_ratio_az', zeros(0,1,'single'), ...
    'monopulse_ratio_el', zeros(0,1,'single'), ...
    'range_cell',         zeros(0,1,'int16'), ...
    'doppler_cell',       zeros(0,1,'int16'), ...
    'azimuth_cell',       zeros(0,1,'int16'), ...
    'elevation_cell',     zeros(0,1,'int16'), ...
    'batch_index',        zeros(0,1,'uint32'), ...
    'sequence_number',    zeros(0,1,'uint32'));
end

function s = emptyBatchHeader()
s = struct( ...
    'payload_magic',       '', ...
    'payload_header_size', uint16(0), ...
    'timestamp_utc_ticks', int64(0), ...
    'reserved1',           uint16(0), ...
    'detection_count',     uint32(0), ...
    'sequence_number',     uint32(0), ...
    'payload_size_bytes',  uint32(0), ...
    'reserved2',           uint32(0));
end

function printHeaderSummary(FH, BH)
fprintf('--- Dopplium Detections Data ---\n');
fprintf('Magic=%s  Version=%d  Endianness=%s  MessageType=%d\n', ...
    FH.magic, FH.version, tern(FH.endianness==1,'LE','BE'), FH.message_type);
fprintf('FileHdr=%d  BodyHdr=%d  BatchHdr=%d  TotalBatchesWritten=%d\n', ...
    FH.file_header_size, BH.body_header_size, BH.payload_header_size, FH.total_frames_written);
fprintf('NodeId="%s"\n', FH.node_id);

fprintf('\n-- Detections Configuration --\n');
fprintf('Detection record size: %d bytes\n', BH.detection_record_size);
fprintf('Algorithm ID: %d\n', BH.algorithm_id);
fprintf('Algorithm version: %d\n', BH.algorithm_version);
fprintf('Body header version: %d\n', BH.body_header_version);
end
