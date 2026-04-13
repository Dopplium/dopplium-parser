function [data, headers] = parseBlobs(fid, FH, machinefmt, filename, opts)
% PARSEBLOBS Parse Dopplium Blobs data (v3/v4/v5 message_type=5)
%   [data, headers] = parseBlobs(fid, FH, machinefmt, filename, opts)
%
%   Returns:
%     data    : struct of column vectors (fields aligned with Python parser)
%     headers : struct with fields .file, .body, .batch

if ~(FH.version == 3 || FH.version == 4 || FH.version == 5) || FH.message_type ~= 5
    error('parseBlobs:InvalidMessageType', ...
        ['This file is not Blobs (version=%d, message_type=%d, ' ...
         'expected version=3/4/5, type=5).'], ...
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
BH = readBlobsBodyHeader(fid, machinefmt);

if opts.verbose
    printHeaderSummary(FH, BH);
end

fileInfo = dir(filename);
bytesAfterHeaders = double(fileInfo.bytes) - double(FH.file_header_size) - double(BH.body_header_size);
avgBatchSize = double(BH.batch_header_size) + (5 * double(BH.blob_record_size));
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
            fprintf('  Reading batch %d: seq=%d, blobs=%d, size=%d bytes\n', ...
                batchesRead + 1, BHdr.sequence_number, BHdr.blob_count, BHdr.payload_size_bytes);
        end

        expectedPayloadSize = double(BHdr.blob_count) * double(BH.blob_record_size);
        if double(BHdr.payload_size_bytes) ~= expectedPayloadSize
            if opts.verbose
                warning('parseBlobs:PayloadSizeMismatch', ...
                    'Batch %d payload size mismatch: expected=%d, got=%d', ...
                    batchesRead, expectedPayloadSize, BHdr.payload_size_bytes);
            end
        end

        % Honor batch_header_size for forward compatibility.
        extraHeaderBytes = double(BHdr.batch_header_size) - 30;
        if extraHeaderBytes < 0
            error('parseBlobs:InvalidHeaderSize', ...
                'Batch %d has invalid batch_header_size=%d (<30).', ...
                batchesRead + 1, BHdr.batch_header_size);
        elseif extraHeaderBytes > 0
            fseek(fid, extraHeaderBytes, 'cof');
        end

        if BHdr.blob_count > 0
            chunk = readBlobRecords(fid, double(BHdr.blob_count), machinefmt, ...
                uint32(batchesRead), uint32(BHdr.sequence_number));
            chunks{end+1} = chunk; %#ok<AGROW>
        end

        batchesRead = batchesRead + 1;
    catch ME
        if strcmp(ME.identifier, 'parseBlobs:UnexpectedEOF')
            if opts.verbose
                fprintf('Reached end of file after reading %d batches.\n', batchesRead);
            end
            break;
        end
        rethrow(ME);
    end
end

if isempty(chunks)
    data = emptyBlobs();
else
    data = concatBlobChunks(chunks);
end

headers.file  = FH;
headers.body  = BH;
headers.batch = batchHeaders;

if opts.verbose
    fprintf('\nTotal batches read: %d\n', batchesRead);
    fprintf('Total blobs: %d\n', numel(data.range_centroid));
end
end

function BH = readBlobsBodyHeader(fid, machinefmt)
BH.body_magic           = char(fread(fid, [1,4], '*char'));
BH.body_header_version  = fread(fid, 1, 'uint16', 0, machinefmt);
BH.body_header_size     = fread(fid, 1, 'uint16', 0, machinefmt);
BH.batch_header_size    = fread(fid, 1, 'uint16', 0, machinefmt);
BH.blob_record_size     = fread(fid, 1, 'uint16', 0, machinefmt);
BH.algorithm_id         = fread(fid, 1, 'uint32', 0, machinefmt);
BH.algorithm_version    = fread(fid, 1, 'uint32', 0, machinefmt);
BH.reserved             = fread(fid, 44, '*uint8', 0, machinefmt);

if ~strcmp(BH.body_magic, 'BLOB')
    warning('parseBlobs:MagicMismatch', ...
        'Unexpected blobs body magic "%s" (expected "BLOB").', BH.body_magic);
end
end

function BHdr = readBatchHeader(fid, machinefmt)
BHdr.batch_magic        = char(fread(fid, [1,4], '*char'));
BHdr.batch_header_size  = fread(fid, 1, 'uint16', 0, machinefmt);
BHdr.timestamp_utc_ticks= fread(fid, 1, 'int64',  0, machinefmt);
BHdr.reserved1          = fread(fid, 1, 'uint16', 0, machinefmt);
BHdr.blob_count         = fread(fid, 1, 'uint16', 0, machinefmt);
BHdr.sequence_number    = fread(fid, 1, 'uint32', 0, machinefmt);
BHdr.payload_size_bytes = fread(fid, 1, 'uint32', 0, machinefmt);
BHdr.reserved2          = fread(fid, 1, 'uint32', 0, machinefmt);

if isempty(BHdr.reserved2)
    error('parseBlobs:UnexpectedEOF', 'Failed to read blobs batch header.');
end
if ~strcmp(BHdr.batch_magic, 'BTCH')
    error('parseBlobs:InvalidBatchMagic', ...
        'Invalid batch magic "%s" (expected "BTCH").', BHdr.batch_magic);
end
end

function chunk = readBlobRecords(fid, count, machinefmt, batchIndex, sequenceNumber)
chunk = emptyBlobs();

chunk.range_centroid    = zeros(count,1,'single');
chunk.velocity_centroid = zeros(count,1,'single');
chunk.azimuth_centroid  = zeros(count,1,'single');
chunk.elevation_centroid= zeros(count,1,'single');
chunk.range_spread      = zeros(count,1,'single');
chunk.velocity_spread   = zeros(count,1,'single');
chunk.azimuth_spread    = zeros(count,1,'single');
chunk.elevation_spread  = zeros(count,1,'single');
chunk.amplitude         = zeros(count,1,'single');
chunk.range_cell        = zeros(count,1,'int16');
chunk.doppler_cell      = zeros(count,1,'int16');
chunk.azimuth_cell      = zeros(count,1,'int16');
chunk.elevation_cell    = zeros(count,1,'int16');
chunk.blob_id           = zeros(count,1,'uint32');
chunk.num_detections    = zeros(count,1,'uint16');
chunk.batch_index       = repmat(batchIndex, count, 1);
chunk.sequence_number   = repmat(sequenceNumber, count, 1);

for i = 1:count
    chunk.range_centroid(i)     = readScalar(fid, 'single', machinefmt, 'range_centroid');
    chunk.velocity_centroid(i)  = readScalar(fid, 'single', machinefmt, 'velocity_centroid');
    chunk.azimuth_centroid(i)   = readScalar(fid, 'single', machinefmt, 'azimuth_centroid');
    chunk.elevation_centroid(i) = readScalar(fid, 'single', machinefmt, 'elevation_centroid');
    chunk.range_spread(i)       = readScalar(fid, 'single', machinefmt, 'range_spread');
    chunk.velocity_spread(i)    = readScalar(fid, 'single', machinefmt, 'velocity_spread');
    chunk.azimuth_spread(i)     = readScalar(fid, 'single', machinefmt, 'azimuth_spread');
    chunk.elevation_spread(i)   = readScalar(fid, 'single', machinefmt, 'elevation_spread');
    chunk.amplitude(i)          = readScalar(fid, 'single', machinefmt, 'amplitude');
    chunk.range_cell(i)         = readScalar(fid, 'int16',  machinefmt, 'range_cell');
    chunk.doppler_cell(i)       = readScalar(fid, 'int16',  machinefmt, 'doppler_cell');
    chunk.azimuth_cell(i)       = readScalar(fid, 'int16',  machinefmt, 'azimuth_cell');
    chunk.elevation_cell(i)     = readScalar(fid, 'int16',  machinefmt, 'elevation_cell');
    chunk.blob_id(i)            = readScalar(fid, 'uint32', machinefmt, 'blob_id');
    chunk.num_detections(i)     = readScalar(fid, 'uint16', machinefmt, 'num_detections');

    padAndReserved = fread(fid, 6, '*uint8', 0, machinefmt); % 2 bytes padding + 4 bytes reserved
    if numel(padAndReserved) ~= 6
        error('parseBlobs:UnexpectedEOF', 'Unexpected EOF while reading blob padding/reserved bytes.');
    end
end
end

function v = readScalar(fid, typeName, machinefmt, fieldName)
v = fread(fid, 1, ['*' typeName], 0, machinefmt);
if isempty(v)
    error('parseBlobs:UnexpectedEOF', ...
        'Unexpected EOF while reading blob field "%s".', fieldName);
end
end

function out = concatBlobChunks(chunks)
out = emptyBlobs();
out.range_centroid    = vertcatField(chunks, 'range_centroid');
out.velocity_centroid = vertcatField(chunks, 'velocity_centroid');
out.azimuth_centroid  = vertcatField(chunks, 'azimuth_centroid');
out.elevation_centroid= vertcatField(chunks, 'elevation_centroid');
out.range_spread      = vertcatField(chunks, 'range_spread');
out.velocity_spread   = vertcatField(chunks, 'velocity_spread');
out.azimuth_spread    = vertcatField(chunks, 'azimuth_spread');
out.elevation_spread  = vertcatField(chunks, 'elevation_spread');
out.amplitude         = vertcatField(chunks, 'amplitude');
out.range_cell        = vertcatField(chunks, 'range_cell');
out.doppler_cell      = vertcatField(chunks, 'doppler_cell');
out.azimuth_cell      = vertcatField(chunks, 'azimuth_cell');
out.elevation_cell    = vertcatField(chunks, 'elevation_cell');
out.blob_id           = vertcatField(chunks, 'blob_id');
out.num_detections    = vertcatField(chunks, 'num_detections');
out.batch_index       = vertcatField(chunks, 'batch_index');
out.sequence_number   = vertcatField(chunks, 'sequence_number');
end

function v = vertcatField(chunks, fieldName)
vals = cellfun(@(c) c.(fieldName), chunks, 'UniformOutput', false);
v = vertcat(vals{:});
end

function d = emptyBlobs()
d = struct( ...
    'range_centroid',     zeros(0,1,'single'), ...
    'velocity_centroid',  zeros(0,1,'single'), ...
    'azimuth_centroid',   zeros(0,1,'single'), ...
    'elevation_centroid', zeros(0,1,'single'), ...
    'range_spread',       zeros(0,1,'single'), ...
    'velocity_spread',    zeros(0,1,'single'), ...
    'azimuth_spread',     zeros(0,1,'single'), ...
    'elevation_spread',   zeros(0,1,'single'), ...
    'amplitude',          zeros(0,1,'single'), ...
    'range_cell',         zeros(0,1,'int16'), ...
    'doppler_cell',       zeros(0,1,'int16'), ...
    'azimuth_cell',       zeros(0,1,'int16'), ...
    'elevation_cell',     zeros(0,1,'int16'), ...
    'blob_id',            zeros(0,1,'uint32'), ...
    'num_detections',     zeros(0,1,'uint16'), ...
    'batch_index',        zeros(0,1,'uint32'), ...
    'sequence_number',    zeros(0,1,'uint32'));
end

function s = emptyBatchHeader()
s = struct( ...
    'batch_magic',         '', ...
    'batch_header_size',   uint16(0), ...
    'timestamp_utc_ticks', int64(0), ...
    'reserved1',           uint16(0), ...
    'blob_count',          uint16(0), ...
    'sequence_number',     uint32(0), ...
    'payload_size_bytes',  uint32(0), ...
    'reserved2',           uint32(0));
end

function printHeaderSummary(FH, BH)
fprintf('--- Dopplium Blobs Data ---\n');
fprintf('Magic=%s  Version=%d  Endianness=%s  MessageType=%d\n', ...
    FH.magic, FH.version, tern(FH.endianness==1,'LE','BE'), FH.message_type);
fprintf('FileHdr=%d  BodyHdr=%d  BatchHdr=%d  TotalBatchesWritten=%d\n', ...
    FH.file_header_size, BH.body_header_size, BH.batch_header_size, FH.total_frames_written);
fprintf('NodeId="%s"\n', FH.node_id);

fprintf('\n-- Blobs Configuration --\n');
fprintf('Blob record size: %d bytes\n', BH.blob_record_size);
fprintf('Algorithm ID: %d\n', BH.algorithm_id);
fprintf('Algorithm version: %d\n', BH.algorithm_version);
fprintf('Body header version: %d\n', BH.body_header_version);
end
