function [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts)
% PARSETRACKS Parse Dopplium Tracks data (v3/v4/v5 message_type=6)
%   [data, headers] = parseTracks(fid, FH, machinefmt, filename, opts)
%
%   Returns:
%     data    : struct of column vectors (fields aligned with Python parser)
%     headers : struct with fields .file, .body, .frame

if FH.version == 2
    error('parseTracks:Version2NotImplemented', ...
        'Version 2 Tracks are not implemented in this parser.');
end
if ~(FH.version == 3 || FH.version == 4 || FH.version == 5) || FH.message_type ~= 6
    error('parseTracks:InvalidMessageType', ...
        ['This file is not Tracks (version=%d, message_type=%d, ' ...
         'expected version=3/4/5, type=6).'], ...
        FH.version, FH.message_type);
end

if nargin < 6 || isempty(opts)
    opts = struct;
end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'maxFrames'), opts.maxFrames = Inf; end
maxFrames = opts.maxFrames;

% Read body header
fseek(fid, FH.file_header_size, 'bof');
BH = readTracksBodyHeader(fid, machinefmt);

if opts.verbose
    printHeaderSummary(FH, BH);
end

fileInfo = dir(filename);
bytesAfterHeaders = double(fileInfo.bytes) - double(FH.file_header_size) - double(BH.body_header_size);
avgFrameSize = double(BH.frame_header_size) + (10 * double(BH.track_record_size));
if avgFrameSize <= 0
    nFramesEstimate = 1;
else
    nFramesEstimate = max(1, floor(bytesAfterHeaders / avgFrameSize));
end

if opts.verbose
    fprintf('\nBytes after headers: %d\n', bytesAfterHeaders);
    fprintf('Estimated frames in file: ~%d\n', nFramesEstimate);
end

% Read frames
fseek(fid, FH.file_header_size + BH.body_header_size, 'bof');
frameHeaders = repmat(emptyFrameHeader(), 0, 1);
chunks = {};
framesRead = 0;

while true
    if framesRead >= maxFrames
        break;
    end
    if ftell(fid) >= fileInfo.bytes
        break;
    end

    try
        FHdr = readFrameHeader(fid, machinefmt);
        frameHeaders(end+1,1) = FHdr; %#ok<AGROW>

        if opts.verbose && (framesRead == 0 || mod(framesRead + 1, 100) == 0)
            fprintf('  Reading frame %d: seq=%d, tracks=%d, size=%d bytes\n', ...
                framesRead + 1, FHdr.sequence_number, FHdr.num_tracks, FHdr.payload_size_bytes);
        end

        expectedPayloadSize = double(FHdr.num_tracks) * double(BH.track_record_size);
        if double(FHdr.payload_size_bytes) ~= expectedPayloadSize
            if opts.verbose
                warning('parseTracks:PayloadSizeMismatch', ...
                    'Frame %d payload size mismatch: expected=%d, got=%d', ...
                    framesRead, expectedPayloadSize, FHdr.payload_size_bytes);
            end
        end

        % Honor frame_header_size for forward compatibility.
        extraHeaderBytes = double(FHdr.frame_header_size) - 30;
        if extraHeaderBytes < 0
            error('parseTracks:InvalidHeaderSize', ...
                'Frame %d has invalid frame_header_size=%d (<30).', ...
                framesRead + 1, FHdr.frame_header_size);
        elseif extraHeaderBytes > 0
            fseek(fid, extraHeaderBytes, 'cof');
        end

        if FHdr.num_tracks > 0
            chunk = readTrackRecords(fid, double(FHdr.num_tracks), machinefmt, uint32(framesRead));
            chunks{end+1} = chunk; %#ok<AGROW>
        end

        framesRead = framesRead + 1;
    catch ME
        if strcmp(ME.identifier, 'parseTracks:UnexpectedEOF')
            if opts.verbose
                fprintf('Reached end of file after reading %d frames.\n', framesRead);
            end
            break;
        end
        rethrow(ME);
    end
end

if isempty(chunks)
    data = emptyTracks();
else
    data = concatTrackChunks(chunks);
end

headers.file  = FH;
headers.body  = BH;
headers.frame = frameHeaders;

if opts.verbose
    fprintf('\nTotal frames read: %d\n', framesRead);
    fprintf('Total tracks: %d\n', numel(data.track_id));
end
end

function BH = readTracksBodyHeader(fid, machinefmt)
BH.body_magic                          = char(fread(fid, [1,4], '*char'));
BH.body_header_version                 = fread(fid, 1, 'uint16', 0, machinefmt);
BH.body_header_size                    = fread(fid, 1, 'uint16', 0, machinefmt);
BH.frame_header_size                   = fread(fid, 1, 'uint16', 0, machinefmt);
BH.track_record_size                   = fread(fid, 1, 'uint16', 0, machinefmt);
BH.association_algorithm_id            = fread(fid, 1, 'uint32', 0, machinefmt);
BH.association_algorithm_version       = fread(fid, 1, 'uint32', 0, machinefmt);
BH.tracker_algorithm_id                = fread(fid, 1, 'uint32', 0, machinefmt);
BH.tracker_algorithm_version           = fread(fid, 1, 'uint32', 0, machinefmt);
BH.track_management_algorithm_id       = fread(fid, 1, 'uint32', 0, machinefmt);
BH.track_management_algorithm_version  = fread(fid, 1, 'uint32', 0, machinefmt);
BH.association_params                  = fread(fid, 20, '*uint8', 0, machinefmt);
BH.tracker_params                      = fread(fid, 20, '*uint8', 0, machinefmt);
BH.track_management_params             = fread(fid, 20, '*uint8', 0, machinefmt);

if ~strcmp(BH.body_magic, 'TRCK')
    warning('parseTracks:MagicMismatch', ...
        'Unexpected tracks body magic "%s" (expected "TRCK").', BH.body_magic);
end
end

function FHdr = readFrameHeader(fid, machinefmt)
FHdr.frame_magic           = char(fread(fid, [1,4], '*char'));
FHdr.frame_header_size     = fread(fid, 1, 'uint16', 0, machinefmt);
FHdr.timestamp_utc_ticks   = fread(fid, 1, 'int64',  0, machinefmt);
FHdr.num_tracks            = fread(fid, 1, 'uint16', 0, machinefmt);
FHdr.num_new_tracks        = fread(fid, 1, 'uint16', 0, machinefmt);
FHdr.num_terminated_tracks = fread(fid, 1, 'uint16', 0, machinefmt);
FHdr.sequence_number       = fread(fid, 1, 'uint32', 0, machinefmt);
FHdr.payload_size_bytes    = fread(fid, 1, 'uint32', 0, machinefmt);
FHdr.flags                 = fread(fid, 1, 'uint16', 0, machinefmt);

if isempty(FHdr.flags)
    error('parseTracks:UnexpectedEOF', 'Failed to read tracks frame header.');
end
if ~strcmp(FHdr.frame_magic, 'FRME')
    error('parseTracks:InvalidFrameMagic', ...
        'Invalid frame magic "%s" (expected "FRME").', FHdr.frame_magic);
end
end

function chunk = readTrackRecords(fid, count, machinefmt, frameIndex)
chunk = emptyTracks();

chunk.track_id                     = zeros(count,1,'uint32');
chunk.sequence_number              = zeros(count,1,'uint32');
chunk.status                       = zeros(count,1,'uint8');
chunk.associated_detection_count   = zeros(count,1,'uint8');
chunk.frames_since_detection       = zeros(count,1,'uint16');
chunk.target_class_id              = zeros(count,1,'uint16');
chunk.track_lifetime_seconds       = zeros(count,1,'single');
chunk.birth_timestamp_utc_ticks    = zeros(count,1,'int64');
chunk.gap_count                    = zeros(count,1,'uint16');

chunk.cart_x                       = zeros(count,1,'single');
chunk.cart_x_std                   = zeros(count,1,'single');
chunk.cart_y                       = zeros(count,1,'single');
chunk.cart_y_std                   = zeros(count,1,'single');
chunk.cart_z                       = zeros(count,1,'single');
chunk.cart_z_std                   = zeros(count,1,'single');
chunk.cart_vx                      = zeros(count,1,'single');
chunk.cart_vx_std                  = zeros(count,1,'single');
chunk.cart_vy                      = zeros(count,1,'single');
chunk.cart_vy_std                  = zeros(count,1,'single');
chunk.cart_vz                      = zeros(count,1,'single');
chunk.cart_vz_std                  = zeros(count,1,'single');

chunk.enu_east                     = zeros(count,1,'single');
chunk.enu_east_std                 = zeros(count,1,'single');
chunk.enu_north                    = zeros(count,1,'single');
chunk.enu_north_std                = zeros(count,1,'single');
chunk.enu_up                       = zeros(count,1,'single');
chunk.enu_up_std                   = zeros(count,1,'single');
chunk.enu_ve                       = zeros(count,1,'single');
chunk.enu_ve_std                   = zeros(count,1,'single');
chunk.enu_vn                       = zeros(count,1,'single');
chunk.enu_vn_std                   = zeros(count,1,'single');
chunk.enu_vu                       = zeros(count,1,'single');
chunk.enu_vu_std                   = zeros(count,1,'single');

chunk.blob_size_range              = zeros(count,1,'single');
chunk.blob_size_azimuth            = zeros(count,1,'single');
chunk.blob_size_elevation          = zeros(count,1,'single');
chunk.blob_size_doppler            = zeros(count,1,'single');
chunk.num_detections_in_blob       = zeros(count,1,'uint16');

chunk.amplitude_db                 = zeros(count,1,'single');
chunk.snr_db                       = zeros(count,1,'single');
chunk.confidence_score             = zeros(count,1,'single');

chunk.frame_index                  = repmat(frameIndex, count, 1);

for i = 1:count
    % Identity & status (30 bytes)
    chunk.track_id(i)                   = readScalar(fid, 'uint32', machinefmt, 'track_id');
    chunk.sequence_number(i)            = readScalar(fid, 'uint32', machinefmt, 'sequence_number');
    chunk.status(i)                     = readScalar(fid, 'uint8',  machinefmt, 'status');
    chunk.associated_detection_count(i) = readScalar(fid, 'uint8',  machinefmt, 'associated_detection_count');
    chunk.frames_since_detection(i)     = readScalar(fid, 'uint16', machinefmt, 'frames_since_detection');
    chunk.target_class_id(i)            = readScalar(fid, 'uint16', machinefmt, 'target_class_id');
    chunk.track_lifetime_seconds(i)     = readScalar(fid, 'single', machinefmt, 'track_lifetime_seconds');
    chunk.birth_timestamp_utc_ticks(i)  = readScalar(fid, 'int64',  machinefmt, 'birth_timestamp_utc_ticks');
    chunk.gap_count(i)                  = readScalar(fid, 'uint16', machinefmt, 'gap_count');
    assertPadding(fid, 2, machinefmt, 'identity padding');

    % Sensor Cartesian (48 bytes)
    cart = fread(fid, 12, '*single', 0, machinefmt);
    if numel(cart) ~= 12
        error('parseTracks:UnexpectedEOF', 'Unexpected EOF while reading sensor Cartesian fields.');
    end
    chunk.cart_x(i)     = cart(1);  chunk.cart_x_std(i) = cart(2);
    chunk.cart_y(i)     = cart(3);  chunk.cart_y_std(i) = cart(4);
    chunk.cart_z(i)     = cart(5);  chunk.cart_z_std(i) = cart(6);
    chunk.cart_vx(i)    = cart(7);  chunk.cart_vx_std(i)= cart(8);
    chunk.cart_vy(i)    = cart(9);  chunk.cart_vy_std(i)= cart(10);
    chunk.cart_vz(i)    = cart(11); chunk.cart_vz_std(i)= cart(12);

    % ENU (48 bytes)
    enu = fread(fid, 12, '*single', 0, machinefmt);
    if numel(enu) ~= 12
        error('parseTracks:UnexpectedEOF', 'Unexpected EOF while reading ENU fields.');
    end
    chunk.enu_east(i)    = enu(1);  chunk.enu_east_std(i) = enu(2);
    chunk.enu_north(i)   = enu(3);  chunk.enu_north_std(i)= enu(4);
    chunk.enu_up(i)      = enu(5);  chunk.enu_up_std(i)   = enu(6);
    chunk.enu_ve(i)      = enu(7);  chunk.enu_ve_std(i)   = enu(8);
    chunk.enu_vn(i)      = enu(9);  chunk.enu_vn_std(i)   = enu(10);
    chunk.enu_vu(i)      = enu(11); chunk.enu_vu_std(i)   = enu(12);

    % Blob info (20 bytes)
    blob = fread(fid, 4, '*single', 0, machinefmt);
    if numel(blob) ~= 4
        error('parseTracks:UnexpectedEOF', 'Unexpected EOF while reading blob size fields.');
    end
    chunk.blob_size_range(i)     = blob(1);
    chunk.blob_size_azimuth(i)   = blob(2);
    chunk.blob_size_elevation(i) = blob(3);
    chunk.blob_size_doppler(i)   = blob(4);
    chunk.num_detections_in_blob(i) = readScalar(fid, 'uint16', machinefmt, 'num_detections_in_blob');
    assertPadding(fid, 2, machinefmt, 'blob padding');

    % Quality (12 bytes)
    quality = fread(fid, 3, '*single', 0, machinefmt);
    if numel(quality) ~= 3
        error('parseTracks:UnexpectedEOF', 'Unexpected EOF while reading quality fields.');
    end
    chunk.amplitude_db(i)    = quality(1);
    chunk.snr_db(i)          = quality(2);
    chunk.confidence_score(i)= quality(3);

    % Final record padding (2 bytes)
    assertPadding(fid, 2, machinefmt, 'final record padding');
end
end

function v = readScalar(fid, typeName, machinefmt, fieldName)
v = fread(fid, 1, ['*' typeName], 0, machinefmt);
if isempty(v)
    error('parseTracks:UnexpectedEOF', ...
        'Unexpected EOF while reading track field "%s".', fieldName);
end
end

function assertPadding(fid, nBytes, machinefmt, paddingName)
pad = fread(fid, nBytes, '*uint8', 0, machinefmt);
if numel(pad) ~= nBytes
    error('parseTracks:UnexpectedEOF', ...
        'Unexpected EOF while reading %s.', paddingName);
end
end

function out = concatTrackChunks(chunks)
out = emptyTracks();
out.track_id                   = vertcatField(chunks, 'track_id');
out.sequence_number            = vertcatField(chunks, 'sequence_number');
out.status                     = vertcatField(chunks, 'status');
out.associated_detection_count = vertcatField(chunks, 'associated_detection_count');
out.frames_since_detection     = vertcatField(chunks, 'frames_since_detection');
out.target_class_id            = vertcatField(chunks, 'target_class_id');
out.track_lifetime_seconds     = vertcatField(chunks, 'track_lifetime_seconds');
out.birth_timestamp_utc_ticks  = vertcatField(chunks, 'birth_timestamp_utc_ticks');
out.gap_count                  = vertcatField(chunks, 'gap_count');

out.cart_x                     = vertcatField(chunks, 'cart_x');
out.cart_x_std                 = vertcatField(chunks, 'cart_x_std');
out.cart_y                     = vertcatField(chunks, 'cart_y');
out.cart_y_std                 = vertcatField(chunks, 'cart_y_std');
out.cart_z                     = vertcatField(chunks, 'cart_z');
out.cart_z_std                 = vertcatField(chunks, 'cart_z_std');
out.cart_vx                    = vertcatField(chunks, 'cart_vx');
out.cart_vx_std                = vertcatField(chunks, 'cart_vx_std');
out.cart_vy                    = vertcatField(chunks, 'cart_vy');
out.cart_vy_std                = vertcatField(chunks, 'cart_vy_std');
out.cart_vz                    = vertcatField(chunks, 'cart_vz');
out.cart_vz_std                = vertcatField(chunks, 'cart_vz_std');

out.enu_east                   = vertcatField(chunks, 'enu_east');
out.enu_east_std               = vertcatField(chunks, 'enu_east_std');
out.enu_north                  = vertcatField(chunks, 'enu_north');
out.enu_north_std              = vertcatField(chunks, 'enu_north_std');
out.enu_up                     = vertcatField(chunks, 'enu_up');
out.enu_up_std                 = vertcatField(chunks, 'enu_up_std');
out.enu_ve                     = vertcatField(chunks, 'enu_ve');
out.enu_ve_std                 = vertcatField(chunks, 'enu_ve_std');
out.enu_vn                     = vertcatField(chunks, 'enu_vn');
out.enu_vn_std                 = vertcatField(chunks, 'enu_vn_std');
out.enu_vu                     = vertcatField(chunks, 'enu_vu');
out.enu_vu_std                 = vertcatField(chunks, 'enu_vu_std');

out.blob_size_range            = vertcatField(chunks, 'blob_size_range');
out.blob_size_azimuth          = vertcatField(chunks, 'blob_size_azimuth');
out.blob_size_elevation        = vertcatField(chunks, 'blob_size_elevation');
out.blob_size_doppler          = vertcatField(chunks, 'blob_size_doppler');
out.num_detections_in_blob     = vertcatField(chunks, 'num_detections_in_blob');

out.amplitude_db               = vertcatField(chunks, 'amplitude_db');
out.snr_db                     = vertcatField(chunks, 'snr_db');
out.confidence_score           = vertcatField(chunks, 'confidence_score');

out.frame_index                = vertcatField(chunks, 'frame_index');
end

function v = vertcatField(chunks, fieldName)
vals = cellfun(@(c) c.(fieldName), chunks, 'UniformOutput', false);
v = vertcat(vals{:});
end

function d = emptyTracks()
d = struct( ...
    'track_id',                   zeros(0,1,'uint32'), ...
    'sequence_number',            zeros(0,1,'uint32'), ...
    'status',                     zeros(0,1,'uint8'), ...
    'associated_detection_count', zeros(0,1,'uint8'), ...
    'frames_since_detection',     zeros(0,1,'uint16'), ...
    'target_class_id',            zeros(0,1,'uint16'), ...
    'track_lifetime_seconds',     zeros(0,1,'single'), ...
    'birth_timestamp_utc_ticks',  zeros(0,1,'int64'), ...
    'gap_count',                  zeros(0,1,'uint16'), ...
    'cart_x',                     zeros(0,1,'single'), ...
    'cart_x_std',                 zeros(0,1,'single'), ...
    'cart_y',                     zeros(0,1,'single'), ...
    'cart_y_std',                 zeros(0,1,'single'), ...
    'cart_z',                     zeros(0,1,'single'), ...
    'cart_z_std',                 zeros(0,1,'single'), ...
    'cart_vx',                    zeros(0,1,'single'), ...
    'cart_vx_std',                zeros(0,1,'single'), ...
    'cart_vy',                    zeros(0,1,'single'), ...
    'cart_vy_std',                zeros(0,1,'single'), ...
    'cart_vz',                    zeros(0,1,'single'), ...
    'cart_vz_std',                zeros(0,1,'single'), ...
    'enu_east',                   zeros(0,1,'single'), ...
    'enu_east_std',               zeros(0,1,'single'), ...
    'enu_north',                  zeros(0,1,'single'), ...
    'enu_north_std',              zeros(0,1,'single'), ...
    'enu_up',                     zeros(0,1,'single'), ...
    'enu_up_std',                 zeros(0,1,'single'), ...
    'enu_ve',                     zeros(0,1,'single'), ...
    'enu_ve_std',                 zeros(0,1,'single'), ...
    'enu_vn',                     zeros(0,1,'single'), ...
    'enu_vn_std',                 zeros(0,1,'single'), ...
    'enu_vu',                     zeros(0,1,'single'), ...
    'enu_vu_std',                 zeros(0,1,'single'), ...
    'blob_size_range',            zeros(0,1,'single'), ...
    'blob_size_azimuth',          zeros(0,1,'single'), ...
    'blob_size_elevation',        zeros(0,1,'single'), ...
    'blob_size_doppler',          zeros(0,1,'single'), ...
    'num_detections_in_blob',     zeros(0,1,'uint16'), ...
    'amplitude_db',               zeros(0,1,'single'), ...
    'snr_db',                     zeros(0,1,'single'), ...
    'confidence_score',           zeros(0,1,'single'), ...
    'frame_index',                zeros(0,1,'uint32'));
end

function s = emptyFrameHeader()
s = struct( ...
    'frame_magic',           '', ...
    'frame_header_size',     uint16(0), ...
    'timestamp_utc_ticks',   int64(0), ...
    'num_tracks',            uint16(0), ...
    'num_new_tracks',        uint16(0), ...
    'num_terminated_tracks', uint16(0), ...
    'sequence_number',       uint32(0), ...
    'payload_size_bytes',    uint32(0), ...
    'flags',                 uint16(0));
end

function printHeaderSummary(FH, BH)
fprintf('--- Dopplium Tracks Data ---\n');
fprintf('Magic=%s  Version=%d  Endianness=%s  MessageType=%d\n', ...
    FH.magic, FH.version, tern(FH.endianness==1,'LE','BE'), FH.message_type);
fprintf('FileHdr=%d  BodyHdr=%d  FrameHdr=%d  TotalFramesWritten=%d\n', ...
    FH.file_header_size, BH.body_header_size, BH.frame_header_size, FH.total_frames_written);
fprintf('NodeId="%s"\n', FH.node_id);

fprintf('\n-- Tracking Configuration --\n');
fprintf('Track record size: %d bytes\n', BH.track_record_size);
fprintf('Association algorithm: id=%d version=%d\n', ...
    BH.association_algorithm_id, BH.association_algorithm_version);
fprintf('Tracker algorithm: id=%d version=%d\n', ...
    BH.tracker_algorithm_id, BH.tracker_algorithm_version);
fprintf('Track management algorithm: id=%d version=%d\n', ...
    BH.track_management_algorithm_id, BH.track_management_algorithm_version);
fprintf('Body header version: %d\n', BH.body_header_version);
end
