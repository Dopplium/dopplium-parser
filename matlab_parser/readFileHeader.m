function FH = readFileHeader(fid, machinefmt)
% READFILEHEADER Read Dopplium file header (v2/v3/v4/v5 base + optional extension)
%   FH = readFileHeader(fid, machinefmt)
%
%   INPUTS
%     fid        : file identifier from fopen
%     machinefmt : 'ieee-le' or 'ieee-be'
%
%   OUTPUTS
%     FH : struct containing file header fields
%          .magic, .version, .endianness, .compression, .product_id,
%          .message_type, .file_header_size, .body_header_size,
%          .payload_header_size/.frame_header_size, .file_created_utc_ticks,
%          .last_written_utc_ticks, .total_frames_written,
%          .total_payload_bytes, .reserved1, .node_id
%          plus v4+ location extension when present.
%          In version >= 5, total_payload_bytes is uint64 and reserved1 is not serialized.

    headerStart = ftell(fid);
    FH.magic                   = char(fread(fid, [1,4], '*char'));
    FH.version                 = fread(fid, 1, 'uint16', 0, machinefmt);
    FH.endianness              = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.compression             = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.product_id              = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.message_type            = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.file_header_size        = fread(fid, 1, 'uint16', 0, machinefmt);
    FH.body_header_size        = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.payload_header_size     = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.frame_header_size       = FH.payload_header_size; % backward-compatible alias
    FH.file_created_utc_ticks  = fread(fid, 1, 'int64',  0, machinefmt);
    FH.last_written_utc_ticks  = fread(fid, 1, 'int64',  0, machinefmt);
    FH.total_frames_written    = fread(fid, 1, 'uint32', 0, machinefmt);
    if double(FH.version) >= 5
        FH.total_payload_bytes = fread(fid, 1, 'uint64', 0, machinefmt);
        FH.reserved1           = uint32(0);
    else
        FH.total_payload_bytes = fread(fid, 1, 'uint32', 0, machinefmt);
        FH.reserved1           = fread(fid, 1, 'uint32', 0, machinefmt);
    end
    nodeBytes                  = fread(fid, 32, '*uint8', 0, machinefmt);
    nodeChars                  = char(nodeBytes(:)');
    nullPos                    = find(nodeBytes == 0, 1, 'first');
    if isempty(nullPos)
        FH.node_id = deblank(nodeChars);
    else
        FH.node_id = nodeChars(1:nullPos-1);
    end

    % v4+ optional location extension (bytes 80..127 in current spec).
    FH.latitude_e7  = [];
    FH.longitude_e7 = [];
    FH.height_cm    = [];
    FH.pitch_deg    = [];
    FH.yaw_deg      = [];
    FH.roll_deg     = [];
    FH.reserved_ext = uint8([]);
    FH.header_extension = uint8([]);

    if double(FH.file_header_size) >= 128
        FH.latitude_e7  = fread(fid, 1, 'int32', 0, machinefmt);
        FH.longitude_e7 = fread(fid, 1, 'int32', 0, machinefmt);
        FH.height_cm    = fread(fid, 1, 'int32', 0, machinefmt);
        FH.pitch_deg    = fread(fid, 1, 'int16', 0, machinefmt);
        FH.yaw_deg      = fread(fid, 1, 'int16', 0, machinefmt);
        FH.roll_deg     = fread(fid, 1, 'int16', 0, machinefmt);
        FH.reserved_ext = fread(fid, 30, '*uint8', 0, machinefmt);
    elseif double(FH.file_header_size) > 80
        FH.header_extension = fread(fid, double(FH.file_header_size) - 80, '*uint8', 0, machinefmt);
    end

    % Keep caller file position aligned to file_header_size for forward compatibility.
    targetPos = headerStart + double(FH.file_header_size);
    if ftell(fid) < targetPos
        fseek(fid, targetPos, 'bof');
    end
end
