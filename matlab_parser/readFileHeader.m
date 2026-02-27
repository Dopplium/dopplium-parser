function FH = readFileHeader(fid, machinefmt)
% READFILEHEADER Read Dopplium file header - Versions 2/3/4 format
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
%          .frame_header_size, .file_created_utc_ticks,
%          .last_written_utc_ticks, .total_frames_written,
%          .total_payload_bytes, .reserved1, .node_id

    headerStartPos             = ftell(fid);

    FH.magic                   = char(fread(fid, [1,4], '*char'));
    FH.version                 = fread(fid, 1, 'uint16', 0, machinefmt);
    FH.endianness              = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.compression             = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.product_id              = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.message_type            = fread(fid, 1, 'uint8',  0, machinefmt);
    FH.file_header_size        = fread(fid, 1, 'uint16', 0, machinefmt);
    FH.body_header_size        = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.frame_header_size       = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.file_created_utc_ticks  = fread(fid, 1, 'int64',  0, machinefmt);
    FH.last_written_utc_ticks  = fread(fid, 1, 'int64',  0, machinefmt);
    FH.total_frames_written    = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.total_payload_bytes     = fread(fid, 1, 'uint32', 0, machinefmt);
    FH.reserved1               = fread(fid, 1, 'uint32', 0, machinefmt);
    nodeBytes                  = fread(fid, 32, '*uint8', 0, machinefmt);
    FH.node_id                 = deblank(char(nodeBytes(:)')); % null-terminated ASCII

    % v4 extension fields (present only when version>=4 and file_header_size>=128)
    FH.latitude_e7             = [];
    FH.longitude_e7            = [];
    FH.height_cm               = [];
    FH.pitch_deg               = [];
    FH.yaw_deg                 = [];
    FH.roll_deg                = [];

    if FH.version >= 4 && FH.file_header_size >= 128
        FH.latitude_e7         = fread(fid, 1, 'int32', 0, machinefmt);
        FH.longitude_e7        = fread(fid, 1, 'int32', 0, machinefmt);
        FH.height_cm           = fread(fid, 1, 'int32', 0, machinefmt);
        FH.pitch_deg           = fread(fid, 1, 'int16', 0, machinefmt);
        FH.yaw_deg             = fread(fid, 1, 'int16', 0, machinefmt);
        FH.roll_deg            = fread(fid, 1, 'int16', 0, machinefmt);

        % Reserved extension bytes (30)
        fread(fid, 30, '*uint8', 0, machinefmt);
    end

    % Ensure caller sees file pointer at the declared end of header.
    fseek(fid, headerStartPos + double(FH.file_header_size), 'bof');
end
