function tests = testAdcTxMap
% TESTADCTXMAP DOP-BIN 2.0 ADC reading (file version 6, multiplexing_mode 2,
% per-frame tx_map) and the legacy behaviour it must keep.
%
% Run from the repository root:
%   results = runtests('tests/matlab'); assertSuccess(results);
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
parserDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab_parser');
testCase.applyFixture(matlab.unittest.fixtures.PathFixture(parserDir));
testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(parserDir, 'utils')));
testCase.TestData.TDMA = [1 2 0 0];
testCase.TestData.TX1 = [1 0 0 0];
testCase.TestData.TX2 = [2 0 0 0];
testCase.TestData.TX1122 = [1 1 2 2];
testCase.TestData.NOTX = [0 0 0 0];
testCase.TestData.INVALID = [1 0 2 0];
end

function setup(testCase)
tmp = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
testCase.TestData.dir = tmp.Folder;
end

% ------------------------------------------------------------------------
% Mode 2
% ------------------------------------------------------------------------

function testMixedPatternsDecodeInTransmissionOrder(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'mixed.dop');
maps = {T.TDMA, T.TX1, T.TDMA, T.TX2, T.TX1122, T.NOTX};
expected = writeAdcFile(path, maps);
[data, headers] = parseQuiet(path);

verifyEqual(testCase, double(headers.file.version), 6);
verifyEqual(testCase, double(headers.file.payload_header_size), 32);
verifyEqual(testCase, double(headers.body.config_version), 4);
verifyEqual(testCase, double(headers.body.multiplexing_mode), 2);
verifyEqual(testCase, size(data), [8 8 4 6]);
for f = 1:numel(maps)
    verifyEqual(testCase, data(:, :, :, f), single(expected{f}));
end
verifyEqual(testCase, {headers.frame.tx_map}, {[1 2], 1, [1 2], 2, [1 1 2 2], zeros(1, 0)});
verifyEqual(testCase, [headers.frame.frame_number], 100:105);
end

function testVerboseSummaryListsPatterns(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'verbose.dop');
writeAdcFile(path, {T.TDMA, T.TX1, T.TDMA, T.NOTX});
output = evalc('doppliumParser(path);');
verifySubstring(testCase, output, 'MultiplexingMode=TDM, per-frame tx_map (2)');
verifySubstring(testCase, output, 'Parsed data shape: [samples=8, chirps=8, receivers=4, frames=4]');
verifySubstring(testCase, output, 'tx_map patterns (frames): [1 2] x2, [1] x1, [] x1');
end

function testStartFrameUsesThirtyTwoByteStride(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'stride.dop');
writeAdcFile(path, {T.TDMA, T.TX1, T.TDMA, T.TX2, T.TX1122, T.NOTX});
full = parseQuiet(path);
[part, headers] = parseQuiet(path, struct('startFrame', 3, 'maxFrames', 2));
verifyEqual(testCase, part, full(:, :, :, 4:5));
verifyEqual(testCase, {headers.frame.tx_map}, {2, [1 1 2 2]});
verifyEqual(testCase, [headers.frame.frame_number], [103 104]);
end

function testSplitPatternSlots(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'slots.dop');
writeAdcFile(path, {T.TDMA, T.TX1, T.TX2, T.TX1122, T.NOTX});
[data, headers] = parseQuiet(path);
nC = size(data, 2);
for f = 1:numel(headers.frame)
    txMap = headers.frame(f).tx_map;
    if isempty(txMap)
        verifyError(testCase, @() splitPatternSlots(data(:, :, :, f), txMap), 'dopplium:noTxMap');
        continue;
    end
    L = numel(txMap);
    slots = splitPatternSlots(data(:, :, :, f), txMap);
    verifyEqual(testCase, size(slots), [8, nC / L, L, 4]);
    for m = 1:nC / L
        for k = 1:L
            verifyEqual(testCase, squeeze(slots(:, m, k, :)), squeeze(data(:, (m-1)*L + k, :, f)));
        end
    end
end
% TDMA frame: slot 1 holds chirps 1,3,5,... (TX1), slot 2 chirps 2,4,6,... (TX2).
tdma = splitPatternSlots(data(:, :, :, 1), headers.frame(1).tx_map);
verifyEqual(testCase, squeeze(tdma(:, :, 1, :)), squeeze(data(:, 1:2:end, :, 1)));
verifyEqual(testCase, squeeze(tdma(:, :, 2, :)), squeeze(data(:, 2:2:end, :, 1)));
verifyError(testCase, @() splitPatternSlots(zeros(8, 6, 4), [1 1 2 2]), 'dopplium:txMapLength');
end

function testInvalidTxMapFrameIsRejected(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'invalid.dop');
expected = writeAdcFile(path, {T.TDMA, T.TX1, T.TX2, T.INVALID, T.TDMA});
verifyError(testCase, @() parseQuiet(path), 'dopplium:invalidTxMap');
try
    parseQuiet(path);
catch err
    verifySubstring(testCase, err.message, 'Frame 4: invalid tx_map [1 0 2 0]');
end
% The frames around it still read, at the right offsets.
[~, headers] = parseQuiet(path, struct('maxFrames', 3));
verifyEqual(testCase, {headers.frame.tx_map}, {[1 2], 1, 2});
[after, headers] = parseQuiet(path, struct('startFrame', 4));
verifyEqual(testCase, headers.frame(1).tx_map, [1 2]);
verifyEqual(testCase, after(:, :, :, 1), single(expected{5}));
end

function testPatternLengthMustDivideChirps(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'length3.dop');
writeAdcFile(path, {T.TDMA, [1 2 1 0]});
verifyError(testCase, @() parseQuiet(path), 'dopplium:txMapLength');
end

function testModeTwoNeedsVersionSixAndConfigVersionFour(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'gate.dop');
writeAdcFile(path, {T.TDMA, T.TX1}, 'version', 5);
verifyError(testCase, @() parseQuiet(path), 'dopplium:txMapRequirements');
writeAdcFile(path, {T.TDMA, T.TX1}, 'configVersion', 3);
verifyError(testCase, @() parseQuiet(path), 'dopplium:txMapRequirements');
writeAdcFile(path, {[], []}, 'payloadHeaderSize', 24);
verifyError(testCase, @() parseQuiet(path), 'dopplium:txMapRequirements');
writeAdcFile(path, {T.TDMA}, 'nChirpsField', 4);
verifyError(testCase, @() parseQuiet(path), 'dopplium:txMapRequirements');
end

function testUnknownMultiplexingModeIsUnsupported(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'mode3.dop');
writeAdcFile(path, {T.TDMA}, 'mode', 3, 'payloadHeaderSize', 32);
verifyError(testCase, @() parseQuiet(path), 'dopplium:unsupportedMultiplexingMode');
end

function testUnknownFileVersionIsRejected(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'v7.dop');
writeAdcFile(path, {T.TDMA}, 'version', 7);
try
    parseQuiet(path);
    verifyFail(testCase, 'version 7 was accepted');
catch err
    verifySubstring(testCase, err.message, 'Unsupported file version: 7');
end
end

function testComplexAndByChirpLayouts(testCase)
T = testCase.TestData;
for sampleType = [0 1]
    for dataOrder = [0 1]
        path = fullfile(T.dir, sprintf('layout_%d_%d.dop', sampleType, dataOrder));
        expected = writeAdcFile(path, {T.TDMA, T.TX2}, 'sampleType', sampleType, 'dataOrder', dataOrder);
        [data, headers] = parseQuiet(path);
        verifyEqual(testCase, size(data), [8 8 4 2]);
        for f = 1:2
            want = single(expected{f});
            if sampleType == 1
                want = complex(want, -want);
            end
            verifyEqual(testCase, data(:, :, :, f), want);
        end
        verifyEqual(testCase, {headers.frame.tx_map}, {[1 2], 2});
    end
end
end

function testBigEndian(testCase)
T = testCase.TestData;
path = fullfile(T.dir, 'be.dop');
expected = writeAdcFile(path, {T.TX1122, T.TX1}, 'machinefmt', 'ieee-be');
[data, headers] = parseQuiet(path);
verifyEqual(testCase, data(:, :, :, 2), single(expected{2}));
verifyEqual(testCase, {headers.frame.tx_map}, {[1 1 2 2], 1});
end

% ------------------------------------------------------------------------
% Legacy files (file version <= 5, 24-byte payload headers)
% ------------------------------------------------------------------------

function testLegacyMimoVersionsFourAndFive(testCase)
for version = [4 5]
    path = fullfile(testCase.TestData.dir, sprintf('legacy_v%d.dop', version));
    expected = writeAdcFile(path, {[], [], []}, 'version', version, 'configVersion', 3, ...
        'mode', 0, 'nChirpsField', 4);
    [data, headers] = parseQuiet(path);
    verifyEqual(testCase, double(headers.file.version), version);
    verifyEqual(testCase, size(data), [8 4 8 3]);
    for f = 1:3
        for c = 1:8
            tx = mod(c - 1, 2) + 1;
            cTx = floor((c - 1) / 2) + 1;
            verifyEqual(testCase, squeeze(data(:, cTx, (tx-1)*4 + (1:4), f)), ...
                squeeze(single(expected{f}(:, c, :))));
        end
    end
    verifyTrue(testCase, all(arrayfun(@(fr) isempty(fr.tx_map), headers.frame)));
end
end

function testLegacyVersionTwoSingleTx(testCase)
path = fullfile(testCase.TestData.dir, 'legacy_v2.dop');
expected = writeAdcFile(path, {[], []}, 'version', 2, 'configVersion', 2, 'mode', 0, 'nTx', 1);
[data, headers] = parseQuiet(path);
verifyEqual(testCase, double(headers.file.version), 2);
verifyEqual(testCase, size(data), [8 8 4 2]);
verifyEqual(testCase, data(:, :, :, 2), single(expected{2}));
end

function testLegacyComplexConfigVersionThree(testCase)
path = fullfile(testCase.TestData.dir, 'legacy_cfg3_complex.dop');
expected = writeAdcFile(path, {[], []}, 'version', 5, 'configVersion', 3, 'mode', 0, ...
    'nTx', 1, 'sampleType', 1);
data = parseQuiet(path);
want = single(expected{2});
verifyEqual(testCase, data(:, :, :, 2), complex(want, -want));
end

% ------------------------------------------------------------------------
% Helpers
% ------------------------------------------------------------------------

function [data, headers] = parseQuiet(path, opts)
if nargin < 2
    opts = struct;
end
opts.verbose = false;
[data, headers] = doppliumParser(path, opts);
end

function expected = writeAdcFile(path, txMaps, varargin)
% WRITEADCFILE Write a synthetic ADCData file with fwrite.
%   txMaps: one entry per frame, a 1x4 tx_map (32-byte payload header) or []
%   (classic 24-byte header). Sample (s, c, r) of frame f (all 1-based) holds
%   (f-1)*1024 + (c-1)*32 + (r-1)*8 + (s-1). expected{f} is [samples, chirps, rx].
p = inputParser;
addParameter(p, 'version', 6);
addParameter(p, 'configVersion', 4);
addParameter(p, 'mode', 2);
addParameter(p, 'nChirps', 8);
addParameter(p, 'nChirpsField', []);
addParameter(p, 'nRx', 4);
addParameter(p, 'nSamples', 8);
addParameter(p, 'nTx', 2);
addParameter(p, 'payloadHeaderSize', []);
addParameter(p, 'sampleType', 0);
addParameter(p, 'dataOrder', 0);
addParameter(p, 'machinefmt', 'ieee-le');
parse(p, varargin{:});
o = p.Results;
if isempty(o.payloadHeaderSize)
    o.payloadHeaderSize = 24 + 8 * (o.mode == 2);
end
if isempty(o.nChirpsField)
    o.nChirpsField = o.nChirps;
end
C = o.nChirps; R = o.nRx; S = o.nSamples;
nFrames = numel(txMaps);
bytesPerSample = 2 * (1 + o.sampleType);
bytesPerFrame = C * R * S * bytesPerSample;
fileHeaderSize = 80 + 48 * (o.version >= 4);
messageType = 1 + 2 * (o.version == 2);

fid = fopen(path, 'w', o.machinefmt);
closer = onCleanup(@() fclose(fid));

% File header
fwrite(fid, 'DOPP', 'uchar');
fwrite(fid, o.version, 'uint16');
fwrite(fid, double(strcmp(o.machinefmt, 'ieee-le')), 'uint8');
fwrite(fid, [0 7 messageType], 'uint8');
fwrite(fid, fileHeaderSize, 'uint16');
fwrite(fid, [192 o.payloadHeaderSize], 'uint32');
fwrite(fid, [1000 2000], 'int64');
fwrite(fid, nFrames, 'uint32');
if o.version >= 5
    fwrite(fid, nFrames * bytesPerFrame, 'uint64');
else
    fwrite(fid, [nFrames * bytesPerFrame 0], 'uint32');
end
fwrite(fid, [uint8('test-node') zeros(1, 23, 'uint8')], 'uint8');
if fileHeaderSize >= 128
    fwrite(fid, [415000000 -37000000 1200], 'int32');
    fwrite(fid, [1 2 3], 'int16');
    fwrite(fid, zeros(1, 30), 'uint8');
end

% ADC body header (192 bytes)
fwrite(fid, 'ADCD', 'uchar');
fwrite(fid, [o.configVersion 192 o.payloadHeaderSize 0], 'uint16');
fwrite(fid, [o.payloadHeaderSize + bytesPerFrame, S, o.nChirpsField], 'uint32');
fwrite(fid, [16 R o.nTx], 'uint16');
fwrite(fid, [o.sampleType o.dataOrder 0 0 o.mode 0], 'uint8');
fwrite(fid, [77 1 5 0 6 60 10000 20 9.92], 'double');
fwrite(fid, [2 bytesPerSample bytesPerFrame / bytesPerSample bytesPerFrame], 'uint32');
fwrite(fid, [100 10 0.15 0.1], 'single');
fwrite(fid, [1:R zeros(1, 30 - R)], 'uint8');
fwrite(fid, zeros(1, 22), 'uint8');

% Frames
expected = cell(1, nFrames);
[s, c, r] = ndgrid(0:S-1, 0:C-1, 0:R-1);
for f = 1:nFrames
    A = (f - 1) * 1024 + c * 32 + r * 8 + s;      % [samples, chirps, rx]
    expected{f} = A;
    fwrite(fid, 'FRME', 'uchar');
    fwrite(fid, [0 o.payloadHeaderSize], 'uint16');
    fwrite(fid, 1000 + f, 'int64');
    fwrite(fid, [99 + f, bytesPerFrame], 'uint32');
    header = zeros(1, o.payloadHeaderSize - 24);
    if ~isempty(txMaps{f})
        header(1:4) = txMaps{f};
    end
    fwrite(fid, header, 'uint8');

    B = permute(A, [1 3 2]);                       % [samples, rx, chirps]
    if o.sampleType == 1                           % interleaved I, Q with Q = -I
        IQ = zeros(2 * S, R, C);
        IQ(1:2:end, :, :) = B;
        IQ(2:2:end, :, :) = -B;
        B = IQ;
    end
    if o.dataOrder == 1                            % chirp -> element -> rx
        B = permute(B, [2 1 3]);
    end
    fwrite(fid, B(:), 'int16');
end
end
