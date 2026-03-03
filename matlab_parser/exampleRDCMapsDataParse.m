% Example: Parse and visualize RDC maps data with physical axes
% This example demonstrates parsing RDCMaps and plotting range-doppler maps.

% Select input file using file explorer dialog.
[fileName, filePath] = uigetfile( ...
    {'*.bin;*.dat', 'Radar files (*.bin, *.dat)'; '*.*', 'All files (*.*)'}, ...
    'Select Dopplium RDCMaps file');

if isequal(fileName, 0)
    fprintf('No file selected. Exiting example.\n');
    return;
end

filename = fullfile(filePath, fileName);
fprintf('Selected file: %s\n', filename);
[data, hdr] = doppliumParser(filename);

% Data shape: [range_bins, doppler_bins, channels, cpis]
nRange = size(data, 1);
nDopp  = size(data, 2);
nChan  = size(data, 3);
nCpi   = size(data, 4);

fprintf('Parsed data shape: [range_bins=%d, doppler_bins=%d, channels=%d, cpis=%d]\n', ...
    nRange, nDopp, nChan, nCpi);

if nCpi == 0
    error('exampleRDCMapsDataParse:NoCPIData', 'No CPI data found in file.');
end

% -------------------------------------------------------------------------
% Compute physical axes from RDC body header
% -------------------------------------------------------------------------
rangeMin = double(hdr.body.range_min_m);
rangeMax = double(hdr.body.range_max_m);
rangeRes = double(hdr.body.range_resolution_m);

velMin = double(hdr.body.velocity_min_mps);
velMax = double(hdr.body.velocity_max_mps);
velRes = double(hdr.body.velocity_resolution_mps);

if isfinite(rangeMin) && isfinite(rangeMax) && rangeMax > rangeMin
    rangeAxis = linspace(rangeMin, rangeMax, nRange);
else
    rangeAxis = (0:nRange-1) * rangeRes;
end

if isfinite(velMin) && isfinite(velMax) && velMax > velMin
    velocityAxis = linspace(velMin, velMax, nDopp);
else
    velocityAxis = ((-nDopp/2):(nDopp/2-1)) * velRes;
end

fprintf('\n--- RDC Maps Parameters ---\n');
fprintf('Range limits: [%.2f, %.2f] m (resolution %.4f m)\n', rangeMin, rangeMax, rangeRes);
fprintf('Velocity limits: [%.2f, %.2f] m/s (resolution %.4f m/s)\n', velMin, velMax, velRes);
if hdr.body.is_db_scale ~= 0
    fprintf('Stored scale: dB\n');
else
    fprintf('Stored scale: linear\n');
end
fprintf('FFT shift flags: range=%d, doppler=%d\n', hdr.body.fftshift_range, hdr.body.fftshift_doppler);

% -------------------------------------------------------------------------
% Select data to visualize
% -------------------------------------------------------------------------
cpiIdx = 1; % Select CPI index
chIdx  = 1; % Select channel index

rdMap = data(:, :, chIdx, cpiIdx); % [range_bins, doppler_bins]

% If Doppler is not pre-shifted, shift for centered velocity visualization.
if hdr.body.fftshift_doppler == 0
    rdMap = fftshift(rdMap, 2);
end

% Convert to dB for plotting when needed.
if hdr.body.is_db_scale ~= 0 && isreal(rdMap)
    rdMapDb = double(rdMap);
else
    rdMapDb = mag2db(abs(rdMap) + eps);
end

% -------------------------------------------------------------------------
% Plot: selected channel/CPI map
% -------------------------------------------------------------------------
figure;
imagesc(velocityAxis, rangeAxis, rdMapDb);
axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
colorbar;
title(sprintf('RDC Map (CPI %d, channel %d)', cpiIdx, chIdx));
grid on;

% -------------------------------------------------------------------------
% Plot: mean over channels for the same CPI (quick overview)
% -------------------------------------------------------------------------
rdMapMean = mean(data(:, :, :, cpiIdx), 3);
if hdr.body.fftshift_doppler == 0
    rdMapMean = fftshift(rdMapMean, 2);
end
if hdr.body.is_db_scale ~= 0 && isreal(rdMapMean)
    rdMapMeanDb = double(rdMapMean);
else
    rdMapMeanDb = mag2db(abs(rdMapMean) + eps);
end

figure;
imagesc(velocityAxis, rangeAxis, rdMapMeanDb);
axis xy;
xlabel('Velocity (m/s)');
ylabel('Range (m)');
colorbar;
title(sprintf('RDC Map Mean Across Channels (CPI %d)', cpiIdx));
grid on;
