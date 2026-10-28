function slots = splitPatternSlots(frame, txMap)
% SPLITPATTERNSLOTS Split one multiplexing_mode 2 ADC frame into its pattern slots
%   slots = splitPatternSlots(frame, txMap)
%
%   INPUTS
%     frame : one frame shaped [samples, chirps, receivers] with the chirps
%             in transmission order, e.g. data(:,:,:,i) from doppliumParser
%     txMap : that frame's transmitter pattern, headers.frame(i).tx_map
%
%   OUTPUTS
%     slots : [samples, chirps/L, L, receivers], L = numel(txMap).
%             slots(:, m, k, :) is chirp c = (m-1)*L + k (1-based), sent by
%             transmitter txMap(k). The chirps of one slot are L*PRI apart,
%             so Doppler processing runs along dimension 2 of each slot;
%             slot k starts (k-1)*PRI after slot 1.

L = numel(txMap);
if L == 0
    error('dopplium:noTxMap', ...
        'Frame carries no transmitter information (empty tx_map), so it has no pattern slots.');
end
if ndims(frame) > 3
    error('frame must be shaped [samples, chirps, receivers].');
end
[nSamples, nChirps, nReceivers] = size(frame);
if mod(nChirps, L) ~= 0
    error('dopplium:txMapLength', 'tx_map length %d does not divide %d chirps.', L, nChirps);
end

% Chirp c = (m-1)*L + k: k runs fastest, so split the chirp dimension as [L, chirps/L].
slots = reshape(frame, nSamples, L, nChirps / L, nReceivers);
slots = permute(slots, [1, 3, 2, 4]);
end
