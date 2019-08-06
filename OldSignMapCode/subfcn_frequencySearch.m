function [phaseSearch] = subfcn_frequencySearch(target, target_dat)
%% Subfunction: Frequency Searcher
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written 12Apr2017 KS
% Last Updated: 
% 30Apr2017 KS - Added header information


% Creates individual phase maps for each chosen bin, then stores them in a
% cell array for future use

%%% Necessary Subfunctions %%%
% None

%%% Inputs %%%
% target                Which frequency bins to examine (usually 1:10)
% target_dat            Which phase data (azimuth or altitude)

%%% Outputs %%%
%phaseSearch            Cell array of collected phase maps for chosen frequency bins
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phaseSearch = cell(length(target),1);

for l = 1:length(target)
    k = target(l);
            phaseSearch{l}= target_dat(:,:,k);
    phaseSearch{l} = rot90(phaseSearch{l});
    phaseSearch{l} = phaseSearch{l} - mean(phaseSearch{l}(:));
end
