function [phaseMap] = subfcn_phaseMapper(FTdat,k)
%% Subfunction: Phase Mapper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written 12Apr2017 KS
% Last Updated: 
% 30Aug2017 KS - Added header info

% Lightly processes phase map data and plots it

%%% Necessary Subfunctions %%%
% None

%%% Inputs %%%
% FTdat                              Azimuth or altitude phase data
% k                                  Chosen frequency bin of interest

%%% Outputs %%%
% phaseMap                           Processed phase map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phaseMap = FTdat(:,:,k);
        phaseMap = rot90(phaseMap);
     phaseMap = phaseMap - mean(phaseMap(:));
        
        
        
        %% from 
        
       