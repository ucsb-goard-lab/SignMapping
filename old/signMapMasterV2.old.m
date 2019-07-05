function [] = signMapMasterV2()
%% Sign Map Master File
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Written 29 Aug 2017 KS
% Last Updated:
% 30 Aug 2017 KS - Added header information
%                - Adjusted the code for better handling of multiple
%                recording blocks

% Takes DFF widefield data, extracts the retinotopic maps, then the sign
% map. Appropriately processes the sign maps, then overlays the borders
% onto the F0 map from DFF (average picture).


% This works on only a multi-page tif. If you have many single page tifs
% (as what normally comes from the scopes), run widefield_tifConvert first.


% NOTE: All further processing steps assume a framerate of 10fps. If this
% is NOT your recording framerate, change the value of fs in A_widefieldFT.


%%% Necessary Subfunctions %%
% widefieldDFF                     -Converts multi-page tif to DFF matrix
% signMapCombiner
% widefieldRespSeparator           -Extracts relevant frames and Fourier
%                                  transforms them
% B_widefieldPhaseMap              -Creates phase maps from Fourier transformed
%                                  data, and allows manual choosing of
%                                  correct frequency bin
% C_widefieldSignMap               -Uses phase map data to create sign maps

%%% Inputs %%%
% multi-page.tif                   -Multipage tif file from recording
% ref_img.picture                  -Reference image from recording for overlay

%%% Outputs %%%
% MATLAB FILES %
% F0.mat                           -Average activity map from widefieldDFF,
%                                  used in lieu of a reference image
% widefield_resp_dat.mat           -Sorted and meaned responses
% widefield_phase_mapped.mat       -Phase maps for azimuth and altitude
% VFS_raw.mat                      -Raw sign map
% VFS_processed.mat                -Binarized and cleaned sign map
%
% PICTURES %
% phase_maps.jpg                   -JPG of the phase maps
% sign_map.jpg                     -JPG of the sign map
% sign_map_overlay.jpg             -JPG of the sign map borders overlay

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic

fs = 10; % Define your acquisition frame rate (in Hz)

%% Choosing Files
disp(['=====================================================================' newline...
    '                           SIGN MAPPING                              ' newline...
    '====================================================================='])

N_recordings = inputdlg('How many recording blocks do you have?');
N_recordings = str2num(N_recordings{1});

for i = 1:N_recordings
    fprintf(['Choose multi-page.tif file for recording #%d/' num2str(N_recordings) '... \n'],i)
    [fn{i}, pn{i}] = uigetfile('.tif');
end

disp('Choose your stimulus data file...')
[fn_info, pn_info] = uigetfile('.mat');

% Seeing if a reference image was taken, if one was taken, it will be used
% for the overlay. If not, we will use the average activity (F0.mat) map.
ref_img_choice = questdlg('Did you take a reference image?','Reference image','Yes','No','Yes');

switch ref_img_choice
    case 'Yes'
        disp('Lastly, choose your reference image for overlay...')
        [fn_ref_img, pn_ref_img] = uigetfile({'*.jpg;*.png;*.gif','All Image Files';...
            '*.*','All Files' });
    case 'No'
        disp('No ref_img, overlay will be done on F0 map from widefieldDFF')
end

ExpInfo = importdata([pn_info fn_info]);
%% Calculating DFF matrix

disp([newline...
    '==================================================' newline...
    'Step 0: Calculating DFF matrix and Extracting Data' newline...
    '==================================================']);

for i = 1:N_recordings
    fprintf([ 'Processing recording block #' num2str(i) '/' num2str(N_recordings) '\n']);
    cd(pn{i})
    DFF = widefieldDFF(pn{i}, fn{i}, 0);    
    widefieldRespSeparatorV2(DFF,ExpInfo,fs);
    if N_recordings > 1
    cd ..
    end
end

%% Combining responses
[aziResp_f,aziResp_b,altResp_u, altResp_d] = signMapCombinerV2(fn,pn,N_recordings);

%% Calculating the Fourier Transforms
disp([newline...
    '=====================================' newline ...
    'Step A: Fourier transforming the data' newline ...
    '====================================='])

[aziFT_f,aziFT_b,altFT_u, altFT_d] = A_widefieldFourierTransformV2(aziResp_f,aziResp_b,altResp_u,altResp_d);

%% Creating the phase maps on your data, provides retinotopic maps
disp([newline...
    '========================================' newline ...
    'Step B: Phase map selection and creation' newline ...
    '========================================'])

[aziPhase_f,aziPhase_b,altPhase_u, altPhase_d] = B_widefieldPhaseMapV2(aziFT_f,aziFT_b,altFT_u, altFT_d, fs);
% 
% aziPhase = rot90(aziPhase_f - aziPhase_b,2); %new camera set up needs rotation
% altPhase = rot90(altPhase_u - altPhase_d,2);

aziPhase = aziPhase_f - aziPhase_b; %new camera set up needs rotation
altPhase = altPhase_u - altPhase_d;

%% Creating the phase map and processing it 
disp([newline...
    '=================================================' newline ...
    'Step C: Creating the sign map and post-processing' newline ...
    '================================================='])
try
    ref_img = imread([pn_ref_img fn_ref_img]);
catch
    cd(pn{1})
    ref_img = importdata('F0.mat');
end

C_widefieldSignMapV2(aziPhase, altPhase,ref_img);


disp([newline...
    '====================== FINISHED =========================='])

toc