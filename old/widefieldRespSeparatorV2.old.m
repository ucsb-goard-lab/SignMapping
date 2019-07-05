function [] = widefieldRespSeparatorV2(DFF,ExpInfo,fs)
%% Step A: Widefield Response Separator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Changes from V1:
% no longer combines the forward and backward responses, do it later, after
% phase maps...

% Written 30Aug2017 KS
% Last Updated: 
% 9/6/2017 RA 
% Changed parameters StimOn and StimOff to match with Exp.Info
% Replaced 'widefield_resp_data.mat' by 'widefield_FT_data.mat' 

% Isolates azimuth and altitude frames, then creates a meaned response
% matrix

%%% Necessary Subfunctions %%%
% None

%%% Inputs %%%
% DFF.mat                    -DFF matrix from widefieldDFF
% ExpInfo.mat                -Stimulus data file from recording

%%% Outputs %%%
% aziResp                     -Meaned azimuth response
% altResp                     -Meaned altitud eresponse
% widefield_resp_data.mat     -aziResp and altResp saved into a single .mat file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 0    
    disp('Choose raw DFF data...')
    [filename1, pathname1] = uigetfile('.mat');
    
    disp('Choose stimulus data file...')
    [filename2, pathname2] = uigetfile('.mat');
    
    disp('Loading data...')
    DFF = importdata([pathname1 filename1]);
    ExpInfo = importdata([pathname2 filename2]);
end

%% Parameters
repeats     = ExpInfo.Repeats;
on_frames   = ExpInfo.StimOn*fs;
off_frames  = ExpInfo.StimOff*fs;
dir_frames  = on_frames + off_frames; % flip frames are same axis, different directions
axis_frames = dir_frames * 2;        % direction frames are different axes (azimuth to altitude)
rep_frames  = axis_frames * 2;

%% Preallocating response matrices
azi_on_fResp  = zeros(size(DFF,1),size(DFF,2),on_frames,repeats,'single');
azi_on_bResp  = zeros(size(DFF,1),size(DFF,2),on_frames,repeats,'single');

azi_off_fResp = zeros(size(DFF,1),size(DFF,2),off_frames,repeats,'single');
azi_off_bResp = zeros(size(DFF,1),size(DFF,2),off_frames,repeats,'single');

alt_on_uResp  = zeros(size(DFF,1),size(DFF,2),on_frames,repeats,'single');
alt_on_dResp  = zeros(size(DFF,1),size(DFF,2),on_frames,repeats,'single');

alt_off_uResp = zeros(size(DFF,1),size(DFF,2),off_frames,repeats,'single');
alt_off_dResp = zeros(size(DFF,1),size(DFF,2),off_frames,repeats,'single');

%% Extracting Responses
disp('(1/4) Extracting responses...')
for rep = 1:repeats
    curr_frame = (rep-1)*rep_frames;
    
    azi_off_fResp(:,:,:,rep)  = DFF(:,:,curr_frame+1:curr_frame+off_frames);
    azi_on_fResp(:,:,:,rep) = DFF(:,:,curr_frame+off_frames+1:curr_frame+on_frames+off_frames);
    
    azi_off_bResp(:,:,:,rep)  = DFF(:,:,curr_frame+dir_frames+1:curr_frame+dir_frames+off_frames);
    azi_on_bResp(:,:,:,rep) = DFF(:,:,curr_frame+dir_frames+off_frames+1:curr_frame+dir_frames+on_frames+off_frames);
    
    alt_off_uResp(:,:,:,rep)  = DFF(:,:,curr_frame+axis_frames+1:curr_frame+axis_frames+off_frames);
    alt_on_uResp(:,:,:,rep) = DFF(:,:,curr_frame+axis_frames+off_frames+1:curr_frame+axis_frames+on_frames+off_frames);
    
    alt_off_dResp(:,:,:,rep)  = DFF(:,:,curr_frame+axis_frames+dir_frames+1:curr_frame+axis_frames+dir_frames+off_frames);
    alt_on_dResp(:,:,:,rep) = DFF(:,:,curr_frame+axis_frames+dir_frames+off_frames+1:curr_frame+axis_frames+dir_frames+on_frames+off_frames);
end
    
%% Calculating off responses for baseline subtraction
disp('(2/4) Baseline subtracting...')

% meaning off responses across frames per pixel, preserving reps
m_azi_off_fResp = squeeze(mean(azi_off_fResp,3));
m_azi_off_bResp = squeeze(mean(azi_off_bResp,3));

m_alt_off_uResp = squeeze(mean(alt_off_uResp,3));
m_alt_off_dResp = squeeze(mean(alt_off_dResp,3));

% subtracting off responses from corresponding on responses
for rep = 1:repeats
    azi_on_fResp(:,:,:,rep) = azi_on_fResp(:,:,:,rep) - m_azi_off_fResp(:,:,rep);
    azi_on_bResp(:,:,:,rep) = azi_on_bResp(:,:,:,rep) - m_azi_off_bResp(:,:,rep);
    alt_on_uResp(:,:,:,rep) = alt_on_uResp(:,:,:,rep) - m_alt_off_uResp(:,:,rep);
    alt_on_dResp(:,:,:,rep) = alt_on_dResp(:,:,:,rep) - m_alt_off_dResp(:,:,rep);
end

%% Meaning responses in both directions
disp('(3/4) Meaning responses...')
% combines responses in either direction
aziResp_f = squeeze(mean(azi_on_fResp,4));
aziResp_b = squeeze(mean(azi_on_bResp,4));
altResp_u = squeeze(mean(alt_on_uResp,4));
altResp_d = squeeze(mean(alt_on_dResp,4));

%% Saving data
disp('(4/4) Saving data...')
%save('widefield_resp_data.mat','altResp', 'aziResp') Raquel Abreu
save('widefield_FT_data.mat','altResp_u', 'altResp_d' ,'aziResp_f','aziResp_b')
