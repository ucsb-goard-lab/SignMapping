function [aziResp_f, aziResp_b, altResp_u, altResp_d] = widefieldRespSeparatorTIMESTAMPS(DFF,ExpInfo,fs)
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
    fs=10;
end

%% Parameters
repeats     = ExpInfo.repeats;
on_frames   = ExpInfo.on_time*fs;
off_frames  = ExpInfo.off_time*fs;
dir_frames  = on_frames + off_frames; % flip frames are same axis, different directions
axis_frames = dir_frames * 2;        % direction frames are different axes (azimuth to altitude)
rep_frames  = axis_frames * 2;
%  old
% sweep_start = round(ExpInfo.TimingInfo.Start_EachSweep*fs);
% sweep_end = round(ExpInfo.TimingInfo.End_EachSweep*fs);
% 
% blank_start = round(ExpInfo.TimingInfo.Start_Blank*fs);
% blank_end = round(ExpInfo.TimingInfo.End_Blank*fs);

 %new
 
sweep_start = round(ExpInfo.mov_on*fs);
blank_start = round(ExpInfo.blank_on*fs);
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
    azi_off_fResp(:,:,:,rep)  = DFF(:,:,blank_start(rep,1)+1:blank_start(rep,1) + off_frames);
    azi_on_fResp(:,:,:,rep) = DFF(:,:,sweep_start(rep,1)+1:sweep_start(rep,1)+ on_frames);
    
    azi_off_bResp(:,:,:,rep)  = DFF(:,:,blank_start(rep,2)+1:blank_start(rep,2)+ off_frames);
    azi_on_bResp(:,:,:,rep) = DFF(:,:,sweep_start(rep,2)+1:sweep_start(rep,2)+ on_frames);
    
    alt_off_uResp(:,:,:,rep)  = DFF(:,:,blank_start(rep,3)+1:blank_start(rep,3)+ off_frames);
    alt_on_uResp(:,:,:,rep) = DFF(:,:,sweep_start(rep,3)+1:sweep_start(rep,3)+ on_frames);
    
    alt_off_dResp(:,:,:,rep)  = DFF(:,:,blank_start(rep,4)+1:blank_start(rep,4)+ off_frames);
    alt_on_dResp(:,:,:,rep) = DFF(:,:,sweep_start(rep,4)+1:sweep_start(rep,4)+ on_frames);
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
% mean across reps in each direction
h = fspecial('gaussian',75,25);

 %aziResp_f = imfilter(squeeze(mean(azi_on_fResp,4)),h);
%aziResp_b = imfilter(squeeze(mean(azi_on_bResp,4)),h);
% altResp_u = imfilter(squeeze(mean(alt_on_uResp,4)),h);
%altResp_d = imfilter(squeeze(mean(alt_on_dResp,4)),h);

%  aziResp_f = imgaussfilt(squeeze(mean(azi_on_fResp,4)),8);
% aziResp_b = imgaussfilt(squeeze(mean(azi_on_bResp,4)),8);
%  altResp_u = imgaussfilt(squeeze(mean(alt_on_uResp,4)),8);
% altResp_d = imgaussfilt(squeeze(mean(alt_on_dResp,4)),8);
% % 
%  aziResp_f = (squeeze(mean(azi_on_fResp,4)));
%  aziResp_b = (squeeze(mean(azi_on_bResp,4)));
%  altResp_u = (squeeze(mean(alt_on_uResp,4)));
%  altResp_d = (squeeze(mean(alt_on_dResp,4)));
 aziResp_f = azi_on_fResp;
 aziResp_b = azi_on_bResp;
 altResp_u = alt_on_uResp;
 altResp_d = alt_on_dResp;


%% new flag, testing smoothing

%% Saving data
disp('(4/4) Saving data...')
%save('widefield_resp_data.mat','altResp', 'aziResp') Raquel Abreu
%save('widefield_FT_data.mat','altResp_u', 'altResp_d' ,'aziResp_f','aziResp_b')
