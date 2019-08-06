function [] = signMapMasterNEW()
%% Sign Map Master File
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Written 29 Aug 2017 KS
% Last Updated:
% 30 Aug 2017 KS - Added header information
%                - Adjusted the code for better handling of multiple
%                  recording blocks
% 01 Feb 2018 KS - Changed the way we combine blocks in different
%                  directions
%                - Fixed some bugs and made it more compatible with all the
%                  multi-tif files in a single directory

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


%% adding the new lines of code for retinotopic mapping, need it
curr_script_dir = mfilename('fullpath');
parent_script_dir = curr_script_dir(1:end-17);
addpath(fullfile(parent_script_dir,'Juavinett et al 2017 Code'));

addpath('C:\Users\sit\Dropbox\Sign Mapping\Juavinett et al 2017 Code');
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

for i = 1:N_recordings
    fprintf(['Choose your stimulus data file for recording #%d/' num2str(N_recordings) '... \n'],i)
    [fn_info{i}, pn_info{i}] = uigetfile('.mat');
    ExpInfo(i) = importdata([pn_info{i},fn_info{i}]);
end

% Seeing if a reference image was taken, if one was taken, it will be used
% for the overlay. If not, we will use the average activity (F0.mat) map.
ref_img_choice = questdlg('Did you take a reference image?','Reference image','Yes','No','Yes');

switch ref_img_choice
    case 'Yes'
        disp('Lastly, choose your reference image for overlay...')
        [fn_ref_img, pn_ref_img] = uigetfile({'*.jpg;*.png;*.gif;*.tif','All Image Files';...
            '*.*','All Files' });
    case 'No'
        disp('No ref_img, overlay will be done on F0 map from widefieldDFF')
end


%% Calculating DFF matrix

disp([newline...
    '==================================================' newline...
    'Step 0: Calculating DFF matrix and Extracting Data' newline...
    '==================================================']);
DFF = cell(1,N_recordings);
for i = 1:N_recordings
    fprintf([ 'Processing recording block #' num2str(i) '/' num2str(N_recordings) '\n']);
    cd(pn{i})
    % DFF{i} = widefieldDFF(pn{i}, fn{i}, 0);
    DFF{i} = widefieldDFF(pn{i}, fn{i}, 0);
    if N_recordings > 1
        cd ..
    end
end


% to adjust for old timing inaccuracies, cutting each recording to the
% expected length
% expected_length = (ExpInfo.Repeats*(ExpInfo.StimOn+ExpInfo.StimOff) * 4)*fs; %4 is for the 4 directions
%
% for ii = 1:length(DFF)
%     if size(DFF{ii},3) ~= expected_length
%         DFF{ii} = DFF{ii}(:,:,1:expected_length);
%     end
% end

%% removing slow changes, ala Kalatsky & Stryker 2003

% doesn't really do anything for fluorescence data... let's just keep it
% out for now

% for ii = 1:length(DFF)
%     for y = 1:size(DFF{ii},1)
%         for x = 1:size(DFF{ii},2)
%             slow_change(y,x,:) = smooth(DFF{ii}(y,x,:),((ExpInfo.StimOn + ExpInfo.StimOff)*2*fs+1)); %2 cycles of stimulus
%         end
%     end
% end

testing_flag = 0;
if testing_flag
    %define your window
    imagesc(mean(DFF{1},3));
    colormap jet
    axis off
    axis square
    window_roi = impoly();
    window_mask = window_roi.createMask;
    for jj = 1:length(DFF)
        for ii =1:size(DFF{jj},3)
            disp(num2str(ii))
            curr_frame = DFF{jj}(:,:,ii);
            window_resp{jj}(:,ii) = curr_frame(window_mask);
        end
    end
    window_resp2 = cellfun(@(x) mean(x,1),window_resp,'UniformOutput',false);
    
    for ii = 1:length(DFF)
        for frme = 1:size(DFF{ii},3)
            disp(num2str(frme))
            sub_mat = repmat(window_resp{jj}(frme),[400 400]);
            curr_frame = DFF{ii}(:,:,frme);
            %curr_frame(~window_mask) = nan;
            DFF_sub{ii}(:,:,frme) = curr_frame - sub_mat;
        end
    end
    
    for ii = 1:length(DFF_sub)
        [aziResp_f(:,:,:,:,ii), aziResp_b(:,:,:,:,ii), altResp_u(:,:,:,:,ii), altResp_d(:,:,:,:,ii)] = widefieldRespSeparatorTIMESTAMPS(DFF_sub{ii},ExpInfo(ii),fs);
    end
else
    %
    for ii = 1:length(DFF)
        [aziResp_f(:,:,:,:,ii), aziResp_b(:,:,:,:,ii), altResp_u(:,:,:,:,ii), altResp_d(:,:,:,:,ii)] = widefieldRespSeparatorTIMESTAMPS(DFF{ii},ExpInfo(ii),fs);
    end
end

style = 'meancat'; %'meanmean' = mean within recording, mean across recording
%'catmean'  = cat within a recording, mean across recordings
%'meancat'  = mean within a recording, cat across recordings
%'catcat'   = cat within a recording, cat across recordings

img_dims = size(altResp_d,1);
switch style
    case 'meanmean'
        
        aziResp_f =  ((aziResp_f(:,:,:,:,2))); %meaning across the recording
        aziResp_b =  ((aziResp_b(:,:,:,:,2)));
        altResp_u =  ((altResp_u(:,:,:,:,2)));
        altResp_d =  ((altResp_d(:,:,:,:,2)));
        aziResp_f =  ((aziResp_f(:,:,:,:,2))); %meaning within the recording
        aziResp_b =  ((aziResp_b(:,:,:,:,2)));
        altResp_u =  ((altResp_u(:,:,:,:,2)));
        altResp_d =  ((altResp_d(:,:,:,:,2)));
%         
%         aziResp_f =  (mean(aziResp_f,5)); %meaning across the recording
%         aziResp_b =  (mean(aziResp_b,5));
%         altResp_u =  (mean(altResp_u,5));
%         altResp_d =  (mean(altResp_d,5));
%         aziResp_f =  (mean(aziResp_f,4)); %meaning within the recording
%         aziResp_b =  (mean(aziResp_b,4));
%         altResp_u =  (mean(altResp_u,4));
%         altResp_d =  (mean(altResp_d,4));
    case 'catmean'
        aziResp_f =  (mean(aziResp_f,5)); %meaning across the recording
        aziResp_b =  (mean(aziResp_b,5));
        altResp_u =  (mean(altResp_u,5));
        altResp_d =  (mean(altResp_d,5));
        aziResp_f =  reshape(aziResp_f,img_dims,img_dims,[]); % now, combining across recordingz
        aziResp_b =  reshape(aziResp_b,img_dims,img_dims,[]);
        altResp_u =  reshape(altResp_u,img_dims,img_dims,[]);
        altResp_d =  reshape(altResp_d,img_dims,img_dims,[]);
    case 'meancat'
        aziResp_f =  (mean(aziResp_f,4)); %meaning within the recording
        aziResp_b =  (mean(aziResp_b,4));
        altResp_u =  (mean(altResp_u,4));
        altResp_d =  (mean(altResp_d,4));
        aziResp_f =  reshape(aziResp_f,img_dims,img_dims,[]); % now, combining across recordingz
        aziResp_b =  reshape(aziResp_b,img_dims,img_dims,[]);
        altResp_u =  reshape(altResp_u,img_dims,img_dims,[]);
        altResp_d =  reshape(altResp_d,img_dims,img_dims,[]);
    case 'catcat'
        aziResp_f =  reshape(aziResp_f,img_dims,img_dims,[]); % now, combining across recordingz
        aziResp_b =  reshape(aziResp_b,img_dims,img_dims,[]);
        altResp_u =  reshape(altResp_u,img_dims,img_dims,[]);
        altResp_d =  reshape(altResp_d,img_dims,img_dims,[]);
end


%% testing the imresize so that I can actually run the fft...
% aziResp_f = imresize(aziResp_f,[200 200]);
% aziResp_b = imresize(aziResp_b,[200 200]);
% altResp_u = imresize(altResp_u,[200 200]);
% altResp_d = imresize(altResp_d,[200 200]);


% 
% %%smoothing in the temporal domain?
% tic
% windowSize = 10; 
% b = (1/windowSize)*ones(1,windowSize);
% a = 1;
% for x = 1:size(aziResp_f,1)
%     for y = 1:size(aziResp_b,2)
%         disp([num2str(x) ',' num2str(y)])
%         aziResp_fs(x,y,:) = filter(b,a,aziResp_f(x,y,:));
%         aziResp_bs(x,y,:) = filter(b,a,aziResp_b(x,y,:));
%         altResp_us(x,y,:) = filter(b,a,altResp_u(x,y,:));
%         altResp_ds(x,y,:) = filter(b,a,altResp_d(x,y,:));
%     end
% end
% toc
% %if testing_flag
%     %designig a butterwort filter to LP 0.5Hz
%     fc = 0.5;
%     fs = 10;
%     [b,a] = butter(5,fc/(fs/2));
%     aziResp_f = filter(b,a,aziResp_f);
%     aziResp_b = filter(b,a,aziResp_b);
%     altResp_u = filter(b,a,altResp_u);
%     altResp_d = filter(b,a,altResp_d);
% %end


%
% aziResp_f = reshape(aziResp_f,400,400,[]);
% aziResp_b = reshape(aziResp_b,400,400,[]);
% altResp_u = reshape(altResp_u,400,400,[]);
% altResp_d = reshape(altResp_d,400,400,[]);

%% Calculating the Fourier Transforms
disp([newline...
    '=====================================' newline ...
    'Step A: Fourier transforming the data' newline ...
    '====================================='])


% %% testing ,running a moving average filter over the data first...
% filt_val = 1;

% for ii = 1:size(aziResp_f,3)
%     aziResp_f_filt(:,:,ii) = imgaussfilt(aziResp_f(:,:,ii),filt_val);
%     aziResp_b_filt(:,:,ii) = imgaussfilt(aziResp_b(:,:,ii),filt_val);
%     altResp_u_filt(:,:,ii) = imgaussfilt(altResp_u(:,:,ii),filt_val);
%     altResp_d_filt(:,:,ii) = imgaussfilt(altResp_d(:,:,ii),filt_val);
% end


tic
aziFT_f = A_widefieldFourierTransformNEW(aziResp_f);
aziFT_b = A_widefieldFourierTransformNEW(aziResp_b);
altFT_u = A_widefieldFourierTransformNEW(altResp_u);
altFT_d = A_widefieldFourierTransformNEW(altResp_d);
toc

% [aziFT_f,aziFT_b,altFT_u, altFT_d] = A_widefieldFourierTransformNEW(aziResp_f,aziResp_b,altResp_u,altResp_d);

% %% Creating the phase maps on your data, provides retinotopic maps
% disp([newline...
%     '========================================' newline ...
%     'Step B: Phase map selection and creation' newline ...
%     '========================================'])
%


% %% new testy thing
% for k= 1:49
%     
%     subplot(7,7,k)
%     [aziPhase, altPhase] = B_widefieldPhaseMapNEW(aziFT_f,aziFT_b,altFT_u, altFT_d,k);
%     %imagesc([aziPhase altPhase])
%     aziPhase = imgaussfilt(aziPhase,2);
%     altPhase = imgaussfilt(altPhase,2);
%     
% 
% 
%  
%     horz_factor =  145/360; % this isn't worknig well right now, so eccentricity is not in degrees, but in arbitrary phase
%     vert_factor = 124/360;
%     
%     aziPhase = aziPhase*horz_factor;
%     altPhase = altPhase*vert_factor;
%     kmap_hor_orig= double(aziPhase); % negative to correct values
%     kmap_vert_orig= double(altPhase);
%     kmap_hor_orig = rot90(kmap_hor_orig,-1);
%     kmap_vert_orig = rot90(kmap_vert_orig,-1);
%     
%     kmap_hor = resample(kmap_hor_orig,2,5);
%     kmap_hor = resample(rot90(kmap_hor),2,5);
%     
%     kmap_vert = resample(kmap_vert_orig,2,5);
%     kmap_vert = resample(rot90(kmap_vert),2,5);
%     
%     kmap_hor_orig = rot90(kmap_hor_orig);
%     kmap_vert_orig = rot90(kmap_vert_orig);
%     
%     pixpermm = 40;
%     
%     mmperpix = 1/pixpermm;
%     
%     [dhdx dhdy] = gradient(kmap_hor);
%     [dvdx dvdy] = gradient(kmap_vert);
%     
%     graddir_hor = atan2(dhdy,dhdx);
%     graddir_vert = atan2(dvdy,dvdx);
%     
%     vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert); %Should be vert-hor, but the gradient in Matlab for y is opposite.
%     VFS = sin(angle(vdiff)); %Visual field sign map
%     id = find(isnan(VFS));
%     VFS(id) = 0;
%     
%     hh = fspecial('gaussian',size(VFS),3);
%     hh = hh/sum(hh(:));
%     VFS = ifft2(fft2(VFS).*abs(fft2(hh)));  %Important to smooth before thresholding below
%     
%     imagesc(VFS)
%     axis off
%     axis square
%     colormap jet
%     title(num2str(k));
% end
% 
% k = input('Input optimal k-value: ');

%% NEWnew testy thing 05Jun2018
figure('units','normalized','outerposition',[0 0 1 1]);
for k= 1:6
   
    subplot(2,3,k)
   [aziPhase, altPhase] = B_widefieldPhaseMapNEW(aziFT_f,aziFT_b,altFT_u, altFT_d,k);
    
    
    
    %imagesc([aziPhase altPhase])
    aziPhase = imgaussfilt(aziPhase,2);
    altPhase = imgaussfilt(altPhase,2);
    


 
    horz_factor =  145/360; % this isn't worknig well right now, so eccentricity is not in degrees, but in arbitrary phase
    vert_factor = 124/360;
    
    aziPhase = aziPhase*horz_factor;
    altPhase = altPhase*vert_factor;
    kmap_hor_orig= double(aziPhase); % negative to correct values
    kmap_vert_orig= double(altPhase);
    kmap_hor_orig = rot90(kmap_hor_orig,-1);
    kmap_vert_orig = rot90(kmap_vert_orig,-1);
    
    kmap_hor = resample(kmap_hor_orig,2,5);
    kmap_hor = resample(rot90(kmap_hor),2,5);
    
    kmap_vert = resample(kmap_vert_orig,2,5);
    kmap_vert = resample(rot90(kmap_vert),2,5);
    
   imagesc([kmap_hor, kmap_vert])
   title(num2str(k));
   axis image
end

k = inputdlg({'Azimuth k: ', 'Altitude k: '});
k = str2num(cell2mat(k));

close all;
% switch style
%     case 'meancat'
%         k=4;
%     case 'catcat'
%         k=31;
% end

% [aziPhase, altPhase] = B_widefieldPhaseMapNEW(aziFT_f,aziFT_b,altFT_u, altFT_d);
%

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

mkdir([pn{1} '/Additional Sign Map Info']);

SaveDir = [pn{1} 'Additional Sign Map Info'];

[maps] = C_widefieldSignMapBIGMONITOR(aziFT_f,aziFT_b,altFT_u, altFT_d,ref_img,SaveDir,k);

VFS_raw = maps.VFS_raw;
VFS_processed = maps.VFS_processed;
VFS_boundaries = maps.VFS_boundaries;

switch style
    case 'meanmean'
        name = 'VFS_meanmean.mat';
    case 'catmean'
        name = 'VFS_catmean.mat';
    case 'meancat' 
        name = 'VFS_meancat.mat';
    case 'catcat'
        name = 'VFS_catcat.mat';
end

save(name, 'VFS_raw' ,'VFS_processed' ,'VFS_boundaries','maps')
old = cd(SaveDir);
save Additional_maps.mat maps
cd(old);

%% making some useful figures for the future
% VFS processed with boundaries

disp([newline...
    '====================== FINISHED =========================='])
toc



