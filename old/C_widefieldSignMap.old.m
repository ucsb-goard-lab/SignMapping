function [] = C_widefieldSignMap(aziPhase,altPhase,ref_img)
%% Step C: Sign Map Creator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written 31Jul2017 KS
% Last Updated: 
% 30Aug2017 KS - Added header info
% 06Nov2017 KS - Added the morphological image processing (closing and
% opening followed by dilation), allows for more expanded maps

% Takes phase maps and calculates the visual field sign maps (VFS). Then it
% processes the VFS (cleans up stray pixels and expands borders), then
% plots it on top of a reference image

%%% Necessary Subfunctions %%%
% None

%%% Inputs %%%
% aziPhase                           -Azimuth phase map
% altPhase                           -Altitude phase map
% ref_img                            -Reference image for overlay

%%% Outputs %%%
% VFS_raw.mat                        -Raw sign map
% VFS_processed.mat                  -Normalized, binarized, and cleaned sign map
% sign_map.jpg                       -Picture of VFS_raw
% sign_map_overlay.jpg               -Bordered sign map overlaid on ref_img

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin == 0
    disp('Choose your widefield_phase_mapped file...')
    [fn1 pn1] = uigetfile('.mat');
   
    disp('Choose your reference image...')
    [fn3, pn3] = uigetfile({'*.jpg;*.tif;*.png;*.gif','All Image Files';...
        '*.*','All Files' });
    
    load([pn1 fn1]);
    ref_img = imread([pn3 fn3]);
end

%% From Ming's code, creates the sign map
[dhdx, dhdy] = gradient(aziPhase); % Gradient in each direction
[dvdx, dvdy] = gradient(altPhase);

graddir_hor  = atan2(dhdy,dhdx); % angle between gradients
graddir_vert = atan2(dvdy,dvdx);

vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert); %Should be vert-hor, but the gradient in Matlab for y is opposite.
VFS = sin(angle(vdiff)); %Visual field sign map
id = find(isnan(VFS));
VFS(id) = 0;
% just filtering
hh = fspecial('gaussian',size(VFS),8); % 8 SD gaussion filter
hh = hh/sum(hh(:));
VFS = ifft2(fft2(VFS).*abs(fft2(hh)));  %Important to smooth before thresholding below

%% Displaying sign map

f = imagesc(VFS); % for checking
colormap(jet)
axis square

disp('(1/3) Press any button to close the sign map and continue...')
pause

saveas(f,'sign_map', 'jpg')
save VFS_raw.mat VFS
close all

%% Processing sign map
% Creating threshhold values for binarizing
thresh = mean(VFS(:));
sd_thresh = std(VFS(:)); %from Garrett et al 2014

% binarizing
VFS_binarized = zeros(size(VFS,1),size(VFS,2));
for i = 1: size(VFS_binarized,1)
    for j = 1:size(VFS_binarized,2)
        if VFS(i,j) > (thresh+sd_thresh)
            VFS_binarized(i,j) = 1;
        elseif VFS(i,j) < (thresh-sd_thresh)
      VFS_binarized(i,j) = -1;
        else
            VFS_binarized(i,j) = 0;
        end
    end
end

% morphological image processing
se = strel('disk',5); %not sure right now what the best pixel size is, so we'll leave it at 5 for now
VFS_processed = imclose(VFS_binarized,se);
VFS_processed = imopen(VFS_processed,se);
VFS_processed = imdilate(VFS_processed,se);

save VFS_processed.mat  VFS_processed

VFS_boundaries = bwmorph(abs(VFS_processed),'remove',10);


try
    % additional processing necessary if the reference image is a picture file
    ref_img_mat = rgb2gray(ref_img);
    
    min_sz = min(size(ref_img_mat));
    ref_img_mat = ref_img_mat(1:min_sz,1:min_sz);
    
    scaling_factor = size(VFS_boundaries,1)/size(ref_img_mat,1);
    ref_img_sc = imresize(ref_img_mat,scaling_factor);
catch
    % no processing necesasry if reference image is F0.mat
    ref_img_sc = ref_img;
end

sign_map_overlay = rot90(ref_img_sc);
sign_map_overlay(VFS_boundaries) = max(sign_map_overlay(:))*1.1;

f = imagesc(sign_map_overlay);
colormap gray
axis square

disp('(2/3) Press any button to close the overlay and continue...')
pause

disp('(3/3) Saving data...')
saveas(f,'sign_map_overlay', 'jpg')
close all

save area_map.mat sign_map_overlay VFS_boundaries
