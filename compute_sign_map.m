function maps = compute_sign_map(azi, alt, options)
% PLOT_RETINOTOPY Plot retinotopic maps with optional reference image
%
% Syntax:
%   plot_retinotopy(azi, alt)
%   plot_retinotopy(azi, alt, ref_image=ref_img)
%
% Inputs:
%   azi - Horizontal (azimuth) retinotopy map
%   alt - Vertical (altitude) retinotopy map
%   ref_image - (optional) Reference image to display as background under all maps

arguments
    azi
    alt
    options.ref_image = []
end

ref_image = options.ref_image;

if ~isempty(ref_image)
   ref_image = imresize(ref_image, [400 400]);
end

% Your existing code
aziPhase = azi;
altPhase = alt;
horz_factor = 145/360;
vert_factor = 124/360;
pixpermm = 20;

aziPhase = imgaussfilt(aziPhase, 3); % filter maps
altPhase = imgaussfilt(altPhase, 3);
aziPhase = aziPhase * horz_factor; % scale for screen
altPhase = altPhase * vert_factor;

kmap_hor_orig = double(aziPhase);
kmap_vert_orig = double(altPhase);

% Rotate & Up/Down Sample Maps
kmap_hor_orig = rot90(kmap_hor_orig, -1);
kmap_vert_orig = rot90(kmap_vert_orig, -1);
kmap_hor = rot90(kmap_hor_orig);
kmap_vert = rot90(kmap_vert_orig);

% Compute visual field sign map
mmperpix = 1/pixpermm;
[dhdx, dhdy] = gradient(kmap_hor);
[dvdx, dvdy] = gradient(kmap_vert);
graddir_hor = atan2(dhdy, dhdx);
graddir_vert = atan2(dvdy, dvdx);
vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert);
VFS = sin(angle(vdiff));

id = find(isnan(VFS));
VFS(id) = 0;
hh = fspecial('gaussian', size(VFS), 3);
hh = hh/sum(hh(:));
VFS = ifft2(fft2(VFS) .* abs(fft2(hh)));

% Plot retinotopic maps
xdom = (0:size(kmap_hor, 2)-1) * mmperpix;
ydom = (0:size(kmap_hor, 1)-1) * mmperpix;

figure(10)
screenDim = get(0, 'ScreenSize');

% Define aspect ratio (always 3 subplots)
aspectRatio = 7/2;

% Choose height as the limiting factor
height = screenDim(4) * 0.4;
width = height * aspectRatio;

% Calculate centered position
left = (screenDim(3) - width) / 2;
bottom = (screenDim(4) - height) / 2;
set(gcf, 'Position', [left, bottom, width, height]);

% Scale the maps
kmap_hor_normalized = (kmap_hor - min(kmap_hor(:))) / (max(kmap_hor(:)) - min(kmap_hor(:)));
kmap_hor_scaled = kmap_hor_normalized * 130;
kmap_vert_scaled = rescale(kmap_vert, -50, 50);

if isempty(ref_image)
    % Standard plots without reference image
    
    % Horizontal
    ax1 = subplot(1, 3, 1);
    imagesc(xdom, ydom, kmap_hor_scaled);
    axis image, colormap(ax1, 'jet'), colorbar
    title('1. Horizontal (azim deg)')
    ax1.XTick = [];
    ax1.YTick = [];
    
    % Vertical
    ax2 = subplot(1, 3, 2);
    imagesc(xdom, ydom, kmap_vert_scaled),
    axis image, colormap(ax2, 'jet'), colorbar
    title('2. Vertical (alt deg)')
    ax2.XTick = [];
    ax2.YTick = [];
    
    % VFS
    ax3 = subplot(1, 3, 3);
    imagesc(xdom, ydom, VFS, [-1 1]), axis image
    colorbar,
    colormap(ax3, 'jet')
    ax3.XTick = [];
    ax3.YTick = [];
    title('3. Sereno: sin(angle(Hor)-angle(Vert))')
    
else
    % Plots with reference image overlay
    
    % Horizontal with reference
    ax1_temp = subplot(1, 3, 1);
    ax1_pos = get(ax1_temp, 'Position');
    delete(ax1_temp);
    
    ax1_bg = axes('Position', ax1_pos);
    im_bg1 = imagesc(ax1_bg, ref_image);
    im_bg1.AlphaData = 1;
    axis equal;
    hold all;
    
    ax1_fg = axes('Position', ax1_pos);
    im_fg1 = imagesc(ax1_fg, kmap_hor_scaled);
    im_fg1.AlphaData = 0.5;
    axis equal;
    
    ax1_fg.Visible = 'off';
    ax1_fg.XTick = [];
    ax1_fg.YTick = [];
    ax1_bg.Visible = 'off';
    ax1_bg.XTick = [];
    ax1_bg.YTick = [];
    
    linkaxes([ax1_bg, ax1_fg])
    colormap(ax1_bg, 'gray')
    colormap(ax1_fg, 'jet')
    
    title(ax1_fg, '1. Horizontal (azim deg)', 'FontSize', 16)
    
    % Vertical with reference
    ax2_temp = subplot(1, 3, 2);
    ax2_pos = get(ax2_temp, 'Position');
    delete(ax2_temp);
    
    ax2_bg = axes('Position', ax2_pos);
    im_bg2 = imagesc(ax2_bg, ref_image);
    im_bg2.AlphaData = 1;
    axis equal;
    hold all;
    
    ax2_fg = axes('Position', ax2_pos);
    im_fg2 = imagesc(ax2_fg, kmap_vert_scaled);
    im_fg2.AlphaData = 0.5;
    axis equal;
    
    ax2_fg.Visible = 'off';
    ax2_fg.XTick = [];
    ax2_fg.YTick = [];
    ax2_bg.Visible = 'off';
    ax2_bg.XTick = [];
    ax2_bg.YTick = [];
    
    linkaxes([ax2_bg, ax2_fg])
    colormap(ax2_bg, 'gray')
    colormap(ax2_fg, 'jet')
    
    title(ax2_fg, '2. Vertical (alt deg)', 'FontSize', 16)
    
    % VFS with reference
    ax3_temp = subplot(1, 3, 3);
    ax3_pos = get(ax3_temp, 'Position');
    delete(ax3_temp);
    
    ax3_bg = axes('Position', ax3_pos);
    im_bg3 = imagesc(ax3_bg, ref_image);
    im_bg3.AlphaData = 1;
    axis equal;
    hold all;
    
    ax3_fg = axes('Position', ax3_pos);
    im_fg3 = imagesc(ax3_fg, imgaussfilt(VFS, 1.5), [-1 1]);
    im_fg3.AlphaData = 0.5;
    axis equal;
    
    ax3_fg.Visible = 'off';
    ax3_fg.XTick = [];
    ax3_fg.YTick = [];
    ax3_bg.Visible = 'off';
    ax3_bg.XTick = [];
    ax3_bg.YTick = [];
    
    linkaxes([ax3_bg, ax3_fg])
    colormap(ax3_bg, 'gray')
    colormap(ax3_fg, 'jet')
    
    title(ax3_fg, '3. Sereno: sin(angle(Hor)-angle(Vert))', 'FontSize', 16)
end

maps = struct();
maps.alt = alt;
maps.azi = azi;
maps.VFS = VFS;

end