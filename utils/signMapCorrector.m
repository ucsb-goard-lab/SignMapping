%signMapCorrecter
clear
%% load the previous mapping data
skip_flag = 0;
load('VFS_meancat.mat')

old_map = maps.VFS_raw;
%% Extract retinotopic maps
vert_ret = maps.VerticalRetinotopy;
horz_ret = maps.HorizontalRetinotopy;

%% Unwrap maps
vert_ret_corrected = phase_unwrap(vert_ret);
horz_ret_corrected = phase_unwrap(horz_ret);

%% re-process if possible..


mkdir([pwd '/Additional Sign Map Info New']);

SaveDir = [pwd '/Additional Sign Map Info New'];

anatomypic = maps.ReferenceImage;

LP = [0.1]; %(0,2]

a=0;
    while a == 0
        %% Creating contour maps

% First, let's try choosing the value closest to the expected stimulus
% frequency, which typically resides in bin 2 (the first peak harmonic
% frequency)

  bw = ones(size(vert_ret_corrected));

L = fspecial('gaussian',50,LP);  %make LP spatial filter
        altPhase = roifilt2(L,vert_ret_corrected,bw,'same');
        aziPhase = roifilt2(L,horz_ret_corrected,bw,'same');

      
%          horz_factor =  145/360; % this isn't worknig well right now, so eccentricity is not in degrees, but in arbitrary phase
%          vert_factor = 124/360;
%          
%        aziPhase = aziPhase*horz_factor;
%         altPhase = altPhase*vert_factor;

       % disp(['Current Low-pass filter value: ' num2str(LP) ' sigma'])
            kmap_hor_orig= double(aziPhase); % negative to correct values
            kmap_vert_orig= double(altPhase);


            %% Rotate & Up/Down Sample Maps
            % The images in Garrett et al '14 were collected at 39 pixels/mm.  It is
            % recommended that kmap_hor and kmap_vert be down/upsampled to this value
            % before running. The code below downsamples by 2 to achieve 39 pixels/mm

            kmap_hor_orig = rot90(kmap_hor_orig,-1);
            kmap_vert_orig = rot90(kmap_vert_orig,-1);

                kmap_hor = resample(kmap_hor_orig,2,5);
                kmap_hor = resample(rot90(kmap_hor),2,5); 
                
                kmap_vert = resample(kmap_vert_orig,2,5);
                kmap_vert = resample(rot90(kmap_vert),2,5);
                


            kmap_hor_orig = rot90(kmap_hor_orig);
            kmap_vert_orig = rot90(kmap_vert_orig);

            pixpermm = 40;

            %% Low pass filtering the retinotopic maps before continuing...


            %% Compute visual field sign map

            mmperpix = 1/pixpermm;

            [dhdx dhdy] = gradient(kmap_hor);
            [dvdx dvdy] = gradient(kmap_vert);

            graddir_hor = atan2(dhdy,dhdx);
            graddir_vert = atan2(dvdy,dvdx);

            vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert); %Should be vert-hor, but the gradient in Matlab for y is opposite.
            VFS = sin(angle(vdiff)); %Visual field sign map
            id = find(isnan(VFS));
            VFS(id) = 0;

            hh = fspecial('gaussian',size(VFS),3);
            hh = hh/sum(hh(:));
            VFS = ifft2(fft2(VFS).*abs(fft2(hh)));  %Important to smooth before thresholding below

            %% Plot retinotopic maps

            xdom = (0:size(kmap_hor,2)-1)*mmperpix;
            ydom = (0:size(kmap_hor,1)-1)*mmperpix;

            screenDim = get(0,'ScreenSize');
            figure(10), clf
            set(10,'Position',[0,0,screenDim(3),screenDim(4)])

            ax1= subplot(3,4,1);
            imagesc(xdom,ydom,kmap_hor,[-50 50]);
            axis image, colormap(ax1,'hsv'), colorbar
            title('1. Horizontal (azim deg)')

            ax2 = subplot(3,4,2);
            imagesc(xdom,ydom,kmap_vert,[-50 50]),
            axis image, colormap(ax2,'hsv'), colorbar
            title('2. Vertical (alt deg)')

            %% Plotting visual field sign and its threshold

            figure(10), ax3 = subplot(3,4,3);
            imagesc(xdom,ydom,VFS,[-1 1]), axis image
            colorbar
            colormap(ax3,'jet')
            title('3. Sereno: sin(angle(Hor)-angle(Vert))')

            gradmag = abs(VFS);
            figure(10), subplot(3,4,4),

            threshSeg = 1.5*std(VFS(:));
            imseg = (sign(gradmag-threshSeg/2) + 1)/2;  %threshold visual field sign map at +/-1.5sig

            id = find(imseg);
            imdum = imseg.*VFS; imdum(id) = imdum(id)+1.1;
            plotmap(imdum,[.1 2.1],pixpermm);
            colorbar off
            axis image
            title(['4. +/-1.5xSig = ' num2str(threshSeg)])

            patchSign = getPatchSign(imseg,VFS);

            figure(10), subplot(3,4,5),
            plotmap(patchSign,[1.1 2.1],pixpermm);
            title('watershed')
            colorbar off
            title('5. Threshold patches')

            id = find(patchSign ~= 0);
            patchSign(id) = sign(patchSign(id) - 1);

            SE = strel('disk',2,0);
            imseg = imopen(imseg,SE);

            patchSign = getPatchSign(imseg,VFS);

            figure(10), subplot(3,4,6),
            ploteccmap(patchSign,[1.1 2.1],pixpermm);
            title('watershed')
            colorbar off
            title('6. "Open" & set boundary')

            %% Make boundary of visual cortex

            Npad = 30;  %Arbitrary padding value.  May need more depending on image size and resolution
            dim = size(imseg);
            imsegpad = [zeros(dim(1),Npad) imseg zeros(dim(1),Npad)];
            dim = size(imsegpad);
            imsegpad = [zeros(Npad,dim(2)); imsegpad; zeros(Npad,dim(2))];

            SE = strel('disk',10,0);
            imbound = imclose(imsegpad,SE);

            imbound = imfill(imbound); %often unnecessary, but sometimes there are small gaps need filling

            %SE = strel('disk',5,0); % removed bc too aggressive, would get rid of small areas
            % imbound = imopen(imbound,SE);

            SE = strel('disk',3,0);
            imbound = imdilate(imbound,SE); %Dilate to account for original thresholding.
            imbound = imfill(imbound);

            %Remove the padding
            imbound = imbound(Npad+1:end-Npad,Npad+1:end-Npad);
            imbound(:,1) = 0; imbound(:,end) = 0; imbound(1,:) = 0;  imbound(end,:) = 0;

            %Only keep the "main" group of patches. Preveiously used opening (see above), but this is more robust:
            bwlab = bwlabel(imbound,4);
            labid = unique(bwlab);
            for i = 1:length(labid)
                id = find(bwlab == labid(i));
                S(i) = length(id);
            end
            S(1) = 0; %To ignore the "surround patch"
            [dum id] = max(S);
            id = find(bwlab == labid(id));
            imbound = 0*imbound;
            imbound(id) = 1;
            imseg = imseg.*imbound;

            clear S

            %This is important in case a patch reaches the edge... we want it to be
            %smaller than imbound
            imseg(:,1:2) = 0; imseg(:,end-1:end) = 0; imseg(1:2,:) = 0;  imseg(end-1:end,:) = 0;

            figure(10), subplot(3,4,6)
            hold on
            contour(xdom,ydom,imbound,[.5 .5],'k')

            %% Morphological thinning to create borders that are one pixel wide

            %Thinning
            bordr = imbound-imseg;
            bordr = bwmorph(bordr,'thin',Inf);
            bordr = bwmorph(bordr,'spur',4);

            %Turn border map into patches
            im = bwlabel(1-bordr,4);
            im(find(im == 1)) = 0;
            im = sign(im);


            %% Plot patches

            patchSign = getPatchSign(im,VFS);
best_i_can_do = patchSign; % so this is if everything goes badly, then at least you have something...
            figure(10), subplot(3,4,7),
            ploteccmap(patchSign,[1.1 2.1],pixpermm);
            hold on,
            contour(xdom,ydom,im,[.5 .5],'k')
            title('"Thinning"')
            colorbar off

            %% Plot eccentricity map, with [0 0] defined as V1's center-of-mass

            SE = strel('disk',10);
            imdum = imopen(imseg,SE);
            [CoMxy Axisxy] = getPatchCoM(imdum);

            V1id = getV1id(imdum);

            AreaInfo.Vcent(1) = kmap_hor(round(CoMxy(V1id,2)),round(CoMxy(V1id,1)));  %Get point in visual space at the center of V1
            AreaInfo.Vcent(2) = kmap_vert(round(CoMxy(V1id,2)),round(CoMxy(V1id,1)));

            az = (kmap_hor - AreaInfo.Vcent(1))*pi/180; %azimuth
            alt = (kmap_vert - AreaInfo.Vcent(2))*pi/180; %altitude
            AreaInfo.kmap_rad = atan(  sqrt( tan(az).^2 + (tan(alt).^2)./(cos(az).^2)  )  )*180/pi;  %Eccentricity


            subplot(3,4,8)
            ploteccmap(AreaInfo.kmap_rad.*im,[0 45],pixpermm);
            hold on
            contour(xdom,ydom,im,[.5 .5],'k')

            axis image
            title('8. Eccentricity map')

            
            
%             try 
            %% ID redundant patches and split them (criterion #2)
             if ~skip_flag
                im = splitPatchesX(im,kmap_hor,kmap_vert,AreaInfo.kmap_rad,pixpermm); %% decreased minimum patch size thing from 0.01 to 0.001 to keep areas like AM...
            

                %Remake the border with thinning
                bordr = imbound-im;
                bordr = bwmorph(bordr,'thin',Inf);
                bordr = bwmorph(bordr,'spur',4);

                %Turn border map into patches
                im = bwlabel(1-bordr,4);
                im(find(im == 1)) = 0;
                im = sign(im);
                SE = strel('disk',2);
                im = imopen(im,SE);


                %% ID adjacent patches of the same VFS and fuse them if not redundant (criterion #3)

                [im fuseflag] = fusePatchesX(im,kmap_hor,kmap_vert,pixpermm);

            [patchSign areaSign] = getPatchSign(im,VFS);

            figure(10), subplot(3,4,9),
            ploteccmap(im.*AreaInfo.kmap_rad,[0 45],pixpermm);
            colorbar
            title('9. Split redundant patches. Fuse exclusive patches.')

            figure(10), subplot(3,4,10)
            ploteccmap(patchSign,[1.1 2.1],pixpermm); colorbar off
            hold on
            contour(xdom,ydom,im,[.5 .5],'k')
            axis image

            subplot(3,4,1)
            hold on,
            contour(xdom,ydom,im,[.5 .5],'k')

            subplot(3,4,2)
            hold on,
            contour(xdom,ydom,im,[.5 .5],'k')
            title('10. visual areas')
             end
%             catch
%                 disp('Couldn''t fuse/split areas... skipped...')
%             end
%                 
            %% Plot contours
            figure(10)
            subplot(3,4,11)
            contour(xdom,ydom,kmap_vert.*im,[-90:4:90],'r')
            hold on
            contour(xdom,ydom,kmap_hor.*im,[-90:4:90],'k')
            axis ij
            title('Red: Vertical Ret;  Black: Horizontal Ret')
            axis image
            xlim([xdom(1) xdom(end)]), ylim([ydom(1) ydom(end)])

            %% Get magnification factor images
            [JacIm prefAxisMF Distort] = getMagFactors(kmap_hor,kmap_vert,pixpermm);

            figure(10)
            subplot(3,4,12)
            plotmap(im.*sqrt(1./abs(JacIm)),[sqrt(.000001) sqrt(.003)],pixpermm);  %This doesn't work
            title('Mag fac (mm2/deg2)')

            dim = size(prefAxisMF);
            DdomX = 10:10:dim(2);
            DdomY = 10:10:dim(1);
            prefAxisMF = prefAxisMF(DdomY,DdomX);
            Distort = Distort(DdomY,DdomX);

            figure(10)
            subplot(3,4,12)
            hold on,
            contour(xdom,ydom,im,[.5 .5],'k')
            colorbar
            for i = 1:length(DdomX)
                for j = 1:length(DdomY)

                    xpart = 5*Distort(j,i)*cos(prefAxisMF(j,i)*pi/180);
                    ypart = 5*Distort(j,i)*sin(prefAxisMF(j,i)*pi/180);

                    if im(DdomY(j),DdomX(i))
                        hold on, plot([DdomX(i)-xpart DdomX(i)+xpart]*mmperpix,[DdomY(j)-ypart DdomY(j)+ypart]*mmperpix,'k')
                    end

                end
            end


        
        map_quality = questdlg('How do your maps look?','Map Quality','Good','Bad','Good');
        switch map_quality
            case 'Bad'
skip_flag =1 ;

            case 'Good'
                a = 1;
        end
    end
    
    disp('Finished processing sign maps...')
    %% SAVE AREA BORDER GENERATION FIGURE
    bordersFig = strcat('LP',num2str(LP),'_Area Border Generation.fig');
    old = cd(SaveDir);
    saveas(gcf,bordersFig,'fig')
    cd(old);
    
    %% Plot blood vessel overlays
    
    figure('Visible','off');
    set(gcf,'Position',[100, 100, 1500, 500]);
    
    % blood vessel picture
    subplot(1,3,1);
    imagesc(xdom,ydom,anatomypic)
    colormap gray
    hold on
    title(strcat('Anatomy'),'FontSize',12,'Interpreter','none');
    set(gca,'FontName','arial');
    set(gcf,'Color','w')
    xlabel('mm'); ylabel('mm')
    axis equal; axis tight
    hold on;
    contour(xdom,ydom,im,[.5 .5],'k','LineWidth',2);
    
    ratio=.2;
    aw = 1-ratio;  %anatomy weight of image (scalar)
    fw = ratio;  %anatomy weight of image (scalar)
    
    grayid = gray(numel(anatomypic));
    hsvid = hsv;
    %normalize overlay maps
    kmap_hor_overlay = kmap_hor_orig;
    kmap_hor_overlay = kmap_hor_overlay-min(kmap_hor_overlay(:));
    kmap_hor_overlay = kmap_hor_overlay/max(kmap_hor_overlay(:));
    kmap_hor_overlay = round(kmap_hor_overlay*49+1);
    
    kmap_vert_overlay = kmap_vert_orig;
    kmap_vert_overlay = kmap_vert_overlay-min(kmap_vert_overlay(:));
    kmap_vert_overlay = kmap_vert_overlay/max(kmap_vert_overlay(:));
    kmap_vert_overlay = round(kmap_vert_overlay*49+1);
    
    dim = size(kmap_hor_overlay);
    
    for i = 1:dim(1)
        for j = 1:dim(2)
            overlay(i,j,:) = fw*hsvid(kmap_hor_overlay(i,j),:) + aw*grayid(anatomypic(i,j),:);
        end
    end
    overlay = overlay/max(overlay(:));
    
    for i = 1:dim(1)
        for j = 1:dim(2)
            vertoverlay(i,j,:) = fw*hsvid(kmap_vert_overlay(i,j),:) + aw*grayid(anatomypic(i,j),:);
        end
    end
    
    vertoverlay = vertoverlay/max(vertoverlay(:));
    
    subplot(1,3,2)
    imagesc(xdom,ydom,overlay,[-50 50])
    title(strcat('Horizontal Retinotopy Overlay'),'FontSize',12,'Interpreter','none');
    set(gca,'FontName','arial');
    set(gcf,'Color','w')
    xlabel('mm'); ylabel('mm')
    axis equal; axis tight
    hold on;
    contour(xdom,ydom,im,[.5 .5],'k','LineWidth',2)
    
    subplot(1,3,3)
    imagesc(xdom,ydom,vertoverlay,[-50 50])
    title(strcat('Vertical Retinotopy Overlay'),'FontSize',12,'Interpreter','none');
    set(gca,'FontName','arial');
    set(gcf,'Color','w')
    xlabel('mm'); ylabel('mm')
    axis equal; axis tight
    hold on;
    contour(xdom,ydom,im,[.5 .5],'k','LineWidth',2)
    
    
    overlaysFig = strcat('LP',num2str(LP),'_Overlays.fig');
    old = cd(SaveDir);
    saveas(gcf,overlaysFig,'fig');
    cd(old)

    %% minor processing of patchsign to chaneg to -1 to 0 to 1
    lower_val = unique(patchSign);
    lower_val = lower_val(2);
    patchSign_sub = patchSign;
    patchSign_sub(patchSign_sub == lower_val) = -lower_val;
    patchSign = -sign(patchSign_sub);
    
    
    %% storing data for giving out
    
    maps.HorizontalRetinotopy = imresize(kmap_hor,[400 400]);
    maps.VerticalRetinotopy = imresize(kmap_vert,[400 400]);
    maps.VFS_raw = imresize(VFS, [400 400]);
    maps.VFS_processed = imresize(patchSign,[400 400]);
    maps.VFS_boundaries = imresize(bwmorph(abs(im),'remove'),[400 400]);
    maps.ReferenceImage = imresize(anatomypic,[400 400]);
    maps.Eccentricity = imresize(AreaInfo.kmap_rad,[400 400]);
    
    disp('Creating further things...')
    %% sawing overlay
    f2 =  figure('units','normalized','outerposition',[0 0 1 1],'Visible','off');
    imagesc(xdom,ydom,VFS)
    colormap('jet');
    hold on
    title(strcat('Overlay'),'FontSize',12,'Interpreter','none');
    set(gca,'FontName','arial');
    set(gcf,'Color','w')
    xlabel('mm'); ylabel('mm')
    axis equal; axis tight
    hold on;
    contour(xdom,ydom,im,[.5 .5],'k','LineWidth',2);
    
    sign_map = strcat('LP',num2str(LP),'_sign_map');
    saveas(f2,[sign_map '.jpg']);
    
    
    
    
    f1 =  figure('units','normalized','outerposition',[0 0 1 1],'Visible','on');
    ax1 = subplot(1,2,1);
    imagesc(xdom,ydom,anatomypic)
    colormap(ax1,'gray');
    hold on
    title(strcat('Overlay'),'FontSize',12,'Interpreter','none');
    set(gca,'FontName','arial');
    set(gcf,'Color','w')
    xlabel('mm'); ylabel('mm')
    axis equal; axis tight
    hold on;
    contour(xdom,ydom,im,[.5 .5],'k','LineWidth',2);
    
    ax2 = subplot(1,2,2);
    imagesc(xdom,ydom,patchSign)
    
    
    colormap(ax2,[1 0 0; 1 1 1; 0 0 1]);
    hold on
    title(strcat('Processed VFS'),'FontSize',12,'Interpreter','none');
    set(gca,'FontName','arial');
    set(gcf,'Color','w')
    xlabel('mm'); ylabel('mm')
    axis equal; axis tight
    hold on;
    contour(xdom,ydom,im,[.5 .5],'k','LineWidth',2);
    
    sign_map_overlay = strcat('LP',num2str(LP),'_sign_map_overlay');
    
    saveas(f1,[sign_map_overlay '.jpg']);
    
    
    %saving sign map
  pause(2)
    
    close all

VFS_raw = maps.VFS_raw;
VFS_processed = maps.VFS_processed;
VFS_boundaries = maps.VFS_boundaries;


        name = 'VFS_corrected.mat';
 
save(name, 'VFS_raw' ,'VFS_processed' ,'VFS_boundaries','maps')
old = cd(SaveDir);
save Additional_maps.mat maps
cd(old);


%% comparison....

subplot(1,2,1)
imagesc(VFS_raw)
axis square
colormap jet
title('New sign map')

subplot(1,2,2)
imagesc(old_map)
axis square
colormap jet
title('Old sign map')


function phi = phase_unwrap(psi, weight)
    if (nargin < 2) % unweighted phase unwrap
        % get the wrapped differences of the wrapped values
        dx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
        dy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
        rho = diff(dx, 1, 2) + diff(dy, 1, 1);
        
        % get the result by solving the poisson equation
        phi = solvePoisson(rho);
        
    else % weighted phase unwrap
        % check if the weight has the same size as psi
        if (~all(size(weight) == size(psi)))
            error('Argument error: Size of the weight must be the same as size of the wrapped phase');
        end
        
        % vector b in the paper (eq 15) is dx and dy
        dx = [wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
        dy = [wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
        
        % multiply the vector b by weight square (W^T * W)
        WW = weight .* weight;
        WWdx = WW .* dx;
        WWdy = WW .* dy;
        
        % applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
        WWdx2 = [zeros([size(psi,1),1]), WWdx];
        WWdy2 = [zeros([1,size(psi,2)]); WWdy];
        rk = diff(WWdx2, 1, 2) + diff(WWdy2, 1, 1);
        normR0 = norm(rk(:));
        
        % start the iteration
        eps = 1e-8;
        k = 0;
        phi = zeros(size(psi));
        while (~all(rk == 0))
            zk = solvePoisson(rk);
            k = k + 1;
            
            if (k == 1) pk = zk;
            else 
                betak = sum(sum(rk .* zk)) / sum(sum(rkprev .* zkprev));
                pk = zk + betak * pk;
            end
            
            % save the current value as the previous values
            rkprev = rk;
            zkprev = zk;
            
            % perform one scalar and two vectors update
            Qpk = applyQ(pk, WW);
            alphak = sum(sum(rk .* zk)) / sum(sum(pk .* Qpk));
            phi = phi + alphak * pk;
            rk = rk - alphak * Qpk;
            
            % check the stopping conditions
            if ((k >= numel(psi)) || (norm(rk(:)) < eps * normR0)) break; end;
        end
    end
end

function phi = solvePoisson(rho)
    % solve the poisson equation using dct
    dctRho = dct2(rho);
    [N, M] = size(rho);
    [I, J] = meshgrid([0:M-1], [0:N-1]);
    dctPhi = dctRho ./ 2 ./ (cos(pi*I/M) + cos(pi*J/N) - 2);
    dctPhi(1,1) = 0; % handling the inf/nan value
    
    % now invert to get the result
    phi = idct2(dctPhi);
    
end

% apply the transformation (A^T)(W^T)(W)(A) to 2D matrix
function Qp = applyQ(p, WW)
    % apply (A)
    dx = [diff(p, 1, 2), zeros([size(p,1),1])];
    dy = [diff(p, 1, 1); zeros([1,size(p,2)])];
    
    % apply (W^T)(W)
    WWdx = WW .* dx;
    WWdy = WW .* dy;
    
    % apply (A^T)
    WWdx2 = [zeros([size(p,1),1]), WWdx];
    WWdy2 = [zeros([1,size(p,2)]); WWdy];
    Qp = diff(WWdx2,1,2) + diff(WWdy2,1,1);
end




