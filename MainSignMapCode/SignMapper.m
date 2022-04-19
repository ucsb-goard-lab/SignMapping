classdef SignMapper < handle
    
    properties (Constant = true)
        harmonic_pool = [2 3 4 5]; % Pool of ft harmonics to check, probably don't need more...
        horz_factor   =  145/360;  % These are simply for conversion into degrees
        vert_factor   = 124/360;
        pixpermm      = 4;        % Info regarding the screen
        SaveDir       = pwd;       % Parent save directory
    end
    
    properties
	    fs = 10; % imaging frame rate
        stimdata cell  % Stimulus data file for timestamps
        data     cell  % DFF data
        
        ref_img  double % Surface reference image
        maps     struct % Finished maps
        
        autorun_flag logical = false; % For auto-running
        
        n_recordings double
    end
    
    methods
        function obj = SignMapper()
            % Nothing to initialize
        end
        
        function autoRunMapping(obj) % This just puts everything together for autorun
            obj.autorun_flag = true;
            
            data_loc = obj.getUserInput(); % Get the input for everything
            
            [data, stimdata] = obj.getData(data_loc); % Get and process data into a usable state
            
            [aziResp,altResp] = obj.separateResponseData(data,stimdata); % Separate each recording into the cardinal directions, based on timestamps
            
            % Run fourier transforms
            fourier_data(:,:,:,1) = fft(aziResp(:,:,:,1),[],3);
            fourier_data(:,:,:,2) = fft(aziResp(:,:,:,2),[],3);
            fourier_data(:,:,:,3) = fft(altResp(:,:,:,1),[],3);
            fourier_data(:,:,:,4) = fft(altResp(:,:,:,2),[],3);
            
            
            k = obj.findRetinotopicMap(fourier_data); % Find the correct harmonic for retinotopic maps
            [azi,alt] = obj.getRetinotopicMap(fourier_data,k); % Get the retinotopic map of determined harmonic
            
            mkdir('AdditionalSignMapMaterials'); % Additional save directory for supplemental stuff
            maps = obj.Juavinett2017_signMapping(azi,alt); % Run the sign map creator, from the phase-maps
            obj.saveSignMaps(maps); % Save everything
            obj.exportSignMaps(maps); % Export the overlay image
        end
        
        function data_loc = getUserInput(obj) % Get the user input to find the recordings and such
            n_recordings = inputdlg('How many recording blocks do you have?');
            obj.n_recordings = str2double(n_recordings{1});
            
            data_loc = cell(obj.n_recordings,2,3); % recordings x filename/pathname x data/Stimdata/ref_img
            
            for ii = 1:obj.n_recordings % Get the data
                obj.msgPrinter(sprintf('Choose multi-page.tif file for recording #%d/%d \n',ii,obj.n_recordings))
                [data_loc{ii,1,1}, data_loc{ii,2,1}] = uigetfile('.tif');
            end
            
            for ii = 1:obj.n_recordings % Get the stimulus data
                obj.msgPrinter(sprintf('Choose your stimulus data file for recording #%d/%d \n',ii,obj.n_recordings))
                [data_loc{ii,1,2},data_loc{ii,2,2}] = uigetfile('.mat');
            end
            
            % Get the reference image
            obj.msgPrinter(sprintf('Lastly, choose your reference image for overlay\n'))
            [data_loc{1,1,3}, data_loc{1,2,3}] = uigetfile({'*.jpg;*.png;*.gif;*.tif','All Image Files';...
                '*.*','All Files' });         
        end
        
        function [data, stimdata] = getData(obj,data_loc)
            obj.n_recordings = size(data_loc,1);
            
            % Get the stimulus data
            stimdata = cell(1,obj.n_recordings);
            for r = 1:obj.n_recordings
                stimdata{r} = importdata([data_loc{r,2,2},data_loc{r,1,2}]);
            end
            
            % Get the recording data
            data = cell(1,obj.n_recordings);
            for r = 1:obj.n_recordings
                obj.msgPrinter(sprintf('Processing recording block #%d/%d \n',r,obj.n_recordings));
                data{r} = obj.widefieldDFF_abridged(data_loc{r,2,1},data_loc{r,1,1});
            end
            
            % Get and process the reference image
            ref_img = imread([data_loc{1,2,3} data_loc{1,1,3}]);
            try
                ref_img_mat = rgb2gray(ref_img);
            catch
                ref_img_mat = ref_img;
            end
            min_sz = min(size(ref_img_mat)); % Resizing the reference image to match recordings
            ref_img_mat = ref_img_mat(1:min_sz,1:min_sz); % Turning it into a square, just in case
            
            scaleFactor = size(data{1},1)./size(ref_img_mat,1);
            
            ref_img_sc = imresize(ref_img_mat,scaleFactor);
            
            % Store all the values into the corresponding property
            obj.stimdata = stimdata;
            obj.data = data;
            obj.ref_img = ref_img_sc;
        end
        
        function [aziResp, altResp] = separateResponseData(obj,raw_data,raw_stimdata)
            for r = 1:obj.n_recordings
                obj.msgPrinter(sprintf('Separating recording block #%d/%d \n',r,obj.n_recordings));
                
                data = raw_data{r}; % not the best way of doing this, but it lets us keep everything from old code
                stimdata = raw_stimdata{r};
                
                % Get stimulus data
                repeats     = stimdata.n_repeats; 
                on_frames   = stimdata.on_time*obj.fs;
                off_frames  = stimdata.post_time*obj.fs;
                
                sweep_start = round(stimdata.mov_on*obj.fs);
                blank_start = round(stimdata.blank_on*obj.fs);
                % Preallocate the matrices
                azi_on_fResp  = zeros(size(data,1),size(data,2),on_frames,repeats,'single');
                azi_on_bResp  = zeros(size(data,1),size(data,2),on_frames,repeats,'single');
                
                azi_off_fResp = zeros(size(data,1),size(data,2),off_frames,repeats,'single');
                azi_off_bResp = zeros(size(data,1),size(data,2),off_frames,repeats,'single');
                
                alt_on_uResp  = zeros(size(data,1),size(data,2),on_frames,repeats,'single');
                alt_on_dResp  = zeros(size(data,1),size(data,2),on_frames,repeats,'single');
                
                alt_off_uResp = zeros(size(data,1),size(data,2),off_frames,repeats,'single');
                alt_off_dResp = zeros(size(data,1),size(data,2),off_frames,repeats,'single');
                % extract responses based on timestamps
                obj.msgPrinter('     (1/3) Extracting responses\n')
                for rep = 1:repeats
                    azi_off_fResp(:,:,:,rep)  = data(:,:,blank_start(rep,1)+1:blank_start(rep,1) + off_frames);
                    azi_on_fResp(:,:,:,rep) = data(:,:,sweep_start(rep,1)+1:sweep_start(rep,1)+ on_frames);
                    
                    azi_off_bResp(:,:,:,rep)  = data(:,:,blank_start(rep,2)+1:blank_start(rep,2)+ off_frames);
                    azi_on_bResp(:,:,:,rep) = data(:,:,sweep_start(rep,2)+1:sweep_start(rep,2)+ on_frames);
                    
                    alt_off_uResp(:,:,:,rep)  = data(:,:,blank_start(rep,3)+1:blank_start(rep,3)+ off_frames);
                    alt_on_uResp(:,:,:,rep) = data(:,:,sweep_start(rep,3)+1:sweep_start(rep,3)+ on_frames);
                    
                    alt_off_dResp(:,:,:,rep)  = data(:,:,blank_start(rep,4)+1:blank_start(rep,4)+ off_frames);
                    alt_on_dResp(:,:,:,rep) = data(:,:,sweep_start(rep,4)+1:sweep_start(rep,4)+ on_frames);
                end
               
                obj.msgPrinter('     (2/3) Baseline subtracting\n')     
                % meaning off responses across frames per pixel, preserving reps
                m_azi_off_fResp = squeeze(mean(azi_off_fResp,3));
                m_azi_off_bResp = squeeze(mean(azi_off_bResp,3));
                
                m_alt_off_uResp = squeeze(mean(alt_off_uResp,3));
                m_alt_off_dResp = squeeze(mean(alt_off_dResp,3));
                
                % nancheck
                m_azi_off_fResp(isnan(m_azi_off_fResp)) = 0;
                m_azi_off_bResp(isnan(m_azi_off_bResp)) = 0;
                m_alt_off_uResp(isnan(m_alt_off_uResp)) = 0;
                m_alt_off_dResp(isnan(m_alt_off_dResp)) = 0;
                
                
                % Preallocating
                aziResp = zeros(size(data,1),size(data,2),on_frames,repeats,2,'single'); % 2 for 2 directions
                altResp = zeros(size(data,1),size(data,2),on_frames,repeats,2,'single');
                
                % subtracting off responses from corresponding on responses
                for rep = 1:repeats
                    aziResp(:,:,:,rep,1) = azi_on_fResp(:,:,:,rep) - m_azi_off_fResp(:,:,rep);
                    aziResp(:,:,:,rep,2) = azi_on_bResp(:,:,:,rep) - m_azi_off_bResp(:,:,rep);
                    altResp(:,:,:,rep,1) = alt_on_uResp(:,:,:,rep) - m_alt_off_uResp(:,:,rep);
                    altResp(:,:,:,rep,2) = alt_on_dResp(:,:,:,rep) - m_alt_off_dResp(:,:,rep);
                end
                
               % Meaning responses in both directions
                 obj.msgPrinter('     (3/3) Meaning responses\n')
                % mean across reps in each direction
                aziResp_all(:,:,:,:,r) =  squeeze(mean(aziResp,4)); %meaning within the recording
                altResp_all(:,:,:,:,r) =   squeeze(mean(altResp,4));
            end
            
            aziResp = mean(aziResp_all,5); % overwriting aziResp, for the final one
            altResp = mean(altResp_all,5);
            img_dims = size(altResp);
            
            aziResp =  reshape(aziResp,img_dims(1),img_dims(2),[],2); % now, combining across recordingz
            altResp =  reshape(altResp,img_dims(1),img_dims(2),[],2);
        end
        
        function k = findRetinotopicMap(obj,fourier_data) % Find the correct harmonic for retinotopic mapping
            idx=1;
            while true
                if idx == length(obj.harmonic_pool) % If you reach the end, something might be wrong, assign to first one
                    idx=1; % assign to the first one
                    break
                end
                [azi,alt] = obj.getRetinotopicMap(fourier_data,obj.harmonic_pool(idx)); % Get the corresponding maps
                s = sum([skewness(azi(:)),skewness(alt(:))]); % the real map is very positively skewed
                if abs(s) > 0.01 % If skewness value is really low, it probably means that it's not a retinotopic map (normal dist)
                    break
                end
                idx=idx+1; % move on to the next harmonic
            end
            k = obj.harmonic_pool(idx); % Get the lowest harmonic which isn't normally distributed
        end

        function [azimuthMap, altitudeMap] = getRetinotopicMap(obj,fourier_data,h) % 
            phaseMap = zeros(size(fourier_data,1),size(fourier_data,2),size(fourier_data,4)); % Preallocate the phaseMaps
            for f = 1:size(fourier_data,4) % Minor preprocessing of phase maps (mean subtraction and rotation)
                phaseMap(:,:,f) = obj.phaseMapChooser(fourier_data(:,:,:,f),h);
            end

            delay_hor = (exp(1i*phaseMap(:,:,1)) + exp(1i*phaseMap(:,:,2))); % Calculating delay
            delay_vert = (exp(1i*phaseMap(:,:,3)) + exp(1i*phaseMap(:,:,4)));
            
            delay_hor = delay_hor + pi/2*(1-sign(delay_hor));
            delay_vert = delay_vert + pi/2*(1-sign(delay_vert));
            
            aziPhase = .5*(angle(exp(1i*(phaseMap(:,:,1)-delay_hor))) - angle(exp(1i*(phaseMap(:,:,2)-delay_hor)))); % Combine opposite directions
            altPhase = .5*(angle(exp(1i*(phaseMap(:,:,3)-delay_vert))) - angle(exp(1i*(phaseMap(:,:,4)-delay_vert))));
            
            azimuthMap = -phase_unwrap(aziPhase*180/pi); % Convert to radians, negative to correct sign
            altitudeMap = -phase_unwrap(altPhase*180/pi);
            
            function phi = phase_unwrap(psi, weight) % From somewhere else
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
            end
        end

        function k = manualFindRetinotopicMap(obj,fourier_data) % Manual locator of retinotopic maps
            figure;
            set(gcf,'Position',[500 200 1000 500])
            for ii = 1:length(obj.harmonic_pool)
                [azi,alt] = obj.getRetinotopicMap(fourier_data,obj.harmonic_pool(ii)); % Get maps for multiple harmonics
                subplot(2,2,ii)
                imagesc([azi alt]); colormap hsv; % show them
                box off, axis off
                title(sprintf('Map #%d',ii))
            end
            
            idx = input('Choose your map #: '); % Manually choose harmonic
            k = obj.harmonic_pool(idx);
        end
        
        function displayMaps(obj,azi,alt) % Just for visualzation of maps
            figure('units','normalized','outerposition',[0.2 0.2 0.6 0.5])
            subplot(1,2,1)
            imagesc(azi)
            colormap hsv
            axis square
            axis off
            
            subplot(1,2,2)
            imagesc(alt)
            colormap hsv
            axis square
            axis off
        end
      
        function saveSignMaps(obj,maps) % Handling the saving of the maps
            if nargin < 2
                maps = obj.maps;
            end
            
            obj.msgPrinter('Saving main sign mapping materials\n')
            VFS_raw = maps.VFS_raw;
            VFS_processed = maps.VFS_processed;
            VFS_boundaries = maps.VFS_boundaries;
            
            save('VFS_maps.mat','VFS_raw','VFS_processed','VFS_boundaries');
            
            save('additional_maps.mat','maps');
        end
        
        function exportSignMaps(obj,maps)
            if nargin < 2
                maps = obj.maps;
            end
            
            figure('units','normalized','outerposition',[0.22 0 0.6 1]) % standard widescreen 1920x1080p
            
            refimg = maps.ReferenceImage;
            bound = maps.VFS_boundaries;
            refimg(bound) = max(refimg(:))*1.1;
            imagesc(refimg)
            axis off
            axis square
            
            axes('Position',[0.155 0.775 0.15 0.15])
            imagesc(maps.VFS_raw)
            axis square
            axis off
    
            ax = findobj(gcf,'Type','axes');
            ax(1).Colormap = jet;
            ax(2).Colormap = gray;
            
            
            saveas(gcf,'overlay_map.jpg') 
            close
        end
        
        % Don't open this unless you want to have a bad time, adapted from
        % Juavinett, A. L., Nauhaus, I., Garrett, M. E., Zhuang, J., & Callaway, E. M. (2017).
        % Automated identification of mouse visual areas with intrinsic signal imaging.
        % Nature protocols, 12(1), 32.
        function maps = Juavinett2017_signMapping(obj,aziPhase,altPhase)
            skip_flag = 0;
            obj.msgPrinter('Processing sign map\n')
            while true
                aziPhase = imgaussfilt(aziPhase,2); % filter maps
                altPhase = imgaussfilt(altPhase,2);

                aziPhase = aziPhase*obj.horz_factor; % scale for screen
                altPhase = altPhase*obj.vert_factor;
                
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
                
                
                %  kmap_vert = downsample(kmap_vert_orig,downsample_val);
                
                kmap_hor_orig = rot90(kmap_hor_orig);
                kmap_vert_orig = rot90(kmap_vert_orig);
                
                
                %% Compute visual field sign map
                mmperpix = 1/obj.pixpermm;
                
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
                plotmap(imdum,[.1 2.1],obj.pixpermm);
                colorbar off
                axis image
                title(['4. +/-1.5xSig = ' num2str(threshSeg)])
                
                patchSign = getPatchSign(imseg,VFS);
                
                figure(10), subplot(3,4,5),
                plotmap(patchSign,[1.1 2.1],obj.pixpermm);
                title('watershed')
                colorbar off
                title('5. Threshold patches')
                
                id = find(patchSign ~= 0);
                patchSign(id) = sign(patchSign(id) - 1);
                
                SE = strel('disk',2,0);
                imseg = imopen(imseg,SE);
                
                patchSign = getPatchSign(imseg,VFS);
                
                figure(10), subplot(3,4,6),
                ploteccmap(patchSign,[1.1 2.1],obj.pixpermm);
                title('watershed')
                colorbar off
                title('6. "Open" & set boundary')
                
                %% Make boundary of visual cortex
                
                %First pad the image with zeros because the "imclose" function does this
                %weird thing where it tries to "bleed" to the edge if the patch near it
                
                Npad = 30;  %Arbitrary padding value.  May need more depending on image size and resolution
                dim = size(imseg);
                imsegpad = [zeros(dim(1),Npad) imseg zeros(dim(1),Npad)];
                dim = size(imsegpad);
                imsegpad = [zeros(Npad,dim(2)); imsegpad; zeros(Npad,dim(2))];
                
                SE = strel('disk',10,0);
                imbound = imclose(imsegpad,SE);
                
                imbound = imfill(imbound); %often unnecessary, but sometimes there are small gaps need filling
                
                % SE = strel('disk',5,0);
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
                ploteccmap(patchSign,[1.1 2.1],obj.pixpermm);
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
                ploteccmap(AreaInfo.kmap_rad.*im,[0 45],obj.pixpermm);
                hold on
                contour(xdom,ydom,im,[.5 .5],'k')
                
                axis image
                title('8. Eccentricity map')
                
                
                
                try
                    %% ID redundant patches and split them (criterion #2)
                    if ~skip_flag
                        im = splitPatchesX(im,kmap_hor,kmap_vert,AreaInfo.kmap_rad,pixpermm); %% decreased minimum patch size thing from 0.01 to 0.001 to keep areas like AM...
                    end
                    
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
                    
                    [im fuseflag] = fusePatchesX(im,kmap_hor,kmap_vert,obj.pixpermm);
                    
                    [patchSign areaSign] = getPatchSign(im,VFS);
                    
                    figure(10), subplot(3,4,9),
                    ploteccmap(im.*AreaInfo.kmap_rad,[0 45],obj.pixpermm);
                    colorbar
                    title('9. Split redundant patches. Fuse exclusive patches.')
                    
                    figure(10), subplot(3,4,10)
                    ploteccmap(patchSign,[1.1 2.1],obj.pixpermm); colorbar off
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
                catch
                    obj.msgPrinter('     Unable to fuse/split patches, skipped\n')
                end
                
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
                [JacIm prefAxisMF Distort] = getMagFactors(kmap_hor,kmap_vert,obj.pixpermm);
                
                figure(10)
                subplot(3,4,12)
                plotmap(im.*sqrt(1./abs(JacIm)),[sqrt(.000001) sqrt(.003)],obj.pixpermm);  %This doesn't work
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
                
                
                if ~obj.autorun_flag
                    map_quality = questdlg('How do your maps look?','Map Quality','Good','Bad','Good');
                    switch map_quality
                        case 'Bad'
                            skip_flag =1;
                            % LP_val = inputdlg('Recording quality low, need to LP filter the retinotopic maps, choose a LP value [0 ,2]:');
                            %LP = str2num(LP_val{1});
                            %close all
                        case 'Good'
                            break
                    end
                else
                    break
                end
            end
            
            %% minor processing of patchsign to change to -1 to 0 to 1
            anatomypic = rot90(obj.ref_img)+1; %% reference image
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
            
            obj.maps = maps;
            close all % clean up
        end
        
    end
    
    methods (Access = private)
        function msgPrinter(obj, msg)
            fprintf(msg)
        end
        
        function dff = widefieldDFF_abridged(obj, pn, fn) % Bundled version for sign mapping
            %% Extracting basic image info (resolution, frames)
            obj.msgPrinter('     (1/4) Getting image info\n')
            info = imfinfo([pn fn]);
            num_images = length(info);
            
            x_pixels = info(1).Width;
            y_pixels = info(1).Height;
            
            
            %preallocation
            dff = zeros(y_pixels, x_pixels, num_images,'single');
            imageRaw = zeros(y_pixels,x_pixels,num_images,'single');

            %% Reading images into MATLAB
            obj.msgPrinter('     (2/4) Reading images\n')
            for f = 1:num_images
                imageRaw(:,:,f) = imread([pn fn], 'Index', f, 'Info', info);
            end
            
            %% Photobleaching check, this is gonna be displayed, and the figure saved.
            frame_F = squeeze(mean(mean(imageRaw,1),2));
            
            plot(frame_F,'linewidth',2)
            F_fit = polyfit([1:length(frame_F)],frame_F',1);
            PhotoBl = round((F_fit(1)*length(frame_F))/F_fit(2)*100);
            ylim([0 max(frame_F)*1.25])
            title(['Fluorescence timecourse, Photobleaching = ' num2str(PhotoBl) '%'])
            xlabel('Frame #')
            ylabel('Raw fluorescence')
            set(gcf,'color',[1 1 1])
            saveas(gcf,'Fluorescence_timecourse')
            close
            
            disp(['          Photobleaching: ' num2str(PhotoBl) '%'])
            
            %% Calculating F0 map
            obj.msgPrinter('     (3/4) Calculating F0 map\n')
            F0 = zeros(y_pixels, x_pixels,'single');
            for p = 1:x_pixels
                for j = 1:y_pixels
                    F0(j,p) = (median(imageRaw(j,p,:),3));
                end
            end
            
            %% Calculating dff
            obj.msgPrinter('     (4/4) Calculating DFF\n')
            block_size = 1000;
            num_blocks = ceil(length(info)/block_size);
            
            for b = 1:num_blocks
                idx_vec = (b-1)*block_size+1:min(b*block_size,length(info));
                curr_image = imageRaw(:,:,idx_vec);
                
                DFF_block = calculateDFF(F0,curr_image);
                
                dff(:,:,idx_vec) = DFF_block;
            end
            
            function [DFF_block] = calculateDFF(F0,curr_frames)
                
                DFF_block = zeros(size(curr_frames,1),size(curr_frames,2),size(curr_frames,3));
                
                for frme = 1:size(curr_frames,3)
                    DFF_block(:,:,frme) = ((curr_frames(:,:,frme) - F0)./F0)*100;
                end
            end
        end
        
        function phaseMap = phaseMapChooser(obj, ft_data, k) % k is the harmonic
            phaseMap = angle(ft_data(:,:,k));
%             phaseMap = rot90(phaseMap);
            phaseMap = phaseMap - mean(phaseMap(:));
        end
    end
end
