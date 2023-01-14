function stackTIFs()
%stackTIFs Stack TIF images and split stimulus files
%   
%   Microscope image stacks are ~80GB for 10 repeats of the stimulus. These
%   cannot fit into memory silmultaniously. Though images will be collected
%   into an image stack, they will be put into several smaller image
%   stacks ("blocks"), each containing the image data for two repeated
%   presentations (1123 frames for a 10 FPS recording).
%   The images are initially written by the microscope as a sequence of
%   indidual .tif files. Just pick the first one for this script, and it
%   will find the images from that recording. This assumes that they will
%   be named [name]_000001 through [name]_999999 and that they are all in a
%   folder together. This will only save two repeats in each block (hard-
%   coded currently...).
%   This assumes that the images were aquired using:
%           https://github.com/ucsb-goard-lab/PsychtoolboxStimulusFramework
%   
%   Steps
%   1. Choose the first image in the sequence (e.g., '[name]_000001.tif')
%   2. Choose the stimfile for that recording
%
%   SignMapping -- Goard lab
%   https://github.com/ucsb-goard-lab/SignMapping
%   Written by DMM, Jan 05 2023
%
%--------------------------------------------------------------------------

    % Choose the first image in the sequence
    [nameStart, filepath] = uigetfile('~/*.tif', "Choose the first image in the sequence.");

    splitName = split(nameStart, '_');
    keepNameParts = splitName(1:end-1,1);
    baseName = string(join(keepNameParts));
    
    % Get all of the individual image files.
    % This will put them in numerical order
    imgList = dir(join([filepath, '\', baseName, '*.tif'], ''));

    % Load the stimulus timing file
    stimFile = uigetfile(string(fullfile(join([filepath,'*.mat'], '\'))), "Choose the stimulus file.");
    load(join([filepath, stimFile], '/'), "Stimdata");
    
    % Number of blocks
    % e.g., for 10 repeats, 5 tif stacks and 5 stim files will be created
    num_blocks = ceil(Stimdata.repeats / 2);

    % How many frames belong in each block?
    % If any frames are dropped, they'll end up missing from the final
    % block. Maybe that's a problem?
    block_size = ceil(size(imgList,1) / num_blocks);
    
    startF = 1; % first image frame
    cRep = 1; % the current stimulus repeat

    % Iterate through the blocks
    for b = 1:num_blocks
    
        
        endF = startF + (block_size-1);

        if endF > size(imgList,1)
            endF = size(imgList,1);
        end

        fprintf('Writing block %d of %d (frames %d through %d) \n',b,num_blocks,startF,endF)
    
        savePath = fullfile(filepath, join([baseName,'_stack_block',b,'.tif'], ''));
        writeTifBlock(imgList(startF:endF), savePath);
        
        startF = endF+1;

        % If there are an odd number of blocks & this is the final block,
        % then only save the Stim data for the final stimulus.
        if (mod(num_blocks, 2) && (b==num_blocks))
            addRepCount = 0;
        else
            addRepCount = 1;
        end

        % Which repeated presentations were included here?
        useReps = cRep:(cRep+addRepCount);
    
        % Create a new stimulus file that will only include the info for
        % the repeats included in the current image stack
        newStimdata = struct;
        newStimdata.repeats = length(useReps);
        newStimdata.blank_on = Stimdata.blank_on(useReps,:);
        newStimdata.blank_end = Stimdata.blank_end(useReps,:);
        newStimdata.mov_on = Stimdata.mov_on(useReps,:);
        newStimdata.mov_end = Stimdata.mov_end(useReps,:);
        newStimdata.off_time = Stimdata.off_time;
        newStimdata.on_time = Stimdata.on_time;

        stimSavePath = string(fullfile(filepath, join([baseName,'_stack_block',b,'_stim.mat'], '')));
        save(stimSavePath, 'newStimdata')

        cRep = cRep + 2;
    
    end

fprintf('Done')

end