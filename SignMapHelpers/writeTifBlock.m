function writeTifBlock(imgList, savePath)
%writeTifBlock Write TIF stack from image list
%   
%   This writes a single .tif image stack from a list of
%   The parameter `imgList` must be in order already. This is the order
%   that they will be added into the stack in.
%
%   Parameters
%       imgList : Cell array of image file paths. Each image should be
%           a .tif for an individual microscope image frame.
%       savePath : filepath (including name and extension). This will be
%           used to write the new TIF stack.
%
%   SignMapping -- Goard lab
%   https://github.com/ucsb-goard-lab/SignMapping
%   Written by DMM, Jan 05 2023
%
%--------------------------------------------------------------------------

    % Read the first image in. Get the metadata, etc., needed to create the
    % stack file before dealing with the remaining images.
    tmpImgPath = fullfile(imgList(1).folder, imgList(1).name);
    t = Tiff(tmpImgPath, 'r');
    im = read(t);
    
    % Write "big tif"... 'w8' is a larger format that can go past 4 GB
    % Otherwise, the TIF is not allowed to be larger than 4 GB
    bt = Tiff(savePath, 'w8');
    setTag(bt, "ImageLength", getTag(t,"ImageLength"));
    setTag(bt, "ImageWidth", getTag(t,"ImageWidth"));
    setTag(bt, "Photometric", getTag(t,"Photometric"));
    setTag(bt, "PlanarConfiguration", getTag(t,"PlanarConfiguration"));
    setTag(bt, "BitsPerSample", getTag(t,"BitsPerSample"));

    % Write the first frame
    write(bt, im);
    
    close(t);
    
    % The rest of the images will be written into the same file as a stack
    for fnum = 2:(size(imgList,1)-1)
    
        % New frame file path
        tmpImgPath = fullfile(imgList(fnum).folder, imgList(fnum).name);
        
        % Read the new frame
        t = Tiff(tmpImgPath, 'r');
        im = read(t);
    
        % Add the TIF directory for the new image inside of the file.
        % Essentially just adding a new index to the stack.
        writeDirectory(bt);
    
        % Keep the metadata w/ each image
        setTag(bt, "ImageLength", getTag(t,"ImageLength"));
        setTag(bt, "ImageWidth", getTag(t,"ImageWidth"));
        setTag(bt, "Photometric", getTag(t,"Photometric"));
        setTag(bt, "PlanarConfiguration", getTag(t,"PlanarConfiguration"));
        setTag(bt, "BitsPerSample", getTag(t,"BitsPerSample"));
    
        write(bt, im);
    
        close(t);
    
    end
    
    % Close the newly written file.
    close(bt);

end