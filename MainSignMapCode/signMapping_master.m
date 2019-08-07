% This script lets you autorun or just use each part individually

% Written 06Aug2019 KS
% Updated

% Adding necessary paths to access the code for sign mapping
curr_script_dir = mfilename('fullpath');
parent_script_dir = curr_script_dir(1:end-length(mfilename)); % -18 to get rid of the filename
addpath(fullfile(parent_script_dir,'Juavinett et al 2017 Code'));

sm = SignMapper(); % Create the sign mapping object

data_loc = sm.getUserInput(); % Get the input for everything

[data, stimdata] = sm.getData(data_loc); % Get and process data into a usable state

[aziResp,altResp] = sm.separateResponseData(data,stimdata); % Separate each recording into the cardinal directions, based on timestamps

% Run fourier transforms
fourier_data(:,:,:,1) = fft(aziResp(:,:,:,1),[],3);
fourier_data(:,:,:,2) = fft(aziResp(:,:,:,2),[],3);
fourier_data(:,:,:,3) = fft(altResp(:,:,:,1),[],3);
fourier_data(:,:,:,4) = fft(altResp(:,:,:,2),[],3);

k = sm.findRetinotopicMap(fourier_data); % Find the correct harmonic for retinotopic maps

[azi,alt] = sm.getRetinotopicMap(fourier_data,k); % Get the retinotopic map of determined harmonic
sm.displayMaps(azi,alt); % Show maps

% Below allows you to manually redefine maps if the auto-chooser got the wrong harmonic
while true
    goodmap = questdlg('Do your maps look good?','Map quality','Yes','No','Yes');
    close
    
    switch goodmap
        case 'No'
            k = sm.manualFindRetinotopicMap(fourier_data);
            [azi,alt] = sm.getRetinotopicMap(fourier_data,k);
            sm.displayMaps(azi,alt);
        case 'Yes'
            break
    end
end

mkdir('AdditionalSignMapMaterials'); % Additional save directory for supplemental stuff
maps = sm.Juavinett2017_signMapping(azi,alt); % Run the sign map creator, from the phase-maps

sm.saveSignMaps(maps); % Save everything

