function calcSignMap()
% Written 06Aug2019 KS


% Adding necessary paths to access the code for sign mapping
curr_script_dir = mfilename('fullpath');
parent_script_dir = curr_script_dir(1:end-length(mfilename)); % -11 to get rid of the filename
addpath(fullfile(parent_script_dir,'utils')); % Just make sure you add the Juavinett 2017 code folder

sm = SignMapper_disk(); % Create the sign mapping object

data_loc = sm.getUserInput(); % Get the input for everything

[data, stimdata] = sm.getData(data_loc); % Get and process data into a usable state

[aziResp,altResp] = sm.separateResponseData(stimdata); % Separate each recording into the cardinal directions, based on timestamps

% Run fourier transforms
fourier_data(:,:,:,1) = fft(aziResp(:,:,:,1),[],3);
fourier_data(:,:,:,2) = fft(aziResp(:,:,:,2),[],3);
fourier_data(:,:,:,3) = fft(altResp(:,:,:,1),[],3);
fourier_data(:,:,:,4) = fft(altResp(:,:,:,2),[],3);

% k = sm.findRetinotopicMap(fourier_data); % Find the correct harmonic for retinotopic maps

% [azi,alt] = sm.getRetinotopicMap(fourier_data,3); % Get the retinotopic map of determined harmonic
% sm.displayMaps(azi,alt); % Show maps

% Below allows you to manually redefine maps if the auto-chooser got the wrong harmonic
k = sm.manualFindRetinotopicMap(fourier_data);
[azi,alt] = sm.getRetinotopicMap(fourier_data, k);
sm.displayMaps(azi,alt);

mkdir('AdditionalSignMapMaterials'); % Additional save directory for supplemental stuff
maps = sm.Juavinett2017_signMapping(azi,alt); % Run the sign map creator, from the phase-maps

sm.saveSignMaps(maps); % Save everything
sm.exportSignMaps(maps); % Export overlay image

end