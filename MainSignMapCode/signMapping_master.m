% signMapping_master

curr_script_dir = mfilename('fullpath');
parent_script_dir = curr_script_dir(1:end-17);
addpath(fullfile(parent_script_dir,'Juavinett et al 2017 Code'));

%% Usage

sm = SignMapper();

%% If you just want a single thing to run, use this:

sm.autoRunMapping();

%{
%% Advanced: Individual running of each module for troubleshooting/more control
data_loc = sm.getUserInput();

[data, stimdata] = sm.getData(data_loc);

[aziResp,altResp] = sm.separateResponseData(data,stimdata);

fourier_data(:,:,:,1) = fft(aziResp(:,:,:,1),[],3);
fourier_data(:,:,:,2) = fft(aziResp(:,:,:,2),[],3);
fourier_data(:,:,:,3) = fft(altResp(:,:,:,1),[],3);
fourier_data(:,:,:,4) = fft(altResp(:,:,:,2),[],3);

k = sm.findRetinotopicMap(fourier_data);

[azi,alt] = sm.getRetinotopicMap(fourier_data,k);
sm.displayMaps(azi,alt);

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
maps = sm.Juavinett2017_signMapping(azi,alt);

sm.saveSignMaps(maps);

%}


