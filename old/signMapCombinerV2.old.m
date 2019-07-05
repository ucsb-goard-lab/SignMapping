function [aziResp_f_combined, aziResp_b_combined, altResp_u_combined, altResp_d_combined] = signMapCombinerV2(fn ,pn, N_recordings)

if nargin == 0
N_recordings = inputdlg('How many recordings are you combining?');
N_recordings = str2num(N_recordings{1});
for i = 1:N_recordings
    fprintf(['Choose your widefield_FT_data.mat file for recording #%d/' num2str(N_recordings) '\n'],i)
    [fn{i}, pn{i}] = uigetfile('.mat');  
end

end

for i = 1:N_recordings
    load([pn{i} 'widefield_FT_data.mat'],'aziResp_f');
    aziResp_f_combined(:,:,:,i) = aziResp_f;
        load([pn{i} 'widefield_FT_data.mat'],'aziResp_b');

        aziResp_b_combined(:,:,:,i) = aziResp_b;

    load([pn{i} 'widefield_FT_data.mat'],'altResp_u');
    altResp_u_combined(:,:,:,i) = altResp_u;
    load([pn{i} 'widefield_FT_data.mat'],'altResp_d');

    altResp_d_combined(:,:,:,i) = altResp_d;
end

aziResp_f_combined = mean(aziResp_f_combined,4);
aziResp_b_combined = mean(aziResp_b_combined,4);
altResp_u_combined = mean(altResp_u_combined,4);
altResp_d_combined = mean(altResp_d_combined,4);