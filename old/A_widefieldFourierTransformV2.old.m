function [aziFT_f, aziFT_b, altFT_u, altFT_d] = A_widefieldFourierTransformV2(aziResp_f,aziResp_b,altResp_u,altResp_d)
%% Widefield Fourier Transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Written 30Aug2017 KS
% Last Updated: 
 
% Conducts a fourier transform on azimuth and altitude response data
 
%%% Necessary Subfunctions %%%
% None

%%% Inputs %%% 
% aziResp                       Azimuth response data
% altResp                       Altitude response data

%%% Outputs %%%
% aziFT                         Structure of Fourier transformed azimuth
%                               data (real and imaginary)
% altFT                         Structure of Fourier transformed altitude
%                               data (real and imaginary)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Azimuth
disp('(1/2) Calculating azimuth map Fourier transform...')

aziFT_combined_f = single(fft(aziResp_f,[],3));
aziMag_f = double(abs(aziFT_combined_f/size(aziFT_combined_f,3))); % real component

aziFT_f.mag   = aziMag_f;
aziFT_f.angle = angle(aziFT_combined_f); % imaginary component

aziFT_combined_b = single(fft(aziResp_b,[],3));
aziMag_b = double(abs(aziFT_combined_b/size(aziFT_combined_b,3))); % real component

aziFT_b.mag   = aziMag_b;
aziFT_b.angle = angle(aziFT_combined_b); % imaginary component

% Altitude 
disp('(2/2) Calculating altitude map Fourier transform...')

altFT_combined_u = single(fft(altResp_u,[],3));
altMag_u = double(abs(altFT_combined_u/size(altFT_combined_u,3)));

altFT_u.mag   = altMag_u; 
altFT_u.angle = angle(altFT_combined_u);

altFT_combined_d = single(fft(altResp_d,[],3));
altMag_d = double(abs(altFT_combined_d/size(altFT_combined_d,3)));

altFT_d.mag   = altMag_d; 
altFT_d.angle = angle(altFT_combined_d);