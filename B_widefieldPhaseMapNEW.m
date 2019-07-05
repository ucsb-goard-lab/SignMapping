function [aziPhase, altPhase] = B_widefieldPhaseMapNEW(aziFT_f,aziFT_b,altFT_u, altFT_d,k)
%% Step B: Phase map creation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written 12Apr2017 KS
% Last Updated:
% 30Aug2017 KS -Added header information
% 31Aug2017 KS -Automatically chooses the 2nd frequency bin, then asks if
% you want to further manually curate it

% Description of function

%%% Necessary Subfunctions %%%
% subfcn_frequencySearch         -Creates a contour map of every local max
%                                for display
% frequencySearcherGUI           -GUI for going through contour maps and
%                                choosing desired frequency
% subfcn_phaseMapper             -Takes the frequency bin of interest and
%                                creates a phase map using the raw data and
%                                the chosen k (k = frequency bin)

%%% Inputs %%%
% aziFT                          -Fourier transformed azimuth data
% altFT                          -Fourier transformed altitude data

%%% Outputs %%%
% aziPhase                       -Frequency selected phase data for azimuth
% altPhase                       -Frequency selected phase data for altitude
% widefield_phase_mapped.mat     -aziPhase and altPhase saved in a .mat file
% phase_maps.jpg                 -JPG of the altitude and azimuth phase maps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 0
    disp('Choose fourier-transformed azimuth data...')
    [filename, pathname] = uigetfile('.mat');
    load([pathname filename])
    
    disp('Choose fourier-transformed altitude data...')
    [filename, pathname] = uigetfile('.mat');
    load([pathname filename])
end

%% Creating contour maps

% First, let's try choosing the value closest to the expected stimulus
% frequency, which typically resides in bin 2 (the first peak harmonic
% frequency)

% k=4;
try %testing 2 different k's
aziPhase_f = subfcn_phaseMapper(aziFT_f,k(1));
aziPhase_b = subfcn_phaseMapper(aziFT_b,k(1));
altPhase_u = subfcn_phaseMapper(altFT_u,k(2));
altPhase_d = subfcn_phaseMapper(altFT_d,k(2));
catch
aziPhase_f = subfcn_phaseMapper(aziFT_f,k);
aziPhase_b = subfcn_phaseMapper(aziFT_b,k);
altPhase_u = subfcn_phaseMapper(altFT_u,k);
altPhase_d = subfcn_phaseMapper(altFT_d,k);
end

% 
%% from juanivett et al 2017
delay_hor = (exp(1i*aziPhase_f) + exp(1i*aziPhase_b));
delay_vert = (exp(1i*altPhase_u) + exp(1i*altPhase_d));

%Make delay go from 0 to pi and 0 to pi, instead of 0 to pi and 0 to -pi.
%The delay can't be negative.  If the delay vector is in the bottom two
%quadrants, it is assumed that the it started at -180.  The delay always
%pushes the vectors counter clockwise.

delay_hor = delay_hor + pi/2*(1-sign(delay_hor));
delay_vert = delay_vert + pi/2*(1-sign(delay_vert));

aziPhase = .5*(angle(exp(1i*(aziPhase_f-delay_hor))) - angle(exp(1i*(aziPhase_b-delay_hor))));
altPhase = .5*(angle(exp(1i*(altPhase_u-delay_vert))) - angle(exp(1i*(altPhase_d-delay_vert))));
% 

%% No delay version
%  aziPhase = -(angle(exp(1i*(aziPhase_f))) - angle(exp(1i*(aziPhase_b))));
%  altPhase = -(angle(exp(1i*(altPhase_u))) - angle(exp(1i*(altPhase_d))));
 

%% remove mean subtraction
 % aziPhase = aziPhase;% - mean(aziPhase(:));
 % altPhase= altPhase;% - mean(altPhase(:));
% 
% aziPhase = aziPhase_f - aziPhase_b;
% altPhase = altPhase_u - altPhase_d;
%%
bw = ones(size(aziPhase));

%radians to degrees
% delay_hor = delay_hor*180/pi.*bw;
aziPhase = aziPhase*180/pi.*bw;
% delay_vert = delay_vert*180/pi.*bw;
altPhase = altPhase*180/pi.*bw;
%%%


aziPhase =- phase_unwrap(aziPhase);
altPhase =- phase_unwrap(altPhase);
end

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




