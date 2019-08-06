function [ft_phase] = A_widefieldFourierTransformNEW(resp_matrix)
%% Widefield Fourier Transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Written 30Aug2017 KS
% Last Updated: 02Apr2019 KS Added phase wrapping to smooth the maps!!
 
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
%phase_wrapped = zeros(size(resp_matrix,1),size(resp_matrix,2),size(resp_matrix,3));
ft_phase = zeros(size(resp_matrix,1),size(resp_matrix,2),size(resp_matrix,3));

for x = 1:size(resp_matrix,1)
    for y =1:size(resp_matrix,2)
        ft_phase(y,x,:) = angle(fft(resp_matrix(y,x,:)));
    end
end

% for img = 1:size(ft_phase,3)
%     ft_phase(:,:,img) = unwrap_phase(phase_wrapped(:,:,img));
% end

%aziFT_combined_f = single(fft(aziResp_f,[],3));
%ft_phase = angle(FT_output); % imaginary component
%b

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fast unwrapping 2D phase image using the algorithm given in:                 %
%     M. A. Herráez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat,             %
%     "Fast two-dimensional phase-unwrapping algorithm based on sorting by     %
%     reliability following a noncontinuous path", Applied Optics, Vol. 41,    %
%     Issue 35, pp. 7437-7444 (2002).                                          %
%                                                                              %
% If using this code for publication, please kindly cite the following:        %
% * M. A. Herraez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, "Fast         %
%   two-dimensional phase-unwrapping algorithm based on sorting by reliability %
%   following a noncontinuous path", Applied Optics, Vol. 41, Issue 35,        %
%   pp. 7437-7444 (2002).                                                      %
% * M. F. Kasim, "Fast 2D phase unwrapping implementation in MATLAB",          %
%   https://github.com/mfkasim91/unwrap_phase/ (2017).                         %
%                                                                              %
% Input:                                                                       %
% * img: The wrapped phase image either from -pi to pi or from 0 to 2*pi.      %
%        If there are unwanted regions, it should be filled with NaNs.         %
%                                                                              %
% Output:                                                                      %
% * res_img: The unwrapped phase with arbitrary offset.                        %
%                                                                              %
% Author:                                                                      %
%     Muhammad F. Kasim, University of Oxford (2017)                           %
%     Email: firman.kasim@gmail.com                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res_img = unwrap_phase(img)
    [Ny, Nx] = size(img);

    % get the reliability
    reliability = get_reliability(img); % (Ny,Nx)

    % get the edges
    [h_edges, v_edges] = get_edges(reliability); % (Ny,Nx) and (Ny,Nx)

    % combine all edges and sort it
    edges = [h_edges(:); v_edges(:)];
    edge_bound_idx = Ny * Nx; % if i <= edge_bound_idx, it is h_edges
    [~, edge_sort_idx] = sort(edges, 'descend');

    % get the indices of pixels adjacent to the edges
    idxs1 = mod(edge_sort_idx - 1, edge_bound_idx) + 1;
    idxs2 = idxs1 + 1 + (Ny - 1) .* (edge_sort_idx <= edge_bound_idx);

    % label the group
    group = reshape([1:numel(img)], Ny*Nx, 1);
    is_grouped = zeros(Ny*Nx,1);
    group_members = cell(Ny*Nx,1);
    for i = 1:size(is_grouped,1)
        group_members{i} = i;
    end
    num_members_group = ones(Ny*Nx,1);

    % propagate the unwrapping
    res_img = img;
    num_nan = sum(isnan(edges)); % count how many nan-s and skip them
    for i = num_nan+1 : length(edge_sort_idx)
        % get the indices of the adjacent pixels
        idx1 = idxs1(i);
        idx2 = idxs2(i);

        % skip if they belong to the same group
        if (group(idx1) == group(idx2)) continue; end

        % idx1 should be ungrouped (swap if idx2 ungrouped and idx1 grouped)
        % otherwise, activate the flag all_grouped.
        % The group in idx1 must be smaller than in idx2. If initially
        % group(idx1) is larger than group(idx2), then swap it.
        all_grouped = 0;
        if is_grouped(idx1)
            if ~is_grouped(idx2)
                idxt = idx1;
                idx1 = idx2;
                idx2 = idxt;
            elseif num_members_group(group(idx1)) > num_members_group(group(idx2))
                idxt = idx1;
                idx1 = idx2;
                idx2 = idxt;
                all_grouped = 1;
            else
                all_grouped = 1;
            end
        end

        % calculate how much we should add to the idx1 and group
        dval = floor((res_img(idx2) - res_img(idx1) + pi) / (2*pi)) * 2*pi;

        % which pixel should be changed
        g1 = group(idx1);
        g2 = group(idx2);
        if all_grouped
            pix_idxs = group_members{g1};
        else
            pix_idxs = idx1;
        end

        % add the pixel value
        if dval ~= 0
            res_img(pix_idxs) = res_img(pix_idxs) + dval;
        end

        % change the group
        len_g1 = num_members_group(g1);
        len_g2 = num_members_group(g2);
        group_members{g2}(len_g2+1:len_g2+len_g1) = pix_idxs;
        group(pix_idxs) = g2; % assign the pixels to the new group
        num_members_group(g2) = num_members_group(g2) + len_g1;

        % mark idx1 and idx2 as already being grouped
        is_grouped(idx1) = 1;
        is_grouped(idx2) = 1;
    end
end

function rel = get_reliability(img)
    rel = zeros(size(img));

    % get the shifted images (N-2, N-2)
    img_im1_jm1 = img(1:end-2, 1:end-2);
    img_i_jm1   = img(2:end-1, 1:end-2);
    img_ip1_jm1 = img(3:end  , 1:end-2);
    img_im1_j   = img(1:end-2, 2:end-1);
    img_i_j     = img(2:end-1, 2:end-1);
    img_ip1_j   = img(3:end  , 2:end-1);
    img_im1_jp1 = img(1:end-2, 3:end  );
    img_i_jp1   = img(2:end-1, 3:end  );
    img_ip1_jp1 = img(3:end  , 3:end  );

    % calculate the difference
    gamma = @(x) sign(x) .* mod(abs(x), pi);
    H  = gamma(img_im1_j   - img_i_j) - gamma(img_i_j - img_ip1_j  );
    V  = gamma(img_i_jm1   - img_i_j) - gamma(img_i_j - img_i_jp1  );
    D1 = gamma(img_im1_jm1 - img_i_j) - gamma(img_i_j - img_ip1_jp1);
    D2 = gamma(img_im1_jp1 - img_i_j) - gamma(img_i_j - img_ip1_jm1);

    % calculate the second derivative
    D = sqrt(H.*H + V.*V + D1.*D1 + D2.*D2);

    % assign the reliability as 1 / D
    rel(2:end-1, 2:end-1) = 1./D;

    % assign all nan's in rel with non-nan in img to 0
    % also assign the nan's in img to nan
    rel(isnan(rel) & ~isnan(img)) = 0;
    rel(isnan(img)) = nan;
end

function [h_edges, v_edges] = get_edges(rel)
    [Ny, Nx] = size(rel);
    h_edges = [rel(1:end, 2:end) + rel(1:end, 1:end-1), nan(Ny, 1)];
    v_edges = [rel(2:end, 1:end) + rel(1:end-1, 1:end); nan(1, Nx)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% from: https://www.mathworks.com/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
% Unwrapping phase based on Ghiglia and Romero (1994) based on weighted and unweighted least-square method
% URL: https://doi.org/10.1364/JOSAA.11.000107
% Inputs:
%   * psi: wrapped phase from -pi to pi
%   * weight: weight of the phase (optional, default: all ones)
% Output:
%   * phi: unwrapped phase from the weighted (or unweighted) least-square phase unwrapping
% Author: Muhammad F. Kasim (University of Oxford, 2016)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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



% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Azimuth
% disp('(1/2) Calculating azimuth map Fourier transform...')
% aziFT_combined_f = zeros(size(aziResp_f,1),size(aziResp_f,2),size(aziResp_f,3));
% for x = 1:size(aziResp_f,1)
%     for y =1:size(aziResp_f,2)
%         disp([num2str(y) ',' num2str(x)])
%         aziFT_combined_f(y,x,:) = single(fft(aziResp_f(y,x,:)));
%     end
% end
% %aziFT_combined_f = single(fft(aziResp_f,[],3));
% aziMag_f = double(abs(aziFT_combined_f/size(aziFT_combined_f,3))); % real component
% 
% aziFT_f.mag   = aziMag_f;
% aziFT_f.angle = angle(aziFT_combined_f); % imaginary component
% %b
% 
% aziFT_combined_b = zeros(size(aziResp_b,1),size(aziResp_b,2),size(aziResp_b,3));
% for x = 1:size(aziResp_b,1)
%     for y =1:size(aziResp_b,2)
%         disp([num2str(y) ',' num2str(x)])
%         aziFT_combined_b(y,x,:) = single(fft(aziResp_b(y,x,:)));
%     end
% end
% 
% %aziFT_combined_b = single(fft(aziResp_b,[],3));
% aziMag_b = double(abs(aziFT_combined_b/size(aziFT_combined_b,3))); % real component
% 
% aziFT_b.mag   = aziMag_b;
% aziFT_b.angle = angle(aziFT_combined_b); % imaginary component
% 
% % Altitude 
% disp('(2/2) Calculating altitude map Fourier transform...')
% 
% altFT_combined_u = zeros(size(altResp_u,1),size(altResp_u,2),size(altResp_u,3));
% for x = 1:size(altResp_u,1)
%     for y =1:size(altResp_u,2)
%         disp([num2str(y) ',' num2str(x)])
%         altFT_combined_u(y,x,:) = single(fft(altResp_u(y,x,:)));
%     end
% end
% 
% %altFT_combined_u = single(fft(altResp_u,[],3));
% altMag_u = double(abs(altFT_combined_u/size(altFT_combined_u,3)));
% 
% altFT_u.mag   = altMag_u; 
% altFT_u.angle = angle(altFT_combined_u);
% 
% 
% %d
% 
% altFT_combined_d = zeros(size(altResp_d,1),size(altResp_d,2),size(altResp_d,3));
% for x = 1:size(altResp_d,1)
%     for y =1:size(altResp_d,2)
%         disp([num2str(y) ',' num2str(x)])
%         altFT_combined_d(y,x,:) = single(fft(altResp_d(y,x,:)));
%     end
% end
% %altFT_combined_d = single(fft(altResp_d,[],3));
% altMag_d = double(abs(altFT_combined_d/size(altFT_combined_d,3)));
% 
% altFT_d.mag   = altMag_d; 
% altFT_d.angle = angle(altFT_combined_d);