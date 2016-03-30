function [u, v] = HS(im1, im2, alpha, ite, uInitial, vInitial, displayFlow, displayImg, pyramid_depth)
% Horn-Schunck optical flow method 
% Horn, B.K.P., and Schunck, B.G., Determining Optical Flow, AI(17), No.
% 1-3, August 1981, pp. 185-203 http://dspace.mit.edu/handle/1721.1/6337
%
% Usage: 
% [u, v] = HS(im1, im2, alpha, ite, uInitial, vInitial, displayFlow)
% For an example, run this file from the menu Debug->Run or press (F5)
%
% -im1,im2 : two subsequent frames or images.
% -alpha : a parameter that reflects the influence of the smoothness term.
% -ite : number of iterations.
% -uInitial, vInitial : initial values for the flow. If available, the
% flow would converge faster and hence would need less iterations ; default is zero. 
% -displayFlow : 1 for display, 0 for no display ; default is 1.
% -displayImg : specify the image on which the flow would appear ( use an
% empty matrix "[]" for no image. )
% -pyramid_depth : depth of the pyramid (ratio = 0.5)
%
% Author: Mohd Kharbat at Cranfield Defence and Security
% mkharbat(at)ieee(dot)org , http://mohd.kharbat.com
% Published under a Creative Commons Attribution-Non-Commercial-Share Alike
% 3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
%
% October 2008
% Rev: Jan 2009

% Some addendums from http://www.ipol.im/pub/art/2013/20/article.pdf

%% Default parameters

if nargin < 1 || nargin < 2
    im1 = imread('eval-data-gray/Yosemite/frame07.png');
    im2 = imread('eval-data-gray/Yosemite/frame14.png');
end
if nargin < 3, alpha = 40; end
if nargin < 4, ite = 1000; end
if (nargin < 5 || nargin < 6) || (size(uInitial, 1) == 0 || size(vInitial,1 ) == 0)
    uInitial = zeros(size(im1(:,:,1)));
    vInitial = zeros(size(im2(:,:,1)));
end
if nargin < 7, displayFlow = 1; end
if nargin < 8, displayImg = im1; end
if nargin < 9, pyramid_depth = 8; end

eta = 0.65;


%% Convert images to grayscale
if size(size(im1), 2) == 3, im1 = rgb2gray(im1); end
if size(size(im2),2) == 3, im2 = rgb2gray(im2); end

im1 = double(im1); im2 = double(im2);

im1 = smoothImg(im1,1); im2 = smoothImg(im2,1);

%% Build pyramid
im1_pyr{1} = im1;
im2_pyr{1} = im2;
for i = 2:pyramid_depth
    im1_pyr{i} = imresize(im1_pyr{i-1}, eta);
    im2_pyr{i} = imresize(im2_pyr{i-1}, eta);
end

%%
% Set initial value for the flow vectors

tic;

%TODO: not usable anymore?
u = uInitial;
v = vInitial;

u = zeros(size(im1_pyr{pyramid_depth}));
v = zeros(size(im1_pyr{pyramid_depth}));



% Averaging kernel
kernel_1 = [1/12 1/6 1/12;
            1/6   0  1/6;
            1/12 1/6 1/12];

hist_u = {};
hist_v = {};
hist_warp = {};

w = 1e-3; %YAHOG: WHY?
epsilon = 1e-4;

% Pyramid
for j = pyramid_depth:-1:1
   
    % warps (SOR?)
    for k = 1:5 
        
        % Estimate spatiotemporal derivatives
        [x, y] = meshgrid(1:size(im1_pyr{j}, 2), 1:size(im1_pyr{j}, 1));
        %imout = interp2(im1_pyr{j}, x-u, y-v);
        imout = interp2(im1_pyr{j}, x+u, y+v);
        imout(isnan(imout)) = 0;
        [fx, fy, ft] = computeDerivatives(imout, im2_pyr{j});
        fxp{j} = fx; fyp{j} = fy; ftp{j} = ft;
        %hist_warp{j} = imout;
        %[fx, fy, ft] = computeDerivatives(im1_pyr{pyramid_depth}, im2_pyr{pyramid_depth});
        %fxp{j} = fx; fyp{j} = fy; ftp{j} = ft;
        
        un = u;
        vn = v;
        
        % Iterations
        for i = 1:ite
            % Compute local averages of the flow vectors
            uAvg = conv2(u, kernel_1, 'same');
            vAvg = conv2(v, kernel_1, 'same');

    %         % Compute flow vectors constrained by its local average and the optical flow constraints
    %         u = uAvg - ( fxp{j} .* ( ( fxp{j} .* uAvg ) + ( fyp{j} .* vAvg ) + ftp{j} ) ) ./ ( alpha^2 + fxp{j}.^2 + fyp{j}.^2); 
    %         v = vAvg - ( fyp{j} .* ( ( fxp{j} .* uAvg ) + ( fyp{j} .* vAvg ) + ftp{j} ) ) ./ ( alpha^2 + fxp{j}.^2 + fyp{j}.^2);

            ou = u; ov = v;
            u = (1 - w)*u + w*(((imout-im2_pyr{j} + fxp{j}.*un - fyp{j}.*(v-vn) ).*fxp{j} + uAvg.*alpha.^2) ./ (fxp{j}.^2 + alpha.^2));
            v = (1 - w)*v + w*(((imout-im2_pyr{j} - fxp{j}.*(u-un) + fyp{j}.*vn ).*fyp{j} + vAvg.*alpha.^2) ./ (fyp{j}.^2 + alpha.^2));
            %u = (1 - w)*u + w*(((im1_pyr{j}-im2_pyr{j} + fxp{j}.*un - fyp{j}.*(v-vn) ).*fxp{j} + uAvg.*alpha.^2) ./ (fxp{j}.^2 + alpha.^2));
            %v = (1 - w)*v + w*(((im1_pyr{j}-im2_pyr{j} - fxp{j}.*(u-un) + fyp{j}.*vn ).*fyp{j} + vAvg.*alpha.^2) ./ (fyp{j}.^2 + alpha.^2));

            % Stopping criterion
            if sum((ou(:) - u(:)).^2 + (ov(:) - v(:)).^2) < epsilon.^2, break; end
        end
    
    end

    u(isnan(u)) = 0;
    v(isnan(v)) = 0;
    
    hist_u{j} = u;
    hist_v{j} = v;
    
%     if j < pyramid_depth
%         im1 = im1_pyr{j};
%         im2 = im2_pyr{j};
%         break
%     end
    
    if j > 1
        u = imresize(u, size(im1_pyr{j-1}));
        v = imresize(v, size(im1_pyr{j-1}));
        u = 1./eta*u; v = 1./eta*v;
        
        % Re-comute derivatives
        [x, y] = meshgrid(1:size(im1_pyr{j-1}, 2), 1:size(im1_pyr{j-1}, 1));
        %imout = interp2(im1_pyr{j-1}, x-u, y-v);
        imout = interp2(im1_pyr{j-1}, x+u, y+v);
        imout(isnan(imout)) = 0;
        [fx, fy, ft] = computeDerivatives(imout, im2_pyr{j-1});
        fxp{j-1} = fx; fyp{j-1} = fy; ftp{j-1} = ft;
        hist_warp{j} = imout;
    end
    
end

%% Plotting
if displayFlow == 1
    figure(161803);
    plotFlow(u, v, displayImg, 5, 5); 
    
    figure(271828);
    subplot(211), imagesc(u), colorbar(); axis equal;
    subplot(212), imagesc(v), colorbar(); axis equal;

    for i = 1:pyramid_depth
        figure(42+i);
        subplot(221), imagesc(hist_u{i}), colorbar(); axis equal;
        subplot(222), imagesc(hist_v{i}), colorbar(); axis equal;
        linkaxes;

        subplot(223), imagesc(fxp{i}), colorbar(); axis equal;
        subplot(224), imagesc(fyp{i}), colorbar(); axis equal;
        
        figure(142+i);
        imshow(uint8(hist_warp{i}));
    end

    %% Display output
    [x, y] = meshgrid(1:size(im2, 2), 1:size(im2, 1));
    %imout = interp2(im1, x-u, y-v);
    imout = interp2(im1, x+u, y+v);
    imout(isnan(imout)) = 0;
    figure(31415);
    subplot(221), imshow(uint8(imout));
    subplot(222), imshow(uint8(im1));
    subplot(223), imshow(uint8(im2));
    subplot(224), imshow(uint8(abs(imout - im2)));
    sum(sum((imout - im2).^2))

    %keyboard;
end