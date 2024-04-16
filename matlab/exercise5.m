clear
% close

proxOps = ProximalOperators();

%% observation generation
imageMatrix = double(imread('../image/culicoidae.png'))/255;
[nRow, nCol] = size(imageMatrix);
nPixel = nRow*nCol;

%% set of parameters
l1NormWeight = 0.01; % regularization parameter
lipschitzConstant = 1; % Lipschitz constant
opDtD = 8; % operator norm of DtD
stepSize1 = 0.8; % stepsize of PDS
stepSize2 = 0.99/(stepSize1*opDtD) - lipschitzConstant/(2*opDtD);
nIter = 5000; % max number of iterations
eps = 1e-5; % stopping criterion
errorStd = 10/255; % noise standard deviation

% randam decimation operator
nObservedPixel = round(nPixel / 2); % decimation rate
indexesSet = randperm(nPixel);
Phi = @(z) z(indexesSet(1:nObservedPixel))';                   % decimation operator
Phit = @(z) deci_trans(z,indexesSet,nObservedPixel,nRow,nCol); % transpose of Phi

observedMatrix = Phi(imageMatrix) + errorStd*randn(nObservedPixel,1); % observation (decimation+noise)

%% initialization

% difference operator
D = @(z) cat(3, z([2:nRow, 1],:) - z, z(:,[2:nCol, 1])-z);
Dt = @(z) [-z(1,:,1)+z(nRow,:,1); - z(2:nRow,:,1) + z(1:nRow-1,:,1)] ...
    +[-z(:,1,2)+z(:,nCol,2), - z(:,2:nCol,2) + z(:,1:nCol-1,2)];

% variables
u = Phit(observedMatrix);
y = D(u);

%% main loop%%
%%%%%%%%%%%%%!!! Excercise !!!%%%%%%%%%%%%%%%%%%%%
for i = 1:nIter
    prevU = u;
    u = u - stepSize1 * (Phit(Phi(u) - observedMatrix) + Dt(y));
    u = proxOps.proxBoxConstraint(u, 0, 1);

    y = y + stepSize2 * D(2 * u - prevU);
    y = y - stepSize2 * proxOps.proxL1Norm(y / stepSize2, l1NormWeight / stepSize2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% result plot

disp(['PSNR = ', num2str(psnr(u,imageMatrix,1),4)]);

figure(1);
subplot(1,3,1), imshow(imageMatrix), title('original');
subplot(1,3,2), imshow(Phit(observedMatrix)), title('observation');
subplot(1,3,3), imshow(u), title('restored');
saveas(gcf, '../output/exercise5-result.jpg');

function[y] = deci_trans(x,I,K,rows,cols)
 y = zeros(rows,cols);
 y(I(1:K)) = x;
end
