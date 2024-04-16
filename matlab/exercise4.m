clear
% close

proxOps = ProximalOperators();

%% image loading
imgMatrix = double(imread('../image/brick.png'))/255;
imgSize = size(imgMatrix);

%% set of parameters
l1NormWeight = 0.12; % weight for the l1 norm
stepSize = 1;        % stepsize of ADMM
nIter = 200;         % number of iterations

%% initialization
L = zeros(imgSize); % low-rank matrix
S = zeros(imgSize); % sparse matrix

Z1 = zeros(imgSize);
Z2 = zeros(imgSize);
Z3 = imgMatrix;

Y1 = zeros(imgSize);
Y2 = zeros(imgSize);
Y3 = zeros(imgSize);

%% algorithm
%%%%%%%%%%%%%!!! Excercise !!!%%%%%%%%%%%%%%%%%%%%
for i = 1:nIter
    L = 1/3 * (2 * (Z1 - Y1) - (Z2 - Y2) + Z3 + Y3);
    S = (Z1 - Y1) + Z3 - Y3 - 2*L;
    Z1 = proxOps.proxNuclearNorm(L+Y1, stepSize);
    Z2 = proxOps.proxL1Norm(S+Y2, stepSize * l1NormWeight);
    Y1 = Y1 + L - Z1;
    Y2 = Y2 + S - Z2;
    Y3 = Y3 + L + S - Z3;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot results
figure(1);
subplot(1,3,1), imshow(imgMatrix);
subplot(1,3,2), imshow(L);
subplot(1,3,3), imshow(S);
saveas(gcf, '../output/exercise4-result.jpg');
