clear
% close

addpath('./lib/prox');
addpath('./lib/difference_operator');

function result = calcMeanPSNR(original, signal)
    n = size(original, 1);
    psnrSum = 0;
    for k = 1:n
        psnrSum = psnrSum + psnr(original(k, :, :), signal(k, :, :));
    end
    result = psnrSum / n;
end

function result = calcMeanSSIM(original, signal)
    n = size(original, 1);
    ssimSum = 0;
    for k = 1:n
        ssimSum = ssimSum + ssim(original(k, :, :), signal(k, :, :));
    end
    result = ssimSum / n;
end

function result = calcObjective(u, l, lambda)
    norm_Dvh = sum(sum(sqrt(sum(sum(Dvh(u).^2, 4), 3)), 2), 1);
    weighted_l1 = lambda * sum(abs(l(:)));
    result = norm_Dvh + weighted_l1;
end

given_variables = load('lib/problem1_given_variables.mat');
trueU = given_variables.U_true;
v = given_variables.V;
epsilon = given_variables.epsilon;
gammaL = given_variables.gamma_L;
gammaU = given_variables.gamma_U;
gammaZ1 = given_variables.gamma_Y1;
gammaZ2 = given_variables.gamma_Y2;
gammaZ3 = given_variables.gamma_Y3;
lambda = given_variables.lambda;
nIter = 20000;

u = v;
l = zeros(size(u));
z1 = Dvh(u);
z2 = Dv(l);
z3 = u+l;


disp(['MPSNR = ', num2str(calcMeanPSNR(u, trueU),4)]);
disp(['MSSIM = ', num2str(calcMeanSSIM(u, trueU),4)]);
disp(['objective = ', num2str(calcObjective(u, l, lambda),4)]);

diffs = zeros(nIter);
objectives = zeros(nIter);
mpsnrs = zeros(nIter);
mssims = zeros(nIter);

loopCount = 0;

for i = 1:nIter
    loopCount = loopCount + 1;

    prevU = u;
    prevL = l;
    u = ProjBox(u - gammaU*(Dvht(z1) + z3), 0, 1);
    l = Prox_l1(l - gammaL*(Dvt(z2)  + z3), lambda * gammaL);

    z1 = z1 + gammaZ1 * Dvh(2*u - prevU);
    z2 = z2 + gammaZ2 * Dv(2*l - prevL);
    z3 = z3 + gammaZ3 * (2*(u+l) - (prevU+prevL));

    z1 = z1 - gammaZ1 * Prox12band(z1 / gammaZ1, 1 / gammaZ1);
%    z2 = z2;
    z3 = z3 - gammaZ3 * ProjL2ball(z3 / gammaZ3, v, epsilon);

%    収束判定 & early break
    diffU = u-prevU;
    diff = sqrt(sum(sum(sum(diffU.^2))) / sum(sum(sum(prevU.^2))));
    diffs(i) = diff;
    objectives(i) = calcObjective(u, l, lambda);
%    mpsnrs(i) = calcMeanPSNR(u, trueU);
%    mssims(i) = calcMeanSSIM(u, trueU);

    if (diff < 1e-5)
        break
    end
end

disp(['loopCount = ', num2str(loopCount)]);
disp(['MPSNR = ', num2str(calcMeanPSNR(u, trueU),4)]);
disp(['MSSIM = ', num2str(calcMeanSSIM(u, trueU),4)]);
disp(['objective = ', num2str(calcObjective(u, l, lambda),4)]);

figure(1);
subplot(1,2,1), plot(diffs(1:loopCount)),      xlim([1, loopCount]), title('diff'), set(gca, 'YScale', 'log');;
subplot(1,2,2), plot(objectives(1:loopCount)), xlim([1, loopCount]), title('objective');

%subplot(2,2,1), plot(diffs(1:loopCount)),      xlim([1, loopCount]), title('diff'), set(gca, 'YScale', 'log');;
%subplot(2,2,2), plot(objectives(1:loopCount)), xlim([1, loopCount]), title('objective');
%subplot(2,2,3), plot(mpsnrs(1:loopCount)),     xlim([1, loopCount]), title('psnr');
%subplot(2,2,4), plot(mssims(1:loopCount)),     xlim([1, loopCount]), title('ssim');
saveas(gcf, '../../output/hyperspectralImage.jpg');
