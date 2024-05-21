clear
% close

addpath('./lib/prox');
addpath('./lib/difference_operator');

function result = calcObjective(u, L)
    result = u.' * L * u;
end

given_variables = load('lib/problem2_given_variables.mat');
Graph = given_variables.Graph;
L = given_variables.L;
M = given_variables.M;
N = given_variables.N;
Phi = given_variables.Phi;
alpha = given_variables.alpha;
epsilon = given_variables.epsilon;
sigma = given_variables.sigma;
u_true = given_variables.u_true;
v = given_variables.v;
gammaU = 0.1;
gammaS = 0.1;
gammaY = 0.1;

%gsp_plot_signal(graph, u_true);

nIter = 20000;
u = zeros(N, 1);
s = zeros(M, 1);

y = Phi * u + s;
disp(['MSE = ', num2str(mean((u - u_true).^2),4)]);
disp(['objective = ', num2str(calcObjective(u, L),4)]);

diffs = zeros(nIter);
objectives = zeros(nIter);

loopCount = 0;

for i = 1:nIter
    loopCount = loopCount + 1;

    prevU = u;
    prevS = s;
    u = max(0, u - gammaU*(2*L*prevU + Phi.' * y));
    s = ProjFastL1Ball(prevS - gammaS*y, alpha);

    y = y + gammaY * (Phi * (2*u - prevU) + (2*s -prevS) );
    y = y - gammaY * ProjL2ball(y / gammaY, v, epsilon);

%    収束判定 & early break
    diffU = u-prevU;
    diff = sqrt(sum(sum(sum(diffU.^2))) / sum(sum(sum(prevU.^2))));
    diffs(i) = diff;
    objectives(i) = calcObjective(u, L);

    if (diff < 1e-5)
        break
    end
end

%gsp_plot_signal(graph, u);

disp(['loopCount = ', num2str(loopCount)]);
disp(['MSE = ', num2str(mean((u - u_true).^2),4)]);
disp(['objective = ', num2str(calcObjective(u, L),4)]);

figure(1);
subplot(1,2,1), plot(diffs(1:loopCount)),      xlim([1, loopCount]), title('diff'), set(gca, 'YScale', 'log');;
subplot(1,2,2), plot(objectives(1:loopCount)), xlim([1, loopCount]), title('objective');

saveas(gcf, '../../output/graphSignalProcessing.jpg');
