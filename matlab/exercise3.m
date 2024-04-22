clear
% close

proxOps = ProximalOperators();
n = 100;          % スパースベクトルの次元 (dimension of a sparse vector)
k = n / 4;        % 観測ベクトルの次元 (dimension of an observed vector)
errorStd = 0.01;  % 白色ガウス雑音の標準偏差 (standard deviation of Gaussian noise)
spaseRate = 0.05; % 非ゼロ要素の割合 (rate of nonzero entries)
l1Weight = 0.3;   % L1ノルムの重要度 (weight of L1 norm)
nIter = 5000;     % 反復数 (number of iterations)

% スパースベクトル作成 (generate a sparse vector to be estimated)
nNonzero = round(spaseRate * n);      % 非ゼロ要素数 (number of nonzero entries)
nonzeroIndexes = randperm(n,nNonzero) % 非ゼロ要素のサポート (support of nonzer entries)
originalSignal = zeros(n,1);
originalSignal(nonzeroIndexes) = [1 0 -1 1 0] % 2 * (round(rand(nNonzero, 1)) - 0.5); % 所望のスパース信号 (sparse vector to be estimated)

% 観測データの作成 (generate an observed vector)
observationMatrix = randn(k,n); % 観測行列 (observation matrix)
observedSignal = observationMatrix*originalSignal + errorStd*randn(k,1); % 観測ベクトル (observed vector)

% アルゴリズム (algorithm)
stepSize = 2 / (svds(observationMatrix,1)^2 + 10); % ステップサイズ (stepsize)
initialGuess = observationMatrix\observedSignal; % 初期解 = 最小二乗解 (initial solution)
currentGuess = initialGuess;

disp(norm(initialGuess, 1))
disp(norm(initialGuess, 2))

%%%%%%%%%%%%%!!! Excercise !!!%%%%%%%%%%%%%%%%%%%%
for i = 1:nIter
    optimizedGuessWithGrad = currentGuess - stepSize *  observationMatrix' * (observationMatrix * currentGuess - observedSignal);
    currentGuess = proxOps.proxL1Norm(optimizedGuessWithGrad, l1Weight * stepSize);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 結果をプロット (plot results)
figure(1);
subplot(1,3,1), plot(originalSignal),ylim([-1,1]), title('original');
subplot(1,3,2), plot(initialGuess),ylim([-1,1]), title('initial');
subplot(1,3,3), plot(currentGuess),ylim([-1,1]), title('optimized');
axis([1,100,-1,1]);
saveas(gcf, '../output/exercise3-result.jpg');