import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy.sparse.linalg import svds
from pathlib import Path
from matplotlib.image import imread
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr


def prox_L1_norm(signal: Union[NDArray[float], NDArray[NDArray[float]]], gamma: float) -> Union[NDArray[NDArray], float]:
    return np.maximum(signal - gamma, 0) + np.minimum(signal + gamma, 0)


def prox_nuclear_norm(signal: NDArray[NDArray[float]], gamma: float) -> NDArray[NDArray]:
    u, s, vt = np.linalg.svd(signal)
    soft_s = np.zeros((u.shape[1], vt.shape[0]))
    np.fill_diagonal(soft_s, prox_L1_norm(s, gamma))
    return u @ soft_s @ vt


def prox_box_constraint(signal: Union[NDArray[float], NDArray[NDArray[float]]], l: float, r: float) -> Union[NDArray[NDArray], float]:
    return np.clip(signal, l, r)


def exercise2():
    arr0 = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    arr1 = np.array([[-3, -2, -1], [-1, 0, 1]], dtype=float)
    gamma = 1
    l = -1
    r = 1
    ret0 = prox_L1_norm(arr0, gamma)
    ret1 = prox_L1_norm(arr1, gamma)
    ret2 = prox_nuclear_norm(arr1, gamma)
    ret3 = prox_box_constraint(arr0, l, r)
    ret4 = prox_box_constraint(arr1, l, r)
    print(ret0)
    print(ret1)
    print(ret2)
    print(ret3)
    print(ret4)


def exercise3():
    n = 100             # スパースベクトルの次元 (dimension of a sparse vector)
    k = n // 4          # 観測ベクトルの次元 (dimension of an observed vector)
    error_std = 0.01    # 白色ガウス雑音の標準偏差 (standard deviation of Gaussian noise)
    sparse_rate = 0.05  # 非ゼロ要素の割合 (rate of nonzero entries)
    l1_weight = 0.3     # L1ノルムの重要度 (weight of L1 norm),  = lambda
    n_iter = 5000       # 反復数 (number of iterations)

    # スパースベクトル作成 (generate a sparse vector to be estimated)
    n_nonzero = round(sparse_rate * n)                      # 非ゼロ要素数 (number of nonzero entries)
    nonzero_indexes = np.random.permutation(n)[:n_nonzero]  # 非ゼロ要素のサポート (support of nonzero entries)
    original_signal = np.zeros(n)
    original_signal[nonzero_indexes] = 2 * (np.round(np.random.rand(n_nonzero)) - 0.5)  # 所望のスパース信号 (sparse vector to be estimated)

    # 観測データの作成 (generate an observed vector)
    observation_matrix = np.random.randn(k, n)  # 観測行列 (observation matrix), = Phi
    observed_signal = observation_matrix @ original_signal + error_std * np.random.randn(k)  # 観測ベクトル (observed vector)

    # アルゴリズム (algorithm)
    step_size = 2 / (svds(observation_matrix, 1, return_singular_vectors=False)[0] ** 2 + 10)  # = gamma

    # 初期推定値の計算
    initial_guess, _, _, _ = np.linalg.lstsq(observation_matrix, observed_signal)  # 初期解 = 最小二乗解 (initial solution)
    current_guess = initial_guess

    print("initial: ", current_guess)
    print("loss: ", np.linalg.norm(observation_matrix.dot(current_guess) - observed_signal))
    print("true loss", np.linalg.norm(original_signal - current_guess))
    print("step_size: ", step_size)

    for _ in range(n_iter):
        optimized_guess_with_grad = current_guess - step_size * observation_matrix.T @ (observation_matrix.dot(current_guess) - observed_signal)
        current_guess = prox_L1_norm(optimized_guess_with_grad, l1_weight * step_size)

    print(original_signal)
    print(current_guess)
    print("loss: ", np.linalg.norm(observation_matrix.dot(current_guess) - observed_signal))
    print("true loss", np.linalg.norm(original_signal - current_guess))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(original_signal)
    axes[0].set_ylim([-1, 1])
    axes[0].set_xlim([1, 100])
    axes[0].set_title("original")

    axes[1].plot(initial_guess)
    axes[1].set_ylim([-1, 1])
    axes[1].set_xlim([1, 100])
    axes[1].set_title("initial")

    axes[2].plot(current_guess)
    axes[2].set_ylim([-1, 1])
    axes[2].set_xlim([1, 100])
    axes[2].set_title("optimized")

    plt.tight_layout()
    plt.show()


def exercise4():
    img_path = Path(__file__).parent.joinpath('../image/brick.png')
    img_matrix = imread(img_path).astype(float)

    img_size = img_matrix.shape

    # set of parameters
    l1_norm_weight = 0.12  # weight for the l1 norm
    step_size = 1          # step size of ADMM
    n_iter = 2000           # number of iterations

    # 初期化
    L = img_matrix          # low rank matrix
    S = np.zeros(img_size)  # sparse matrix

    Z1 = np.zeros(img_size)
    Z2 = np.zeros(img_size)
    Z3 = img_matrix

    Y1 = np.zeros(img_size)
    Y2 = np.zeros(img_size)
    Y3 = np.zeros(img_size)

    def objective_function():
        return np.sum(np.linalg.svd(L, compute_uv=False)) + l1_norm_weight * np.sum(np.abs(S)) + (0 if np.sum(np.abs(L+S - img_matrix)) / (img_size[0] * img_size[1]) < 1e-4 else 1e9)

    print("nuclear norm: ", np.sum(np.linalg.svd(img_matrix, compute_uv=False)))
    print("objective: ", objective_function())

    prev_objective = objective_function()
    for _ in range(n_iter):
        L = 1/3 * (2 * (Z1 - Y1) - (Z2 - Y2) + Z3 + Y3)
        S = (Z1 - Y1) + Z3 - Y3 - 2*L
        Z1 = prox_nuclear_norm(L+Y1, step_size)
        Z2 = prox_L1_norm(S+Y2, step_size * l1_norm_weight)
        Y1 = Y1 + L - Z1
        Y2 = Y2 + S - Z2
        Y3 = Y3 + L + S - Z3
        next_objective = objective_function()
        if next_objective >= prev_objective:
            print("objective: ", objective_function(), np.sum(np.linalg.svd(L)[1]))

        prev_objective = next_objective

    print("nuclear norm: ", np.sum(np.linalg.svd(L)[1]))
    print("objective: ", objective_function())

    # show result
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_matrix, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(L, cmap='gray')
    plt.title('Low-rank Matrix')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(S, cmap='gray')
    plt.title('Sparse Matrix')
    plt.axis('off')

    plt.text(0, 250, f"lambda: {l1_norm_weight}, objective: {objective_function(): .2f}", fontsize=13)

    plt.tight_layout()
    plt.show()


def exercise5():
    img_path = Path(__file__).parent.joinpath('../image/culicoidae.png')
    image_matrix = imread(img_path).astype(float)
    n_row, n_col = image_matrix.shape
    n_pixel = n_row * n_col

    # set of parameters
    l1_norm_weight = 1e-5  # regularization parameter
    lipschitz_constant = 1  # Lipschitz constant
    opDtD = 8  # operator norm of DtD
    step_size_1 = 0.8  # step size of PDS
    step_size_2 = 0.99 / (step_size_1 * opDtD) - lipschitz_constant / (2 * opDtD)
    n_iter = 5000  # max number of iterations
    eps = 1e-5  # stopping criterion
    error_std = 10 / 255  # error std

    # random decimation operator
    n_observed_pixel = round(n_pixel / 2)  # decimation rate
    indexes_set = np.random.permutation(n_pixel)
    Phi = lambda z: z.flatten()[indexes_set[:n_observed_pixel]]  # decimation operator

    def Phit(x):  # transpose of Phi
        tmp = np.zeros(n_row * n_col)
        tmp[indexes_set[:n_observed_pixel]] = x
        return tmp.reshape(n_row, n_col)

    observed_matrix = Phi(image_matrix) + error_std * np.random.randn(n_observed_pixel)

    # init
    D = lambda z: np.stack([np.roll(z, -1, axis=0) - z, np.roll(z, -1, axis=1) - z], axis=2)
    Dt = lambda z: np.roll(z[:, :, 0], 1, axis=0) - z[:, :, 0] + np.roll(z[:, :, 1], 1, axis=1) - z[:, :, 1]

    u = Phit(observed_matrix)
    y = D(u)

    for i in range(n_iter):
        prev_u = u.copy()
        u = u - step_size_1 * (Phit(Phi(u) - observed_matrix) + Dt(y))
        u = prox_box_constraint(u, 0, 1)

        y = y + step_size_2 * D(2 * u - prev_u)
        y = y - step_size_2 * prox_L1_norm(y / step_size_2, l1_norm_weight / step_size_2)

    # show result
    psnr_val = psnr(u, image_matrix, data_range=1)
    print(f"PSNR = {psnr_val:.3f}")

    plt.figure(figsize=(17, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(image_matrix, cmap='gray')
    plt.title('original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(Phit(observed_matrix).reshape(n_row, n_col), cmap='gray')
    plt.title('observation')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(u.reshape(n_row, n_col), cmap='gray')
    plt.title('restored')
    plt.axis('off')

    plt.text(1, 270, f"lambda: {l1_norm_weight}, psnr: {psnr_val: .3f}", fontsize=13)

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)
    # exercise2()
    exercise3()
    # exercise4()
    # exercise5()


if __name__ == '__main__':
    main()

