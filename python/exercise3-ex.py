import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from typing import NamedTuple, Union, List
from numpy.typing import NDArray



class Result(NamedTuple):
    initial_loss: float
    initial_true_loss: float
    loss: float
    true_loss: float


class Params(NamedTuple):
    n: int = 100
    k: int = n // 4
    error_std: float = 0.01
    sparse_rate: float = 0.05
    l1_weight: float = 0.3
    seed: int = 42


def prox_L1_norm(signal: Union[NDArray[float], NDArray[NDArray[float]]], gamma: float) -> Union[NDArray[NDArray], float]:
    return np.maximum(signal - gamma, 0) + np.minimum(signal + gamma, 0)


def exercise3(params: Params) -> Result:
    n, k, error_std, sparse_rate, l1_weight, _ = params
    np.random.seed(params.seed)
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

    initial_loss = np.linalg.norm(observation_matrix.dot(current_guess) - observed_signal)
    initial_true_loss = np.linalg.norm(original_signal - current_guess)

    for _ in range(n_iter):
        optimized_guess_with_grad = current_guess - step_size * observation_matrix.T @ (observation_matrix.dot(current_guess) - observed_signal)
        current_guess = prox_L1_norm(optimized_guess_with_grad, l1_weight * step_size)

    loss = np.linalg.norm(observation_matrix.dot(current_guess) - observed_signal)
    true_loss = np.linalg.norm(original_signal - current_guess)
    return Result(initial_loss, initial_true_loss, loss, true_loss)


def grid_search(params_arr: List[Params], xs: NDArray[float], param_name: str):
    initial_losses = []
    initial_true_losses = []
    losses = []
    true_losses = []
    for params in params_arr:
        result = exercise3(params)
        initial_losses.append(result.initial_loss)
        initial_true_losses.append(result.initial_true_loss)
        losses.append(result.loss)
        true_losses.append(result.true_loss)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(xs, initial_losses, label="initial loss")
    axes[0].plot(xs, losses, label="loss")
    axes[0].set_title("loss")
    axes[0].legend()


    axes[1].plot(xs, initial_true_losses, label="initial true loss")
    axes[1].plot(xs, true_losses, label="true loss")
    axes[1].set_title("true loss")
    axes[1].legend()

    fig.suptitle(f"grid search: {param_name}")
    plt.tight_layout()
    plt.show()


def grid_search_sparse_rate():
    sparse_rates = np.linspace(0.02, 0.3, 57)
    params_arr = [Params(sparse_rate=sparse_rate) for sparse_rate in sparse_rates]
    grid_search(params_arr, sparse_rates, "nonzero entries")


def grid_search_k():
    ks = np.linspace(5, 80, 61, dtype=int)
    params_arr = [Params(k=k) for k in ks]
    grid_search(params_arr, ks, "dimension of an observed vector")


def grid_search_error_std():
    error_stds = np.linspace(0, 0.3, 50)
    params_arr = [Params(error_std=error_std) for error_std in error_stds]
    grid_search(params_arr, error_stds, "standard deviation of Gaussian noise")


def grid_search_l1_weight():
    l1_weights = np.linspace(0, 1, 100)
    params_arr = [Params(l1_weight=l1_weight) for l1_weight in l1_weights]
    grid_search(params_arr, l1_weights, "weight of L1 norm")


def grid_search_random_seed():
    randoms = np.linspace(0, 50, 50, dtype=int)
    params_arr = [Params(seed=seed) for seed in randoms]
    grid_search(params_arr, randoms, "random seed")


if __name__ == '__main__':
    np.random.seed(42)
    grid_search_sparse_rate()
    grid_search_k()
    grid_search_error_std()
    grid_search_l1_weight()
    grid_search_random_seed()
