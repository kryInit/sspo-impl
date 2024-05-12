import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio


def calc_mean_psnr(original, signal) -> float:
    return np.mean([psnr(original[i, :, :], signal[i, :, :], data_range=1) for i in range(original.shape[0])])


def calc_mean_ssim(original, signal) -> float:
    return np.mean([ssim(original[i, :, :], signal[i, :, :], data_range=1) for i in range(original.shape[0])])


def Dvh(z):
    diff1 = np.concatenate((z[1:], z[-1:]), axis=0) - z
    diff2 = np.concatenate((z[:, 1:], z[:, -1:]), axis=1) - z
    result = np.stack((diff1, diff2), axis=3)
    return result


def Dvht(z):
    ret0 = np.concatenate([-z[0:1, :, :, 0], -z[1:-1, :, :, 0] + z[:-2, :, :, 0], z[-2:-1, :, :, 0]], axis=0)
    ret1 = np.concatenate([-z[:, 0:1, :, 1], -z[:, 1:-1, :, 1] + z[:, :-2, :, 1], z[:, -2:-1, :, 1]], axis=1)
    return ret0 + ret1


def Dv(z):
    n = z.shape[0]
    return np.concatenate((z[1:n], z[n-1:]), axis=0) - z


def Dvt(z):
    n = z.shape[0]
    return np.concatenate((-z[0:1], -z[1:n-1] + z[0:n-2], z[n-2:n-1]), axis=0)


def proj_box(signal, left, right):
    return np.clip(signal, left, right)


def prox_l1(signal, gamma):
    return np.sign(signal) * np.maximum(np.abs(signal) - gamma, 0)


def prox_12_band(signal, gamma):
    norm = np.sqrt(np.sum(np.sum(signal ** 2, axis=3), axis=2)) + 1e-8
    tmp = np.maximum(1 - gamma / norm, 0)
    return tmp[..., np.newaxis, np.newaxis] * signal


def proj_l2_ball(signal, center, epsilon):
    diff_vec = signal - center
    radius = np.linalg.norm(diff_vec)

    return center + (epsilon / radius) * diff_vec if radius > epsilon else signal


def calc_objective(u, l, l1_norm_weight) -> float:
    return np.sum(np.linalg.norm(Dvh(u), axis=(-2, -1))) + l1_norm_weight * np.abs(np.sum(l))


def main():
    given_variables_path = Path(__file__).absolute().parent.parent.joinpath('matlab/practice/lib/problem1_given_variables.mat')
    given_variables = loadmat(str(given_variables_path))

    true_u = given_variables['U_true']
    v = given_variables['V']
    epsilon = given_variables['epsilon'][0, 0]
    gamma_l = given_variables['gamma_L'][0, 0]
    gamma_u = given_variables['gamma_U'][0, 0]
    gamma_z1 = given_variables['gamma_Y1'][0, 0]
    gamma_z2 = given_variables['gamma_Y2'][0, 0]
    gamma_z3 = given_variables['gamma_Y3'][0, 0]
    l1_norm_weight = given_variables['lambda'][0, 0]
    n_iter = 20000

    u = v.copy()
    l = np.zeros(u.shape)
    z1 = Dvh(u)
    z2 = Dv(l)
    z3 = u+l

    print(f"psnr: {calc_mean_psnr(u, true_u):.4}")
    print(f"ssim: {calc_mean_ssim(u, true_u):.4}")
    print(f"objective: {calc_objective(u, l, l1_norm_weight):.4}")

    diffs = []
    objectives = []
    mpsnrs = []
    mssims = []

    for _ in tqdm(range(n_iter)):
        prev_u = u.copy()
        prev_l = l.copy()

        u = proj_box(u - gamma_u*(Dvht(z1) + z3), 0, 1)
        l = prox_l1(l - gamma_l*(Dvt(z2) + z3), l1_norm_weight * gamma_l)

        z1 = z1 + gamma_z1 * Dvh(2*u - prev_u)
        z2 = z2 + gamma_z2 * Dv(2*l - prev_l)
        z3 = z3 + gamma_z3 * (2*(u+l) - (prev_u+prev_l))

        z1 = z1 - gamma_z1 * prox_12_band(z1 / gamma_z1, 1 / gamma_z1)
        # z2 = z2
        z3 = z3 - gamma_z3 * proj_l2_ball(z3 / gamma_z3, v, epsilon)

        # 収束判定 & early break
        diff = np.linalg.norm(u-prev_u) / np.linalg.norm(prev_u)
        objective = calc_objective(u, l, l1_norm_weight)
        # mpsnr = calc_mean_psnr(u, true_u)
        # mssim = calc_mean_ssim(u, true_u)
        diffs.append(diff)
        objectives.append(objective)
        # mpsnrs.append(mpsnr)
        # mssims.append(mssim)

        if diff < 1e-5:
            break

    print(f"psnr: {calc_mean_psnr(u, true_u):.4}")
    print(f"ssim: {calc_mean_ssim(u, true_u):.4}")
    print(f"objective: {calc_objective(u, l, l1_norm_weight):.4}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(diffs)
    axes[0].set_title("diff")
    axes[0].set_yscale('log')

    axes[1].plot(objectives)
    axes[1].set_title("objective")

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # axes[0][0].plot(diffs)
    # axes[0][0].set_title("diff")
    # axes[0][0].set_yscale('log')
    # axes[0][1].plot(objectives)
    # axes[0][1].set_title("objective")
    # axes[1][0].plot(mpsnrs)
    # axes[1][0].set_title("mpsnr")
    # axes[1][1].plot(mssims)
    # axes[1][1].set_title("mssim")

    plt.tight_layout()
    plt.show()

    print(np.min(v), np.max(v), np.min(true_u), np.max(true_u), np.min(u), np.max(u))

    scaled_v = (255 * np.transpose(v, (2, 0, 1))).astype(np.uint8)
    scaled_true_u = (255 * np.transpose(true_u, (2, 0, 1))).astype(np.uint8)
    scaled_restored_u = (255 * np.transpose(u, (2, 0, 1))).astype(np.uint8)

    output_path = Path(__file__).absolute().parent.parent.joinpath('output')
    imageio.mimsave(output_path.joinpath('v.gif'), scaled_v, duration=0.1)
    imageio.mimsave(output_path.joinpath('true_u.gif'), scaled_true_u, duration=0.1)
    imageio.mimsave(output_path.joinpath('restored_u.gif'), scaled_restored_u, duration=0.1)


if __name__ == '__main__':
    main()
