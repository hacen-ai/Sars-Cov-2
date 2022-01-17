#%%
import numpy as np
import pandas as pd
from numba import jit, njit, prange, objmode
import threading
import concurrent.futures
import warnings
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")


@njit(fastmath=True, parallel=True, cache=True)
def P_x_ik(BAM_k_i, variants_matrix_i_j):
    if BAM_k_i == -1:  # case: X{i,k} was not read
        return 1
    elif BAM_k_i == 1:
        return variants_matrix_i_j
    else:
        return 1 - variants_matrix_i_j


@njit(fastmath=True, parallel=True, cache=True)
def P_x_k(BAM, variants_matrix, nb_mutation, k, j):
    res = 1
    for i in prange(nb_mutation):
        res *= P_x_ik(BAM[k, i], variants_matrix[i, j])
    return res


@njit(fastmath=True, parallel=True, cache=True)
def dist(v1, v2):
    return np.sum(np.abs(v1 - v2))


@njit(fastmath=True, parallel=True, cache=True)
def Expectation_v(alpha, size, nb_variants, nb_mutation, BAM, variants_matrix, T, theta_new):
    variants_matrix_prim = variants_matrix - 2 * alpha * variants_matrix + alpha
    theta_new_log = [np.log(theta_new_j) for theta_new_j in theta_new]
    result = 0.0
    for k in prange(size):
        for j in prange(nb_variants):
            result += T[k, j] * theta_new_log[j]
            for i in prange(nb_mutation):
                result += T[k, j] * np.log(P_x_ik(BAM[k, i], variants_matrix_prim[i, j]))
    return -1 * result


def call_minimize_scalar(size, nb_variants, nb_mutation, BAM, variants_matrix, T, theta_new):
    result = minimize_scalar(Expectation_v, bounds=(0, 0.5), method="bounded", args=(size, nb_variants, nb_mutation, BAM, variants_matrix, T, theta_new))
    return result.x


# %%
@njit(fastmath=True, parallel=True, cache=True)
def algo_EM(size, nb_variants, nb_mutation, BAM, variants_matrix, alpha=-1, eps=0.0001, max_iter=100):
    # print("EM_start")
    # initialisation
    if alpha == -1 or alpha == 0.0:  # alpha = -1 means, 2) alpha = 0. cause probleme so change it to a verry small value
        alpha = 0.00001
        alpha_provided = False
    else:
        alpha_provided = True
    T = np.zeros((size, nb_variants), np.float64)
    theta = np.array([1.0 / nb_variants for j in range(nb_variants)])
    theta_new = np.array([1.0 / nb_variants for j in range(nb_variants)])  # just to enter in the while loop
    theta_new[0] = theta[0] + 0.02
    M_prim = variants_matrix - 2 * alpha * variants_matrix + alpha

    # Start Iteration
    idx_iter = 0
    while (dist(theta, theta_new) > eps) and (idx_iter < max_iter):
        ## E step:
        theta = theta_new
        for k in prange(size):
            denominator = 0
            for jj in prange(nb_variants):
                denominator += P_x_k(BAM, M_prim, nb_mutation, k, jj) * theta[jj]

            for j in prange(nb_variants):
                T[k, j] = P_x_k(BAM, M_prim, nb_mutation, k, j) * theta[j] / denominator

        ## M step:
        theta_new = T.sum(axis=0) / size
        # print('new : ',theta_new)
        idx_iter += 1

        ## Optimise for alpha:
        if not alpha_provided:
            with objmode(alpha_optimal="float64"):
                alpha_optimal = call_minimize_scalar(size, nb_variants, nb_mutation, BAM, variants_matrix, T, theta_new)
            # print(alpha_optimal)
            M_prim = variants_matrix - 2 * alpha_optimal * variants_matrix + alpha_optimal
        print(theta_new, alpha_optimal)

    return (theta_new, alpha_optimal)
