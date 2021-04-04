import numpy as np
from sklearn.feature_selection import mutual_info_regression


def compute_mutual_infos(z, factors):
    n_z = z.shape[1]
    n_l = factors.shape[1]
    mutual_infos = np.zeros((n_z, n_l), dtype=np.float32)
    print("compute mutual infos:")

    for zidx in range(n_z):
        for lidx in range(n_l):
            code = z[:, zidx]
            vals = factors[:, lidx]
            mutual_infos[zidx, lidx] = calculate_mi(code, vals, bins=20)
            print("zidx:{}, lidx:{}, MI:{}".format(
                zidx, lidx, mutual_infos[zidx, lidx]))

    return mutual_infos


# here originally a is continous
# the discretization implememnted by me
# def calculate_mi(a, b, bins=20):
#     a_vals, a_bins = np.histogram(a, bins=bins)
#     b_vals, b_bins = np.histogram(b, bins=bins)
#     mi = 0.
#     n = len(a)
#     epsilon = 1e-4
#     b_match = a_match = np.zeros(len(b))
#     for b_l, b_r in zip(b_bins[:-1], b_bins[1:]):
#         if b_r != b_bins[-1]: c d
#             b_match = (b >= b_l) & (b < b_r)
#         else:
#             b_match = (b >= b_l) & (b <= b_r)

#         for a_l, a_r in zip(a_bins[:-1], a_bins[1:]):
#             if a_r != a_bins[-1]:
#                 a_match = (a >= a_l) & (a < a_r)
#             else:
#                 a_match = (a >= a_l) & (a <= a_r)
#             p_ab = float((a_match & b_match).sum()) / n
#             p_a = float(a_match.sum()) / n
#             p_b = float(b_match.sum()) / n
#             if p_a != 0. and p_b != 0. and p_ab > epsilon:
#                 # print a_l, a_r, b_val, p_ab, p_a, p_b, p_ab/(p_a*p_b)
#                 mi += p_ab * np.log2((p_ab) / (p_a * p_b))
#     return mi

# using the sklearn method
def calculate_mi(a, b, bins=20):
    print("shape: a{}, b{}".format(a.shape, b.shape))
    a = a.reshape(-1, 1)
    #b = b.reshape(-1, 1)
    mi = mutual_info_regression(a, b)
    return mi


def compute_deviations(mutual_infos, result_str=None):
    n_z = mutual_infos.shape[0]
    n_l = mutual_infos.shape[1]
    deviations = np.zeros(n_z)
    thetas = np.zeros(n_z)
    list_l_id = []
    vec_l_id = []
    for zidx in range(n_z):
        print("zidx:{}".format(zidx))
        row = mutual_infos[zidx, :]
        max_mi_idx = np.argmax(row)
        max_mi_list = np.argsort(-row)
        print('max_mi_idx:', max_mi_list)
        list_l_id.append(max_mi_idx)
        vec_l_id.append(max_mi_list)
        theta = mutual_infos[zidx, max_mi_idx]
        template = np.zeros(n_l)
        template[max_mi_idx] = theta
        dist = np.sum((row - template)**2.) / np.sum((theta**2.) * (n_l - 1))
        # if not (factor_names is None):
        #     print zidx, max_mi_idx, factor_names[max_mi_idx], row[max_mi_idx], dist, row
        # else:
        #     print zidx, max_mi_idx, row[max_mi_idx], dist, row
        deviations[zidx] = dist
        thetas[zidx] = theta
    max_lid_vec = np.array(list_l_id)
    max_lid_vec_list = np.stack(vec_l_id, axis=0)
    np.save("modularity/{}_lid.npy".format(result_str), max_lid_vec)
    np.save("modularity/{}_mis.npy".format(result_str), mutual_infos)
    np.save("modularity/{}_lid_list.npy".format(result_str), max_lid_vec_list)
    return deviations, thetas


def get_modularity(z, factors, result_str):
    mis = compute_mutual_infos(z, factors)
    deviations, thetas = compute_deviations(mis, result_str)
    modularity = 1 - deviations
    mean_modularity = modularity.mean()
    return modularity, mean_modularity
