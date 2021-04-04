# Explore wavelets in python
# import pywt
import numpy as np
import torch
from torch.autograd import Variable

# print(pywt.families())

# data = range(1, 5)
# print(data)

# print(pywt.dwt(data, 'haar', 'symmetric'))  # 'dwt' does just one level of wavelet deconstruction

# coefs = pywt.wavedec(data, 'haar')  # 'wavedec' does the whole decomposition
# print(coefs)


def create_haar_dictionary(size=2, vectorType=np.ndarray):
    """ Build a Haar matrix """
    d = {}
    c = np.sqrt(2)/2
    h0 = np.array([[c, c], [c, -c]])
    for i in range(size):
        if i == 0:
            d[i] = h0
        else:
            d[i] = np.array(np.concatenate(
                (np.kron(d[i-1], [c, c]),
                 np.kron(np.identity(d[i-1].shape[0]), [c, -c]))
            ))
    if vectorType is Variable:
        for i in range(size):
            d[i] = Variable(torch.from_numpy(d[i]),requires_grad=False)
    return d


# logsize = 2
# c = np.sqrt(2)/2
# h0 = np.array([[c, c], [c, -c]])
# # print h0

# h1 = np.array(np.concatenate((np.kron(h0, [c, c]),
#                               np.kron(np.identity(h0.shape[0]), [c, -c]))))
# print(h1)

# data = range(1, 5)

# # data = [-10,5,0,-8]

# print(h1.dot(data))
# print(h1.dot(data).dot(h1))  # reconstruction using that inv(h1)=h1.T

### note that this process is considerably slower than the Mallat reconstruction algorithm

# from pprint import PrettyPrinter as pppp
# pp = pppp(indent=4)
# pp.pprint(create_haar_dictionary(3))
# print(np.count_nonzero(create_haar_dictionary(10)[9]))
# print(create_haar_dictionary(10)[9].shape)

# TODO make sparse with scipy.sparse and map to sparse matrix in pytorch



# # Do 2-d decomposition and reconstruction
# latee = Variable(torch.ones((4,8)).double(), requires_grad=False) + \
#     Variable(0.1 * torch.ones((4,8)).double().uniform_(0,0.1), requires_grad=False)
# latee[1:4,2:6] = 0
# latee[1, 4:7] = 4
# latee[3, 0:2] = 2
# latee[2,3] = -2
# print(latee)

# hd = create_haar_dictionary(10)
# for i in range(len(hd)):
#     hd[i] = Variable(torch.from_numpy(np.array(hd[i])).type(torch.DoubleTensor), requires_grad=False)
# print(latee.matmul(hd[2].t()).t().matmul(hd[1].t()).matmul(hd[1]).t().matmul(hd[2]))
