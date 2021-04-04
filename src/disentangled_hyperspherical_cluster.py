# Generative Adversarial Networks (GAN) example in PyTorch.
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import datetime as dt
import jcw_pywavelets as jpw
import pdb
import os
import tqdm
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
import pandas as pd
import datetime as dt
# from data_format import sigmoid
import sys
from importlib import reload
from F_Loss import F_Loss
# from F_Loss import F_Loss
reload(sys.modules["F_Loss"])
from F_Loss import F_Loss
from autoencoder import Generator
from autoencoder import Outputter
from autoencoder import Enforcer
from autoencoder import Clusterer
reload(sys.modules["autoencoder"])
from autoencoder import Generator
from autoencoder import Outputter
from autoencoder import Enforcer
from autoencoder import Clusterer

from sklearn.metrics import silhouette_score
# import sklearn.preprocessing.normalize as normalize
from sklearn import preprocessing
# import matplotlib as plt
import configparser
import argparse
from evaluations import get_modularity
from distutils.util import strtobool

from Soft_F_Loss import Soft_F_Loss

np.random.seed(int(1e8 + 1))
torch.manual_seed(1e8 + 1)
# gpu-mode
# gpu_mode = True
gpu_mode = torch.cuda.is_available()
device = 'cuda' if gpu_mode else 'cpu'
print("gpu mode available? ", gpu_mode)

# parse name of the config file
fileparse = argparse.ArgumentParser()
fileparse.add_argument('--filename', default='best.ini',
                       action="store")
fileparse.add_argument('--train',
                       default=False, action="store")
ftrain = "/home/cc/sepsis_data/xtrain_not_normed_notime.csv"
fileparse.add_argument('--data', default=ftrain, action="store")
fileparse.add_argument('--bootstrap',
                       default=False, action="store_true")

args = fileparse.parse_args()
config_file_name = args.filename
doTrain = bool(strtobool(args.train))
doBoot = args.bootstrap
bootdata = args.data
# doTrain = bool(doTrain)
# Parse config variables\
print("filename: ", config_file_name)
print("DO train: ", doTrain)

config = configparser.ConfigParser()
config.read(config_file_name)
proj_dir = config['Path']['proj_dir']
result_dir = config['Path']['result_dir']
parameters = config['Autoencoder']
minibatch_size = int(parameters['minibatch_size'])
subsample_stride = int(parameters['subsample_stride'])
o_learning_rate = float(parameters['o_learning_rate'])
c_learning_rate = float(parameters['c_learning_rate'])
num_epochs = int(parameters['num_epochs'])
burn_in = int(parameters['burn_in'])
print_interval = int(parameters['print_interval'])
image_interval = int(parameters['image_interval'])

desired_centroids = int(parameters['desired_centroids'])
noise_sd = float(parameters['noise_sd'])
explode_factor = float(parameters['explode_factor'])

print('do train type: ', type(doTrain))
# Synthetic data
do_validate = not doTrain  # default to true with is validating

xdata = None
print("do validate: ", do_validate)
if doBoot:
    print('do boot')
    xdata = np.load(bootdata)
else:
    # if do_validate:
    #     print("true validate")
    #     xdata = np.genfromtxt(
    #         "/home/cc/sepsis_data/xval_not_normed_notime.csv", delimiter=',')
    # else:

    #     xdata = np.genfromtxt(x
    #         "/home/cc/sepsis_data/xtrain_not_normed_notime.csv",
    #         delimiter=',')

    # the deleted is for scaled using gaussian
    # xdata = np.genfromtxt(
     #   '/home/cc/sepsis_data/xval_scaled.csv', delimiter=',')
    # xdata = xdata[range(1, xdata.shape[0]),]
    # xdata = preprocessing.normalize(xdata)

    # now scaled using minmax

    xdata = np.genfromtxt(
        '/home/cc/sepsis_data/xscaled.csv', delimiter=',')

    # xdata = np.genfromtxt(
    #     '/home/cc/sepsis_data/xscaled_minmax.csv', delimiter=',')
    xdata = xdata[range(1, xdata.shape[0]), ]
#    print('xdata shape:', xdata.shape)


mydatasize = torch.Size(xdata.shape)
mydata = torch.FloatTensor(xdata)
mydatadf = pd.DataFrame(mydata.data.numpy())
data_size = mydata.size()[0]
print("my data size: ", mydata.size())

g_input_size = int(parameters['g_input_size'])  # noise input size
hidden_size = int(parameters['hidden_size'])
g_output_size = hidden_size
side_channel_size = int(parameters['side_channel_size'])
c_input_size = hidden_size - side_channel_size
c_hidden_size = int(parameters['c_hidden_size'])
c_output_size = int(parameters['c_output_size'])  # now 2*desired_centroids
o_input_size = hidden_size
o_hidden_size = hidden_size
o_output_size = mydata.size()[1]


pr_g_update = float(parameters['pr_g_update'])
g_lambda = float(parameters['g_lambda'])  # hidden is on hypersphere
g_o_ratio = float(parameters['g_o_ratio'])
pr_c_update = float(parameters['pr_c_update'])
c_only = parameters.getboolean('c_only')
# L_{c,h}, clusters sqrt(p) match hidden angle
c_lambda = float(parameters['c_lambda'])
# cluster probabilities are l2 regularized
c_l2_lambda = float(parameters['c_l2_lambda'])
# L_{entropy}, clusters probabilities are entropic
l2_lambda = float(parameters['l2_lambda'])
cos_lambda = float(parameters['cos_lambda'])
o_lambda = float(parameters['o_lambda'])  # L_{x, x^hat}
e_lambda = float(parameters['e_lambda'])  # L_{}, H and M similarity
s_lambda = float(parameters['s_lambda'])  # 0  # spread out
f_lambda = float(parameters['f_lambda'])
bool_f_statistics = parameters.getboolean('bool_f_statistics')
#bool_f_statistics = False
print('bool_f_statistics:', bool_f_statistics)
top_d = int(parameters['top_d'])
# do_merge = parameters.getboolean('do_merge')
do_l2 = parameters.getboolean('do_l2')
do_cos = parameters.getboolean('do_cos')
# ### Uncomment only one of these
(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
# (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" % (name))


# ##### DATA: Target data and generator input data
def get_distribution_sampler(mu, sigma):
    # Gaussian
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


def get_generator_input_sampler():
    # Uniform-dist data into generator, _NOT_ Gaussian
    return lambda m, n: torch.rand(m, n)


hd = [w.float() for w in jpw.create_haar_dictionary(
    10, vectorType=Variable).values()]


def subsample(matrix, subsample_stride=1, flip_long=True):
    sx = subsample_stride
    sy = subsample_stride
    if type(subsample_stride) == tuple:
        sx = subsample_stride[0]
        sy = subsample_stride[1]
    if(matrix.shape[0] < matrix.shape[1]):
        matrix = matrix.transpose(1, 0)
        temp = sx
        sx = sy
        sy = temp
    return matrix[::sx, ::sy]


# ##### MODELS: Generator model and discriminator model
def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


def batch_cosine(data, normalize=True):
    if normalize:
        # temp = data / (1e-10 + data.matmul(data.t()).sum(1, keepdim=True).pow(0.5))
        temp = F.normalize(data)
    else:
        temp = data
    return temp.matmul(temp.t())


def batch_2norm(x):
    ''' Compute 2-norms of rows of x '''
    return (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(2)


def tsne_functional(sigma2, dist='normal'):
    '''
    Traditionally sigma2 is determined by preprocessing to find perplexity.
    '''
    def tsne_similarity(data):
        '''
        Input: batch of hidden representations. Computes the Gaussian similarity for p_{i|j} and then averages p_{i|j} and p_{j|i}.
        Returns: the log p_{ij} values
        Note the t-sne formulation calculates p_{ij} up front, but this is not feasible when the number of datapoints is too large.
        We approximate it batchwise instead
        '''
        # pdb.set_trace()
        if dist == 'normal':
            numer = torch.exp(-batch_2norm(data) / 2 / sigma2) - \
                torch.eye(data.shape[0]).to(data.device.type)
        elif dist == 't':
            numer = torch.pow(1 + batch_2norm(data), -1) - \
                torch.eye(data.shape[0]).to(data.device.type)
        numer = (1 - 1e-16) * numer + 1e-16 / (numer.shape[0] - 1)
        denom = numer.sum(1, keepdim=True)
        # avoid negative infinities; diagonal is ignored in tsne_kl so long as not inf or nan
        numer = numer + torch.eye(data.shape[0]).to(data.device.type)
        return torch.log(0.5 * (numer / denom + numer / denom.t()))
    return tsne_similarity


def tsne_kl(x, y):
    ''' x and y are in log probability space; this ignores the diagonal '''
    if torch.isnan(torch.exp(x)).any():
        pdb.set_trace()
        x = x.clamp(max=10)
    return ((torch.ones(x.shape[0]).to(x.device.type) - torch.eye(x.shape[0]).to(x.device.type)) * (torch.exp(x) * (x - y))).sum()


def arangeIntervals(stop, step):
    numbers = np.arange(stop, step=step)
    if np.any(numbers == stop):
        pass
    else:
        numbers = np.concatenate((numbers, [stop]))
    return zip(numbers[:-1], numbers[1:])


# mynoise = Variable(gi_sampler(data_size, g_input_size))
mynoise = mydata  # autoencoder
g_input_size = mynoise.shape[1]

outputter_enforce_pd = False
# outputter_enforce_pd = True  # if you want the outputter to be injective up to rank |H|
outputter_enforce_pd_str = '' if not outputter_enforce_pd else '_pdon'

# d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()
Gen = Generator(input_size=g_input_size, hidden_size=g_output_size,
                output_size=g_output_size, hd=hd)
Out = Outputter(input_size=o_input_size, hidden_size=o_hidden_size,
                output_size=o_output_size, injectivity_by_positivity=outputter_enforce_pd)
Clu = Clusterer(input_size=c_input_size,
                hidden_size=c_hidden_size, output_size=c_output_size)

# using_tsne = False
# Enf = Enforcer(nn.MSELoss(), batch_cosine)  # cosine sim
using_tsne = True
tsne_sigma2 = 1
Enf = Enforcer(tsne_kl, tsne_functional(np.nan, 't'),
               tsne_functional(tsne_sigma2, 'normal'))  # t-SNE objective

using_tsne_str = '' if using_tsne is False else '_tsne' + str(tsne_sigma2)

tzero = Variable(torch.zeros(1))
if gpu_mode:
    Gen = Gen.cuda()
    # Gen.permuteTensor = Gen.permuteTensor.cuda()
    # Gen.hd = Gen.hd.cuda()
    Out = Out.cuda()
    Clu = Clu.cuda()
    tzero = tzero.cuda()


MSE = nn.MSELoss()

o_optimizer = optim.Adam(itertools.chain(Out.parameters(), Gen.parameters()),
                         lr=o_learning_rate, weight_decay=1e-8)  # , weight_decay=1e-3)
c_optimizer = optim.Adam(itertools.chain(
    Clu.parameters(), Gen.parameters()), lr=c_learning_rate, weight_decay=1e-10)


# take care of the result strings
logdir = 'simulation_output'
if not os.path.exists(logdir):
    os.makedirs(logdir)
result_base = os.path.basename(config_file_name)
# result_str = 'c_lr_' + str(c_learning_rate) + '_s_lmda_' +\
#     str(s_lambda) + '_f_lmda_' + str(f_lambda) + '_eps_' + str(num_epochs) + \
#     '_merge_' + str(do_merge)
result_str = os.path.splitext(result_base)[0]
boot_prefix = os.path.basename(bootdata)
boot_prefix = os.path.splitext(boot_prefix)[0]
if doTrain:
    result_str = 'train_' + result_str
if doBoot:
    result_str = 'boot_' + boot_prefix + '_' + result_str

if not bool_f_statistics:
    result_str = 'hard'

result_file = 'results/' + result_str + '.txt'
rfile = open(result_file, "w")
rfile.close()
save_losses = []


total_batches = data_size / minibatch_size
for epoch in range(num_epochs):
    # print('epochs')
    epoch_losses = Variable(torch.zeros(6))
    if gpu_mode:
        epoch_losses = epoch_losses.cuda()

    gcounter, ccounter = 0, 0
    fcounter = 0
    g_epoch_indices = torch.LongTensor(np.random.choice(
        data_size, size=data_size, replace=False))
    count_batch_time = 0

    for i0, i1 in arangeIntervals(data_size, minibatch_size):
        count_batch_time += 1
        g_minibatch_epoch_indices = g_epoch_indices[i0:i1]
        if outputter_enforce_pd:
            Out.enforce_pd()

        # zeros the gradient buffers of all parameters
        Gen.zero_grad()
        Out.zero_grad()
        Clu.zero_grad()

        # batch_noise = mynoise[g_minibatch_epoch_indices]
        batch_mydata = mydata[g_minibatch_epoch_indices]
        if gpu_mode:
            # batch_noise = batch_noise.cuda()
            batch_mydata = batch_mydata.cuda()
            g_minibatch_epoch_indices = g_minibatch_epoch_indices.cuda()

        # feedfoward Gen
        hidden = Gen(batch_mydata)

        if np.random.uniform() < pr_g_update:  # which is 1, always T

            g_norms = torch.norm(hidden, 2, 1)
            g_loss = g_lambda * torch.max(g_norms,
                                          1 - torch.log(g_norms + 1e-10)).mean()
            # spread out (batch)
            s_loss = s_lambda * torch.pow(batch_cosine(hidden).mean() - 0.5, 2)
            gcounter += 1
            # print('calculating s_loss')
        else:
            g_loss, s_loss = tzero, tzero
            # print('not calculating s_loss')

        # feedfoward Enf
        if not c_only:  # now c_only=F
            output = Out(hidden)
            # L_{x, \hat{x}}
            o_loss = o_lambda * MSE(output, batch_mydata)
            o_alone = o_lambda * MSE(output, batch_mydata)
            # Enforce H and M similarity
            # e_loss = tzero
            if do_cos:
                #L_{z, x}
                e_loss = cos_lambda * (batch_cosine(batch_mydata) -
                                       batch_cosine(hidden)).pow(2).sum(1).mean()
                o_loss += e_loss
            if do_l2:
                e_loss = l2_lambda * (batch_2norm(batch_mydata) -
                                      batch_2norm(hidden)).pow(2).sum(1).mean()
                o_loss += e_loss

            o_loss += g_loss
            o_loss.backward(retain_graph=True)
            [p.grad.clamp_(-1, 1)
             for p in Gen.parameters() if p.grad is not None]
            # print("check use enf")
            o_optimizer.step()
        else:
            o_loss, e_loss = tzero, tzero

        # feedfoward Clu
        if np.random.uniform() < pr_c_update:  # 1, so always to T
            clusters = Clu(hidden)
            c_loss = c_l2_lambda * clusters.sum(0).pow(2).mean()
            if epoch < burn_in:
                pass
            else:
                # L_||.||
                c_loss += c_lambda * (
                    batch_cosine(torch.sqrt(clusters + 1e-10), normalize=False) -
                    F.relu(batch_cosine(hidden))).pow(2).sum(1).mean()
            # L_{entropy}
            c_e_loss = s_lambda * ((clusters * torch.log(clusters + 1e-10)).mean() +
                                   0.8 / minibatch_size).pow(2)  # Match on desired

            c_loss += c_e_loss
            # F-statistics loss

            tmp_f = g_loss - g_loss
            if bool_f_statistics and epoch > 20:
                unique_counts = torch.round(clusters.sum(dim=0))
                count_no_zero_class = torch.sum(unique_counts > 0)
                if count_no_zero_class == unique_counts.shape[0]:
                    # print('True f')sside
                    f_loss_fn = Soft_F_Loss()
                    # DEBUG: changed chidden to hidden
                    f_loss = f_loss_fn(hidden, clusters,
                                       top_d, epoch, num_epochs, doTrain,
                                       count_batch_time, result_str, total_batches)
                    # print('f_loss:', f_loss)
                    # print("f_loss:", f_loss)
                    if c_loss == float('nan') or f_loss == float('nan') or c_loss == float('inf') or f_loss == float('inf'):
                        print("c_loss before:{}, f_loss:{},epoch{}".format(
                            c_loss, f_loss, epoch))
                        sys.exit(0)
                    c_loss = c_loss + f_lambda * f_loss
                    tmp_f = f_lambda * f_loss
                    fcounter += 1
                    # print("f_loss:{}".format(f_loss))
                # print("chidden ")
            # c_loss += o_loss
            c_loss.backward()
            c_optimizer.step()
            ccounter += 1
        else:
            c_loss = tzero
        # this is added per batch per epoch
        # print("dims:g:{}, s:{}, o:{}, e:{}, c:{}, f:{}".format(g_loss.shape,
        #                                                       s_loss.shape,
        #                                                       o_alone.shape,
        #                                                       e_loss.shape,
        #                                                       c_loss.shape, tmp_f.shape))
        epoch_losses += torch.stack((g_loss,  o_alone, e_loss, c_loss,
                                     tmp_f, o_loss + c_loss))
    print("count batch time: ", count_batch_time)
    if epoch % print_interval == 0:
        el = epoch_losses.cpu().data.numpy()
        # save_losses.append(el)
        # el *= 1 * minibatch_size / data_size *\
        #     np.array([1. * data_size / minibatch_size / gcounter,
        #               1. * data_size / minibatch_size / gcounter,
        #               1,
        #               1,
        #               1. * data_size / minibatch_size / ccounter,
        #               1. * data_size / minibatch_size / fcounter,
        #               1])
        save_losses.append(el)
        print("%s: [H: %6.4f]; [O: %6.4f; E: %6.4f]; C: %6.4f, f:%6.4f, total: %6.4f" %
              (epoch, el[0], el[1], el[2], el[3], el[4], el[5]))



stacked_loss = np.stack(save_losses, axis=0)
print('stacked_loss shape:', stacked_loss.shape)
np.save('loss_pics/' + result_str + '_losses.npy', stacked_loss)
lbs = ['H', 's', 'O', 'e', 'C', 'f', 'TOTAL']
for i in range(stacked_loss.shape[1]):
    plt.plot(
        np.arange(0, stacked_loss.shape[0]), stacked_loss[:, i], label=lbs[i])
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.savefig('loss_pics/' + result_str + '.png')
print("fig:", 'loss_pics/' + result_str + '.png')

plt.clf()
plt.plot(np.arange(0, stacked_loss.shape[0]), stacked_loss[:, -2], label="f")
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
plt.savefig('loss_pics/' + result_str + '_f.png')
print("fig:", 'loss_pics/' + result_str + '_f.png')


prefix = logdir + '/' + result_str
Gen = Gen.eval()
assignments = np.zeros(mynoise.shape[0])
for i0, i1 in arangeIntervals(data_size, 100):
    assignments[i0:i1] = np.argmax(
        Clu(Gen(mynoise[i0:i1].to(device))[:, side_channel_size:]).
        cpu().data.numpy(), axis=1)
    # for i0, i1 in arangeIntervals(data_size, minibatch_size):
print(np.bincount(assignments.astype(int)))
# print("np assignment type: ", assignments)


# Get hidden:
hiddenVectors = np.zeros((mynoise.shape[0], hidden_size))
for i0, i1 in arangeIntervals(data_size, 100):
    hiddenVectors[i0:i1] = Gen(mynoise[i0:i1].to(device)).cpu().data.numpy()
npihiddendf = pd.DataFrame(hiddenVectors)
# npihiddendf['truth'] = 'NA'
# npihiddendf.to_csv(prefix + 'hidden.csv')
np.save('hidden/' + result_str + '.npy', hiddenVectors)

# Get kmeans
from sklearn.cluster import MiniBatchKMeans
mydatanumpy = mydata.cpu().data.numpy()
kmeans = MiniBatchKMeans(n_clusters=desired_centroids,
                         random_state=0).fit(mydatanumpy)
assignments_kmeans = kmeans.labels_
npidf = pd.DataFrame({
    'cluster': assignments_kmeans})
npidf.to_csv(prefix + 'clusters_kmeans.csv')


# Get hidden kmeans
# hiddenkmeans = MiniBatchKMeans(n_clusters=desired_centroids).fit(hiddenVectors)
# assignments_hidden_kmeans = hiddenkmeans.labels_
# npidf = pd.DataFrame({
#     'cluster': assignments_hidden_kmeans})
# npidf.to_csv(prefix + 'clusters_hidden_kmeans.csv')

# Get our method
npidf = pd.DataFrame({
    'cluster': assignments})
npidf.to_csv(prefix + 'clusters_ours.csv')


# Get our method merged  # Post process merge clusters
print('Post-process to get k clusters: ',
      len(np.unique(assignments)), ' -> ', desired_centroids)
mydata = mydata.to(device)
ncs = c_output_size
merged_assignments = assignments.copy()
Gen = Gen.cuda()
# Gen.permuteTensor = Gen.permuteTensor.cuda()
# Gen.hd = Gen.hd.cuda()
Out = Out.cuda()
Clu = Clu.cuda()
sim_dim = 100


npidf = pd.DataFrame({
    'cluster': merged_assignments})
npidf.to_csv(prefix + 'clusters_ours_merged.csv')

# sihouette score for our merged method
small = -1000
s_score_merged, s_score_kmeans, s_score_kmeans_cs, s_score_merged_cs = small, small, small, small

if len(np.unique(merged_assignments.astype(int))) > 1:

    s_score_merged = silhouette_score(mydata.cpu().numpy(), merged_assignments,
                                      metric='l1')
    s_score_kmeans = silhouette_score(mydata.cpu().numpy(), assignments_kmeans,
                                      metric='l1')
    s_score_merged_cs = silhouette_score(
        mydata.cpu().numpy(), merged_assignments, metric='cosine')
    s_score_kmeans_cs = silhouette_score(
        mydata.cpu().numpy(), assignments_kmeans, metric='cosine')
    print("ours:[{}], km:[{}]".format(s_score_merged, s_score_kmeans))
    print("ours:[{}], km:[{}]".format(s_score_merged_cs, s_score_kmeans_cs))

result_file = 'results/' + result_str + '.txt'
file = open(result_file, "w")

file.write("c_lr: {}, s_lmda: {}, f_lmda: {}, epochs: {}\n".format(
    c_learning_rate, s_lambda, f_lambda, num_epochs))

file.write("sihouette k_means: our_merged: {}, \
           km: {}\n".format(s_score_merged, s_score_kmeans))
file.write("sihouette_score_cosine: our_merged: {},\
           km: {}\n".format(s_score_merged_cs, s_score_kmeans_cs))
file.write("centroids: {}".format(
    str(np.bincount(merged_assignments.astype(int)))))
file.close()

print('cluster does not make sense but still saved, filename:{}', result_file)


# The other metrices
from sklearn import metrics
# The higher the CHI the better
CHI_our = metrics.calinski_harabasz_score(mydata.cpu().numpy(),
                                          merged_assignments)
# The lower the DBI he better
DBI_our = metrics.davies_bouldin_score(mydata.cpu().numpy(),
                                       merged_assignments)
CHI_km = metrics.calinski_harabasz_score(mydata.cpu().numpy(),
                                         assignments_kmeans)
# The lower the DBI he better
DBI_km = metrics.davies_bouldin_score(mydata.cpu().numpy(),
                                      assignments_kmeans)
# not merged
print('the higher the CHI the better, the lower the DBI the better')
print('CHI_our:{}, CHI_km:{}, DBI_our:{}, DBI_km:{}'.format(CHI_our, CHI_km,
                                                            DBI_our, DBI_km))

