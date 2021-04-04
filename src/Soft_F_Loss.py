from importlib import reload
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from betainc import Betainc
reload(sys.modules['betainc'])
from betainc import Betainc
import simplejson


class Soft_F_Loss(torch.nn.Module):

    def __init__(self):
        super(Soft_F_Loss, self).__init__()
        # self.hidden = hidden
        # self.assignment = assignment

    def forward(self, hidden=None, cluster=None, d=20, epoch=None,
                numEpoch=None, train=False, count_batch=None, result_str=None, total_batches=None):
        # hidden shape: N*D, cluster.shape: N*C
        shidden = hidden.unsqueeze(-1)
        scluster = cluster.unsqueeze(1)
        softHidden = shidden * scluster
        #print('softHidden.shape(expect: N*D*C):', softHidden.shape)
        # process_class_within
        class_means = torch.sum(softHidden, dim=0, keepdim=True)
        #print('class_means.shape(expect: 1*D*C):', class_means.shape)
        within_diffs = (softHidden - class_means)**2
        within_diffs = within_diffs.sum(dim=0, keepdim=True)
        #print('within diffs.shape(expect: 1*D*C):', within_diffs.shape)
        # Done with denominator here
        #print('not round: ', cluster.sum(dim=0))
        unique_counts = torch.round(cluster.sum(dim=0))  # (shape: 1 * 4)
        #print('unique_counts.shape(expect: c)', unique_counts.shape)
        class_means = class_means.squeeze(0).T
        within_diffs = within_diffs.squeeze(0).T
        #print('unique counts:', unique_counts)

        def compute_f_pairs(class_ids, epoch, numEpoch, train, count_batch):
            #print('class_means.shape(expect: C*D):', class_means.shape)
            #print('within diffs.shape(expect: C*D):', within_diffs.shape)

            pair_cdf_list = []
            top_list = []
            for cidx in class_ids:
                l_mean = class_means[cidx]
                r_means = class_means[cidx + 1:]
                l_within = within_diffs[cidx]
                r_withins = within_diffs[cidx + 1:]
                # pair_global_means in num
                pair_global_means = (l_mean + r_means) / 2.0
                l_between = (l_mean - pair_global_means)**2
                r_betweens = (r_means - pair_global_means)**2
                pair_within_diffs = l_within + r_withins
                l_count = unique_counts[cidx]
                r_counts = unique_counts[cidx + 1:]

                pair_between_diffs = l_between * l_count + \
                    r_betweens * r_counts.unsqueeze(1)
                # requires attention for reshape for above
                pair_counts = l_count + r_counts

                pair_cdfs, topk_inds = self.compute_pair_f(
                    (pair_between_diffs, pair_within_diffs, pair_counts,
                     r_means), d, epoch)
                if epoch >= numEpoch - 1 and count_batch >= total_batches-2:
                    top_list.append(topk_inds.data.cpu().numpy())
            # if epoch == numEpoch-1:
            # if train and epoch >= numEpoch-3:  # save the topd files
                pair_cdf_list.append(torch.sum(pair_cdfs))

            if epoch >= numEpoch-1 and count_batch >= total_batches-2:  # save the topd files
                    # top_list = torch.stack(top_list)
                    # print("top_list: ", top_list)
                fname = 'dimensions/' + result_str + '_ep_' + \
                    str(epoch) + '_b_id_' + str(count_batch) + '_top_d.txt'
                # np.savetxt(fname, np.array(top_list))

                with open(fname, 'w') as f:
                        # rint >> f, 'Filename:', filename     # Python 2.x
                    #print("unique classes: {}".format(unique_classes), file=f)
                    print(top_list, file=f)  # Python 3.x
            return pair_cdf_list
        cdfs = compute_f_pairs(
            np.arange(cluster.shape[1] - 1), epoch, numEpoch, train, count_batch)
        sum_cdfs = -torch.sum(torch.stack(cdfs))
        # return 0.1
        return sum_cdfs

    def compute_pair_f(self, items, d=None, epoch=None):
        results = []
        # #print("What does items look like: {}".format(items))
        betweens, withins, pair_counts, r_means = items
        # if epoch == 11:
        #    print("pair_counts: {}".format(pair_counts))
        topk_ind_list = []
        for pair_id in range(betweens.size()[0]):
            between, within, pair_count, r_mean = betweens[pair_id], withins[
                pair_id], pair_counts[pair_id], r_means[pair_id]
            d1 = torch.tensor(1.0)
            # if epoch == 11:
            #    print("Pair_count: {}".format(pair_count))
            d2 = pair_count - 2.0
            if d2 == 0.0:
                d2 += 1.0e-5
            x = between / (between + within)
            xmin = 1.0e-37
            xmax = 1. - 1.0e-5
            x_limit = torch.clamp(x, xmin, xmax)
            # print("x_limit: ", x_limit)
            xbetainc = Betainc.apply(x_limit, d1 / 2.0, d2 / 2.0)
            # gradcheck_result = torch.autograd.gradcheck(
            #     Betainc, (x_limit, d1 / 2.0, d2 / 2.0))
            # print("xbetainc: {}".format(xbetainc))
            # print("##########gradcheck_result###############\n:{}".format(
            #     gradcheck_result))
            top_k, top_k_ind = torch.topk(xbetainc, k=d, sorted=False)
            # print("top_k_ind: ", top_k_ind)
            result = torch.sum(torch.log(top_k))
            results.append(result)
            topk_ind_list.append(top_k_ind)

            # if epoch == 11:
            # print("result: {}, top_k:{}, xbetainc:{}, x_limit:{}".format(
            #     result, top_k, xbetainc, x_limit))
            # sys.exit(0)
        return torch.stack(results), torch.stack(topk_ind_list)
