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


class F_Loss(torch.nn.Module):

    def __init__(self):
        super(F_Loss, self).__init__()
        # self.hidden = hidden
        # self.assignment = assignment

    # def forward(self, outputs, labels):
    #    return
    # hidden =
    # z_ij = embedding coordinate of instance j of class i
    # hidden = embedding cordinate of batch n = (ixj)
    def get_z(self, hidden, batch_ids, unique_classes, idx):
        # print("unique_classes:{}, batch_ids:{}".format(unique_classes,
        # batch_ids))

        z_list = []
        for i, class_id in enumerate(unique_classes):
            z_list.append(hidden[batch_ids == class_id])
        #print("z_list: ", z_list)
        return z_list

    def forward(self, hidden=None, batch_ids=None, d=None, epoch=None,
                numEpoch=None, train=False, count_batch=None):
        unique_classes, idx = torch.unique(batch_ids, return_inverse=True)
        # z_nd = tf.shape(z)[1]
        z_list = self.get_z(hidden, batch_ids, unique_classes, idx)
        # print("hidden require_grad: {}".format(hidden.requires_grad))

        def process_class_within(class_ids):
            # z_list[class_id]: batch of class_id x H
            # print("hidden require_grad: {}".format())
            class_means = []
            within_diffs = []
            unique_counts = []
            #print("class_ids: ", class_ids)
            #print("z_list: ", z_listb)
            for class_id in class_ids:
                # print("z_list[class_id] size: {}".format(
                #    z_list[class_id].size()))
                this_class_mean = torch.mean(z_list[class_id], dim=0)
                within_class_diffs = (z_list[class_id] - this_class_mean)**2
                within_class_diff_sum = torch.sum(within_class_diffs, dim=0)
                class_means.append(this_class_mean)
                within_diffs.append(within_class_diff_sum)
                unique_counts.append(z_list[class_id].size()[0])
                # print("unique_counts:{}".format(unique_counts))
            return torch.stack(class_means), torch.stack(within_diffs), torch.tensor(unique_counts, dtype=hidden.dtype, device=hidden.device)

        class_means, within_diffs, unique_counts = process_class_within(
            np.arange(len(unique_classes)))
        # class_means, within_diffs, unique_counts = process_class_within(
        #    unique_classes)

        # get class_means, within_class_diff_sum

        def compute_f_pairs(class_ids, epoch, numEpoch, train, count_batch):
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
                # print("cidx:{}, l_count:{}, unique_counts:{}".format(cidx,
                #                                                     l_count, unique_counts))
                # print("lb:{} lc:{} rb:{}, rc:{}".format(l_between.size(),
                #                                        l_count.size(),
                #                                        r_betweens.size(), r_counts.size()))
                # print("size and type:l_b:{},{},l_c:{}{},r_bs:{}{},r_cs:{}{}".format(l_between.size(), l_between.type(
                # ), l_count.size(), l_count.type(), r_betweens.size(),
                #    r_betweens.type(), r_counts.size(), r_counts.type()))
                pair_between_diffs = l_between * l_count + \
                    r_betweens * r_counts.unsqueeze(1)
                # requires attention for reshape for above
                pair_counts = l_count + r_counts
                # print("pair_counts:{}".format(bpair_counts))
                # print("PBDs: {} PWDs: {} PCs: {} RMs:{}".format(pair_between_diffs.size(
               # ), pair_within_diffs.size(), pair_counts.size(), r_means.size()))
                pair_cdfs, topk_inds = self.compute_pair_f(
                    (pair_between_diffs, pair_within_diffs, pair_counts,
                     r_means), d, epoch)
                if epoch >= numEpoch - 3:
                    top_list.append(topk_inds.data.cpu().numpy())
            # if epoch == numEpoch-1:
            # if train and epoch >= numEpoch-3:  # save the topd files
                pair_cdf_list.append(torch.sum(pair_cdfs))

            if epoch >= numEpoch-3:  # save the topd files
                # top_list = torch.stack(top_list)
                # print("top_list: ", top_list)
                fname = 'dimensions/' + 'ep_' + \
                    str(epoch) + '_b_id_' + str(count_batch) + '_top_d.txt'
                # np.savetxt(fname, np.array(top_list))

                with open(fname, 'w') as f:
                    # rint >> f, 'Filename:', filename     # Python 2.x
                    print("unique classes: {}".format(unique_classes), file=f)
                    print(top_list, file=f)  # Python 3.x
                # file = open(fname, 'w')
                # # file.write(top_list)
                # file.write(','.join(top_list))
                # #simplejson.dump(top_list, file)
                # file.close()

            # print("pair_cdf_list: ", pair_cdf_list)
            return pair_cdf_list
        # finally calculate the -logs to be returned
        cdfs = compute_f_pairs(
            np.arange(len(unique_classes) - 1), epoch, numEpoch, train, count_batch)
        sum_cdfs = -torch.sum(torch.stack(cdfs))
        # if epoch == 10 or epoch == 11:
        #    print("cdfs:{},sum_cdfs:{},unique_classes:{}".format(cdfs,
        #                                                         sum_cdfs, unique_classes))
        # print("unique_classes:{},cdfs: {}".format(unique_classes, cdfs))
        # if cdfs = 0:  # only one class scenario
        #    cdfs = 0

        return -torch.sum(torch.stack(cdfs))  # if len(cdfs) > 0 else 0

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
