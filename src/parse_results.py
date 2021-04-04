import argparse
import subprocess
import configparser
import glob
import os
import re
import numpy as np
from sklearn import preprocessing
import pandas as pd
from evaluations import get_modularity
from dimension_results import parse_dimensions
from hidden import get_hidden_map
import sys


def parse_results(multiple=3):
    allconfigs = glob.glob('results/*.txt')
    # mylines = []
    myfiles = []
    ss_l2_list = []
    ss_cos_list = []
    fnames = []
    print('Num all results: ', len(allconfigs))
    for i in range(len(allconfigs)):
        # fnames.append(allconfigs[i])
        mylines = []
        with open(allconfigs[i], 'rt') as myfile:
            for myline in myfile:
                mylines.append(myline)
        # text =
        # print("myline: ", mylines[1])
        # multiple = 4.0
        if len(mylines) > 0:
            inner1 = re.findall(
                r"our_merged: (\-?\d+\.\-?\d+)", mylines[1])
            inner2 = re.findall(
                r"our_merged: (\-?\d+\.\-?\d+)", mylines[2])
            if len(inner1) != 0 and len(inner2) != 0:
                our_ss_l2 = float(inner1[0])
                # print("our ss l2: ", our_ss_l2)
                our_ss_cos = float(inner2[0])

                cluster_ns = re.findall(r"\d+", mylines[3])
                cluster_ns = [int(i) for i in cluster_ns]
                # print("ss_line: ", mylines[1])
                # sorted

                cluster_ns.sort(reverse=True)
                # print('cluster_ns:', cluster_ns)
                # ensure the cluster numbers are relatively even
                if cluster_ns[-1] != 0 and cluster_ns[0] / cluster_ns[-1] <= multiple and len(cluster_ns) == 4:
                    ss_l2_list.append(our_ss_l2)
                    ss_cos_list.append(our_ss_cos)
                    myfiles.append(mylines)
                    fnames.append(allconfigs[i])
    max_id = np.argmax(np.array(ss_l2_list))
    # print("best result: ", myfiles[max_id])
    sort_id = np.argsort(np.array(ss_l2_list))

    # print("best cofig: ", np.array(myfiles)[sort_id])
    sorted_array = np.array(myfiles)[sort_id]
    # print("best config: ", sorted_array[:, 0])
    print("best values cosine: ", np.array(ss_cos_list)[sort_id])
    print("associated l2:", np.array(ss_l2_list)[sort_id])
    print('num available files: ', len(ss_cos_list))
    sorted_names = np.array(fnames)[sort_id]
    print("\nbest config name: ", sorted_names[-1])
    return sorted_names
# print("our_ss_l2:", our_ss_l2)x
# print("ss_line[2]: ", mylines[2])
# print("our_cos_l2:", our_ss_cos)

# our_ss
# for i in range(len(allconfigs)):
#    filename = allconfigs[i]

# currently only try to get modularity of the best file


def get_mod_result(result_str, train=False):
    hidden_fname = 'hidden/' + result_str + '.npy'
    mod_fout = 'modularity/' + result_str + '.txt'
    mod_fname = 'modularity/' + result_str + '.csv'
    print('name: ', hidden_fname)
    z = np.load(hidden_fname)
    # xdata = np.genfromtxt(
    #    "/Users/Serena/Dropbox/x_data_not_normed_notime.csv", delimiter=',')
    xdata = None
#    if train:
#        xdata = np.genfromtxt("/home/cc/sepsis_data/xtrain_not_normed_notime.csv",
    #                        delimiter = ",")
#    else:
#        xdata = np.genfromtxt(
#           "/home/cc/sepsis_data/xval_not_normed_notime.csv", delimiter=',')
#    xdata = xdata[range(1, xdata.shape[0]), ]
 #   xdata = preprocessing.normalize(xdata)
    # xdata =     xdata = np.genfromtxt(
    # f1 =
    if train:
        xdata = np.genfromtxt(
            '/home/cc/sepsis_data/xscaled.csv', delimiter=',')
    else:
        xdata = np.genfromtxt(
            '/home/cc/sepsis_data/xscaled.csv', delimiter=',')

    xdata = xdata[range(1, xdata.shape[0]), ]

    print("shape: xdata {}, z {}".format(xdata.shape, z.shape))

    # the step
    score, mean_scores = get_modularity(z, xdata, result_str)

    # save files
    npidf = pd.DataFrame({
        'modularity': score})
    # modprefix = 'modularity/' + result_str
    npidf.to_csv(mod_fname)
    # return mean_scores
    file = open(mod_fout, "w")
    file.write("mean modularity: {}".format(mean_scores))


def dlc_mv():
    allmods = glob.glob('dlc_mod/*.txt')
    print('num files: ', len(allmods))
    modList = []
    for i in range(len(allmods)):
        line = ''
        with open(allmods[i]) as fp:
            line = fp.readline()
        if line != '':
            mod = float(re.findall(r"\d+\.\d+", line)[0])
            modList.append(mod)
    modList = np.array(modList)
    print('modList', modList, 'num results: ', modList.shape[0])
    print('dlc mod mean:', np.mean(modList), 'std:', np.std(modList))


def hc_bootmod_result(filename):

    all_data = glob.glob('../data/cls/*.npy')
    print('num files:', len(all_data))
    for i in range(len(all_data)):
        bootfile = all_data[i]
        print('boot file:', bootfile)
        xdata = np.load(bootfile)

        # cbootfile = "--data={}".format(bootfile)
        base = os.path.basename(filename)
        name_str = os.path.splitext(base)[0]
        boot_prefix = os.path.basename(bootfile)
        boot_prefix = os.path.splitext(boot_prefix)[0]
        name_str = 'boot_' + boot_prefix + '_' + 'hard'
        z = np.load('hidden/' + name_str + '.npy')

        print("shape: xdata {}, z {}".format(xdata.shape, z.shape))
        # the step
        result_str = 'dhc_' + boot_prefix
        mod_fname = 'hc_mod/' + result_str + '.csv'
        mod_fout = 'hc_mod/' + result_str + '.txt'
        score, mean_scores = get_modularity(z, xdata, result_str)

        # save files
        npidf = pd.DataFrame({
            'modularity': score})
        # modprefix = 'modularity/' + result_str
        npidf.to_csv(mod_fname)
        # return mean_scores
        file = open(mod_fout, "w")
        file.write("mean modularity: {}".format(mean_scores))


def dhc_bootmod_result(filename):

    all_data = glob.glob('../data/cls/*.npy')
    print('num files:', len(all_data))
    for i in range(len(all_data)):
        bootfile = all_data[i]
        print('boot file:', bootfile)
        xdata = np.load(bootfile)

        # cbootfile = "--data={}".format(bootfile)
        base = os.path.basename(filename)
        name_str = os.path.splitext(base)[0]
        boot_prefix = os.path.basename(bootfile)
        boot_prefix = os.path.splitext(boot_prefix)[0]
        name_str = 'boot_' + boot_prefix + '_' + name_str
        z = np.load('hidden/' + name_str + '.npy')

        print("shape: xdata {}, z {}".format(xdata.shape, z.shape))
        # the step
        result_str = 'dhc_' + boot_prefix
        mod_fname = 'dhc_mod/' + result_str + '.csv'
        mod_fout = 'dhc_mod/' + result_str + '.txt'
        score, mean_scores = get_modularity(z, xdata, result_str)

        # save files
        npidf = pd.DataFrame({
            'modularity': score})
        # modprefix = 'modularity/' + result_str
        npidf.to_csv(mod_fname)
        # return mean_scores
        file = open(mod_fout, "w")
        file.write("mean modularity: {}".format(mean_scores))

    # allhidden = glob.glob('../DCN_New/hidden/boot*reshaped.npy')
    # # allhidden = [
    # #    '../DCN_New/hidden/val_lbd_0.1_beta_0.001_pre_eps_100_h_4_ep_200_reshaped.npy']
    # print('num hidden:{}'.format(len(allhidden)))
    # for i in range(len(allhidden)):
    #     hidden_fname = allhidden[i]
    #     result_base = os.path.basename(hidden_fname)
    #     print('result_base', result_base)
    #     result_str = os.path.splitext(result_base)[0]
    #     result_str = result_str.split('_reshaped')[0]
    #     print('result_base', result_str)
    #     mod_fout = 'dlc_mod/' + result_str + '.txt'
    #     # if not os.path.isfile(mod_fout):
    #     file = open(mod_fout, "w")
    #     file.close()
    #     mod_fname = 'dlc_mod/' + result_str + '.csv'
    #     print('name: ', hidden_fname)
    #     z = np.load(hidden_fname)
    #     # xdata = np.genfromtxt(
    #     #    "/Users/Serena/Dropbox/x_data_not_normed_notime.csv", delimiter=',')
    #     # xdata = None
    #     xdata = np.load(
    #         '../DCN_New/dlc_data/{}.npy'.format(result_str))
    #     # xdata = np.genfromtxt(
    #     #    '/home/cc/sepsis_data/xscaled.csv', delimiter=',')
    #     #xdata = xdata[range(1, xdata.shape[0]), ]
    #     # xdata = preprocessing.normalize(xdata)

    #     print("shape: xdata {}, z {}".format(xdata.shape, z.shape))

    #     # the step
    #     score, mean_scores = get_modularity(z, xdata, result_str)

    #     # save files
    #     npidf = pd.DataFrame({
    #         'modularity': score})
    #     # modprefix = 'modularity/' + result_str
    #     npidf.to_csv(mod_fname)
    #     # return mean_scores
    #     file = open(mod_fout, "w")
    #     file.write("mean modularity: {}".format(mean_scores))
    #     # else:
    #     # print('...already completed...')


def get_dlc_mod_result():
    allhidden = glob.glob('../DCN_New/hidden/boot*reshaped.npy')
    # allhidden = [
    #    '../DCN_New/hidden/val_lbd_0.1_beta_0.001_pre_eps_100_h_4_ep_200_reshaped.npy']
    print('num hidden:{}'.format(len(allhidden)))
    for i in range(len(allhidden)):
        if i < 6:
            continue
        hidden_fname = allhidden[i]
        result_base = os.path.basename(hidden_fname)
        print('result_base', result_base)
        result_str = os.path.splitext(result_base)[0]
        result_str = result_str.split('_reshaped')[0]
        print('result_base', result_str)
        mod_fout = 'dlc_mod/' + result_str + '.txt'
        # if not os.path.isfile(mod_fout):
        file = open(mod_fout, "w")
        file.close()
        mod_fname = 'dlc_mod/' + result_str + '.csv'
        print('name: ', hidden_fname)
        z = np.load(hidden_fname)
        # xdata = np.genfromtxt(
        #    "/Users/Serena/Dropbox/x_data_not_normed_notime.csv", delimiter=',')
        # xdata = None
        xdata = np.load(
            '../DCN_New/dlc_data/{}.npy'.format(result_str))
        # xdata = np.genfromtxt(
        #    '/home/cc/sepsis_data/xscaled.csv', delimiter=',')
        # xdata = xdata[range(1, xdata.shape[0]), ]
        # xdata = preprocessing.normalize(xdata)

        print("shape: xdata {}, z {}".format(xdata.shape, z.shape))

        # the step
        score, mean_scores = get_modularity(z, xdata, result_str)

        # save files
        npidf = pd.DataFrame({
            'modularity': score})
        # modprefix = 'modularity/' + result_str
        npidf.to_csv(mod_fname)
        # return mean_scores
        file = open(mod_fout, "w")
        file.write("mean modularity: {}".format(mean_scores))
        # else:
        # print('...already completed...')


def get_dec_mod_result():
    # allhidden = glob.glob('../DCN_New/hidden/*reshaped.npy')
    # lr=1e-4
    # preeps_50_eps100
    allhidden = [
        '../DEC_pytorch/hidden/lr_0.001_mom_0.1_preeps_100_eps100.npy']
    for i in range(len(allhidden)):
        hidden_fname = allhidden[i]
        result_base = os.path.basename(hidden_fname)
        print('result_base', result_base)
        result_str = os.path.splitext(result_base)[0]
        result_str = result_str.split('_reshaped')[0]
        print('result_base', result_str)
        mod_fout = 'dlc_mod/' + result_str + '.txt'
        # if not os.path.isfile(mod_fout):
        file = open(mod_fout, "w")
        file.close()
        mod_fname = 'dlc_mod/' + result_str + '.csv'
        print('name: ', hidden_fname)
        z = np.load(hidden_fname)
        # xdata = np.genfromtxt(
        #    "/Users/Serena/Dropbox/x_data_not_normed_notime.csv", delimiter=',')
        xdata = None
        xdata = np.load(
            '../DCN_New/dlc_data/{}.npy'.format(result_str))
        # xdata = np.genfromtxt("~/sepsis_data/xscaled.csv")
        # xdata = np.genfromtxt(
        #    '/home/cc/sepsis_data/xscaled.csv', delimiter=',')
        # xdata = xdata[range(1, xdata.shape[0]), ]
        # xdata = preprocessing.normalize(xdata)

        print("shape: xdata {}, z {}".format(xdata.shape, z.shape))

        # the step
        score, mean_scores = get_modularity(z, xdata, result_str)

        # save files
        npidf = pd.DataFrame({
            'modularity': score})
        # modprefix = 'modularity/' + result_str
        npidf.to_csv(mod_fname)
        # return mean_scores
        file = open(mod_fout, "w")
        file.write("mean modularity: {}".format(mean_scores))
        # else:
        # print('...already completed...')


def get_idec_mod_result():
    # allhidden = glob.glob('../DCN_New/hidden/*reshaped.npy')
    # lr=1e-4
    # preeps_50_eps100
    allhidden = [
        '../IDEC-pytorch/hidden/lr_0.001_H4_32_pre_200_epochs_100.npy']
    for i in range(len(allhidden)):
        hidden_fname = allhidden[i]
        result_base = os.path.basename(hidden_fname)
        print('result_base', result_base)
        result_str = os.path.splitext(result_base)[0]
        result_str = result_str.split('_reshaped')[0]
        print('result_base', result_str)
        mod_fout = 'idec_mod/' + result_str + '.txt'
        # if not os.path.isfile(mod_fout):
        file = open(mod_fout, "w")
        file.close()
        mod_fname = 'dlc_mod/' + result_str + '.csv'
        print('name: ', hidden_fname)
        z = np.load(hidden_fname)
        # xdata = np.genfromtxt(
        #    "/Users/Serena/Dropbox/x_data_not_normed_notime.csv", delimiter=',')
        xdata = None
        # xdata = np.load(
        #    '../DCN_New/dlc_data/{}.npy'.format(result_str))
        # xdata = np.genfromtxt("~/sepsis_data/xscaled.csv")
        xdata = np.genfromtxt(
            '/home/cc/sepsis_data/xscaled.csv', delimiter=',')
        xdata = xdata[range(1, xdata.shape[0]), ]
        # xdata = preprocessing.normalize(xdata)

        print("shape: xdata {}, z {}".format(xdata.shape, z.shape))

        # the step
        score, mean_scores = get_modularity(z, xdata, result_str)

        # save files
        npidf = pd.DataFrame({
            'modularity': score})
        # modprefix = 'modularity/' + result_str
        npidf.to_csv(mod_fname)
        # return mean_scores
        file = open(mod_fout, "w")
        file.write("mean modularity: {}".format(mean_scores))
        # else:
        # print('...already completed...')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False,
                        action="store_true")
    parser.add_argument('--doMod', default=False,
                        action="store_true")
    parser.add_argument('--doModHard', default=False,
                        action="store_true")
    parser.add_argument('--dim', default=False, action="store_true")
    parser.add_argument('--bootstrap',
                        default=False, action="store_true")
    parser.add_argument('--doTry',
                        default=False, action="store_true")
    parser.add_argument('--dlcMod',
                        default=False, action="store_true")
    parser.add_argument('--decMod',
                        default=False, action="store_true")
    parser.add_argument('--idecMod',
                        default=False, action="store_true")
    parser.add_argument('--dlcMV',
                        default=False, action="store_true")
    parser.add_argument("multiple", default=3, help="display a square of a given number",
                        type=int)
    parser.add_argument('--hard',
                        default=False, action="store_true")
    parser.add_argument('--dhcBootMod',
                        default=False, action="store_true")
    parser.add_argument('--hcBootMod',
                        default=False, action="store_true")
    parser.add_argument('--dcnBootMod',
                        default=False, action="store_true")
    parser.add_argument('--hcBootstrap',
                        default=False, action="store_true")
    parser.add_argument('--dcnBootstrap',
                        default=False, action="store_true")
    args = parser.parse_args()
    doTrain = args.train
    getMod = args.doMod
    getDim = args.dim
    doBoot = args.bootstrap
    doTry = args.doTry
    dlcMV = args.dlcMV
    decMod = args.decMod
    multiple = args.multiple
    idecMod = args.idecMod
    print('Do train', doTrain)
    fname_vec = parse_results(multiple)
    print("all results: ", fname_vec)

    fname = 'results_10/train_c_ld_1.0_s_ld_0.1_f_ld_10.0_o_1.0_eps_200_dim_10_l2_True_cos_True_l2lr_1e-05_coslr_0.1.txt'
    # the commented out one above is the one for top_10 dimensions

    #fname = 'results/train_c_ld_1.0_s_ld_0.1_f_ld_100.0_o_1.0_eps_150_dim_5_l2_True_cos_True_l2lr_0.0001_coslr_0.01_top_5.txt'

    result_base = os.path.basename(fname)
    result_str = os.path.splitext(result_base)[0]
    # get_mod_result(result_str, doTrain)
    filename = 'config/' + result_str + '.ini'

    # parse_results()]e
    dlcMod = args.dlcMod
    if args.dhcBootMod:
        dhc_bootmod_result(filename)
    if args.hcBootMod:
        hc_bootmod_result(filename)
    if dlcMV:
        dlc_mv()
    if dlcMod:
        get_dlc_mod_result()
    if idecMod:
        get_idec_mod_result()
    if decMod:
        get_dec_mod_result()
    if doTry and not doBoot:
        cfilename = "--filename={}".format(filename)
        # print('filenam')
        # ctry = "--doTry"
        ctrain = "--train=False"
        out = subprocess.run(
            ["python", "disentangled_hypershperical_cluster.py", cfilename, ctrain])
    if doTrain and not doBoot:
        cfilename = "--filename={}".format(filename)
        ctrain = "--train=true"
        out = subprocess.run(
            ["python", "disentangled_hypershperical_cluster.py", cfilename,
             ctrain])
        print('wa')
    if args.hard:
        cfilename = "--filename=config/hard.ini"
        ctrain = "--train=false"
        out = subprocess.run(
            ["python", "disentangled_hypershperical_cluster.py", cfilename,
             ctrain])
        # print('wa')
    if args.hcBootstrap:
        # cfilename = "--filename={}".format(filename)
        cfilename = "--filename=config/hard.ini"
        ctrain = "--train=false"
        # ctrain = "--train=true"
        cboot = "--bootstrap"
        # result_str = 'hard'
        # filename = 'results/hard.txt'
        all_data = glob.glob('../data/cls/*.npy')
        for i in range(len(all_data)):
            bootfile = all_data[i]
            cbootfile = "--data={}".format(bootfile)
            base = os.path.basename(filename)
            name_str = os.path.splitext(base)[0]
            boot_prefix = os.path.basename(bootfile)
            boot_prefix = os.path.splitext(boot_prefix)[0]
            name_str = 'hard_' + boot_prefix + name_str
            print("name_str: ", name_str)
            if not os.path.isfile('results/{}.txt'.format(name_str)):
                print(" ...running: ... ")
                # main.run(args)
                out = subprocess.run(
                    ["python", "disentangled_hypershperical_cluster.py", cfilename,
                     ctrain, cboot, cbootfile])
            else:
                print("...already completed...")
    if args.bootstrap:
        cfilename = "--filename={}".format(filename)
        # cfilename = "--filename=config/hard.ini"
        ctrain = "--train=false"
        # ctrain = "--train=true"
        cboot = "--bootstrap"
        # result_str = 'hard'
        # filename = 'results/hard.txt'
        all_data = glob.glob('../data/cls/*.npy')
        for i in range(len(all_data)):
            bootfile = all_data[i]
            print('bootfile:', bootfile)
            cbootfile = "--data={}".format(bootfile)
            base = os.path.basename(filename)
            name_str = os.path.splitext(base)[0]
            boot_prefix = os.path.basename(bootfile)
            boot_prefix = os.path.splitext(boot_prefix)[0]
            name_str = boot_prefix + name_str
            print("name_str: ", name_str)
#            if not os.path.isfile('results/{}.txt'.format(name_str)):
#                print(" ...running: ... ")
            # main.run(args)
            print('cfilename:', cfilename)
            print('cboot:', cboot)
            out = subprocess.run(
                ["python", "disentangled_hypershperical_cluster.py", cfilename,
                 ctrain, cboot, cbootfile])
#            else:
#                print("...already completed...")

            # sys.exit(0)

    # assume we will only get here if we finished training
    if args.doModHard:
        #       get_mod_result(result_str, True)
        # for hard
        # result_str = 'hard'

        # get_mod_result(result_str, True)
        # f3 = 'c_ld_0.001_s_ld_100.0_f_ld_10.0_e_0.1_o_0.001_eps_200_dim_30'
        # f4 = 'c_ld_0.001_s_ld_10.0_f_ld_1.0_e_0.01_o_0.01_eps_200_dim_20'
        # f5 = 'c_ld_0.001_s_ld_1.0_f_ld_10.0_e_0.01_o_0.01_eps_100_dim_20'
        # f#iles = [f3, f4]
        # for f in files:

        # for s in flist:
        modfile = 'modularity/{}.txt'.format(result_str)
        print('modfile:', modfile)
        # if not os.path.isfile(modfile):
        #     print(" ...running: ... ")

        # else:
        #     print("...already completed...")
        result_base = 'hard'
        get_mod_result(result_str, False)
        # assume we calculate dim only when we have done train and mod_fname
    if args.doMod:
        #       get_mod_result(result_str, True)
        # for hard
        # result_str = 'hard'

        # get_mod_result(result_str, True)
        # f3 = 'c_ld_0.001_s_ld_100.0_f_ld_10.0_e_0.1_o_0.001_eps_200_dim_30'
        # f4 = 'c_ld_0.001_s_ld_10.0_f_ld_1.0_e_0.01_o_0.01_eps_200_dim_20'
        # f5 = 'c_ld_0.001_s_ld_1.0_f_ld_10.0_e_0.01_o_0.01_eps_100_dim_20'
        # f#iles = [f3, f4]
        # for f in files:

        # for s in flist:
        modfile = 'modularity/{}.txt'.format(result_str)
        print('modfile:', modfile)
        # if not os.path.isfile(modfile):
        #     print(" ...running: ... ")

        # else:
        #     print("...already completed...")
        get_mod_result(result_str, False)
    # assume we calculate dim only when we have done train and mod_fname

    if getDim:
        dim = 10
#        result_str = 'train_' + result_str
        # commented out inter_list for dim 10
        inter_list = [[0,  3, 11, 15, 16, 19, 20, 28, 29, 24],
                      [1,  3,  4, 22, 23, 24, 26, 28, 31, 30],
                      [1,  3, 10, 15, 16, 25, 28, 29, 31, 19], [1,  2,  4, 10,
                                                                11, 15, 16, 29, 31,  7],  [0,  1,  8, 17, 22, 24, 25, 26,
                                                                                           31,  7], [2,  4, 10, 15, 16, 17, 22, 24, 29, 11]]
        # inter_list = [[0,  6,  9, 24,  4],
        #               [8, 19, 30, 31, 14],
        #               [2,  9, 11, 24,  1],
        #               [2, 14, 16, 19,  0],
        #               [0,  2, 11, 18, 27],
        #               [8, 19, 27, 31, 30]]
        topx = 1
        # result_str = result_str
        get_hidden_map(result_str, inter_list, topx, dim)


if __name__ == "__main__":
    # this file is when running validation to get the best result file
    main()

# best validation result:
#  'results/c_lr_0.0001_s_lmda_0.001_f_lmda_0.9_eps_400_merge_False.txt'

# parse_results()


# get top1
# k = 1
# fname_vec = parse_results()


# for i in range(k):
#     fname = fname_vec[-1]

#     result_base = os.path.basename(fname)
#     result_str = os.path.splitext(result_base)[0]

#     get_mod_result(result_str)
