import subprocess
import configparser
import glob
import os
import numpy as np


# generate config files
do_merge = False
config = configparser.ConfigParser()
config['Path'] = {}
path_params = config['Path']
path_params['proj_dir'] = '~/Dropbox/PHD/sepsis/code'
path_params['result_dir'] = '~/Dropbox/PHD/sepsis/results'
config['Autoencoder'] = {}
auto_params = config['Autoencoder']
auto_params['minibatch_size'] = '128'
auto_params['subsample_stride'] = '1'
auto_params['o_learning_rate'] = '1e-3'
auto_params['c_learning_rate'] = '1e-4'
auto_params['num_epochs'] = '100'
auto_params['burn_in'] = '10'
auto_params['print_interval'] = '1'
auto_params['image_interval'] = '16'
auto_params['desired_centroids'] = '4'
auto_params['g_input_size'] = '1024'
auto_params['hidden_size'] = '32'
auto_params['side_channel_size'] = '0'
# side channel is changed from 1 to 0
auto_params['c_hidden_size'] = '128'
auto_params['noise_sd'] = '1e-2'
auto_params['explode_factor'] = '10000'
auto_params['pr_g_update'] = '1'
auto_params['g_lambda'] = '1e-4'
auto_params['g_o_ratio'] = '1e-1'
auto_params['pr_c_update'] = '1'
auto_params['c_only'] = 'False'
auto_params['c_lambda'] = '1e-0'
auto_params['c_l2_lambda'] = '0'
auto_params['l2_lambda'] = '1e-4'
auto_params['cos_lambda'] = '1e-4'
auto_params['o_lambda'] = '1e-1'
auto_params['e_lambda'] = '1e-0'
auto_params['s_lambda'] = '1e-4'
auto_params['f_lambda'] = '0.5'
auto_params['bool_f_statistics'] = 'True'
auto_params['top_d'] = '5'
auto_params['c_output_size'] = '4'
auto_params['do_merge'] = str(do_merge)
# auto_params['']
#do_l2 = True
#do_cos = True
auto_params['do_l2'] = str(False)
auto_params['do_cos'] = str(False)


c_learning_vec = [1e-3]
c_lambda_vec = [1e-1, 1e-2, 1e-3, 1e-4, 1e0, 1e1]
s_lambda_vec = [1e-1, 1e-2, 1e-3, 1e-4, 1e0, 1e1]
#s_lambda_vec = [0]
#o_lambda_vec = [1e0, 1e-1]
o_lambda_vec = [1e-1, 1e-2, 1e-3, 1e-4, 1e0, 1e1]
#e_lambda_vec = [1e0, 1e-1, 1e-2]
# 1e-3,1e1 change the exponents
f_lambda_vec = [1e-1, 1e-2, 1e-3, 1e-4, 1e0, 1e1]
num_epochs_vec = [150, 200]
d_dims = [10]  # 20
l2_lambda_vec = [1e-4, 1e-5]
l2_vec = [True]

cos_lambda_vec = [1e-2, 1e-3, 1e-4, 1e-1, 1e0, 1e1]
cos_vec = [True]
# c_learning_vec = [1e-1]
# s_lambda_vec = [1e-1]
# f_lambda_vec = [1e-1]  # 1e-3,1e1 change the exponents
# num_epochs_vec = [10]x
# d_dims = [10]


# d_dim=10 was first trained, but the cavaet is that the name does not include
# '_dim_' inside the config file. the files locate inside all corresponding
# results folders

# Let doL2 be equivalent to adding a batch2norm between input and hidden
#l2_vec = ["", True]
for c_ld in c_lambda_vec:
    for s_lmda in s_lambda_vec:
        for f_lmda in f_lambda_vec:
            for eps in num_epochs_vec:
                for d in d_dims:
                    for o_lr in o_lambda_vec:
                        # for e_lr in e_lambda_vec:
                        for l2 in l2_vec:
                            for l2_lr in l2_lambda_vec:
                                for cos in cos_vec:
                                    for cos_lr in cos_lambda_vec:
                                        tmp_name = 'config/c_ld_' + str(c_ld) + '_s_ld_' + str(s_lmda) + \
                                            '_f_ld_' + str(f_lmda) +\
                                            '_o_' + str(o_lr) + '_eps_' + \
                                            str(eps) + '_dim_' + \
                                            str(d) + '_l2_' + str(l2) +\
                                            '_cos_' + str(cos)
                                        auto_params['c_lambda'] = str(c_ld)
                                        auto_params['s_lambda'] = str(
                                            s_lmda)
                                        auto_params['f_lambda'] = str(
                                            f_lmda)
                                        auto_params['num_epochs'] = str(
                                            eps)
                                        auto_params['top_d'] = str(d)
                                        auto_params['o_lambda'] = str(o_lr)
                                        #auto_params['e_lambda'] = str(e_lr)
                                        if l2:
                                            auto_params['do_l2'] = str(l2)
                                            auto_params['l2_lambda'] = str(
                                                l2_lr)
                                            tmp_name += '_l2lr_' + \
                                                str(l2_lr)
                                        if cos:
                                            auto_params['do_cos'] = str(
                                                cos)
                                            auto_params['cos_lambda'] = str(
                                                cos_lr)
                                            tmp_name += '_coslr_' + \
                                                str(cos_lr)
                                            tmp_name += '_top_' + \
                                                auto_params['top_d']
                                        tmp_name += '.ini'
                                        with open(tmp_name, 'w') as configfile:
                                            config.write(configfile)

# read config files
allconfigs = glob.glob('config/*.ini')
print(allconfigs)


# run program with different config files]
for f in allconfigs:
    cfilename = "--filename={}".format(f)
    ctrain = "--train=true"
    base = os.path.basename(f)
    name_str = os.path.splitext(base)[0]
    # result_str = 'c_lr_' + str(c_learning_rate) + '_s_lmda' +\
    #     str(s_lambda) + '_f_lmda_' + str(f_lambda) + '_epochs_' + str(num_epochs) + \
    #     '_merge_' + str(do_merge)
    print("config file name: {}".format(f))
    if not os.path.isfile('results/{}.txt'.format(name_str)):
        print(" ...running: ... ")
        # main.run(args)
        out = subprocess.run(
            ["python", "disentangled_hypershperical_cluster.py", cfilename, ctrain])
    else:
        print("...already completed...")

# cfilename = "--filename=test.ini"
# out = subprocess.run(
#     ["python", "disentangled_hypershperical_cluster.py", cfilename])


#configfile = []
