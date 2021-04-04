- Perform grid search with: ``python runprofile.py``
- The optimal config file for our data is stored in: ``config/train_c_ld_1.0_s_ld_0.1_f_ld_10.0_o_1.0_eps_200_dim_10_l2_True_cos_True_l2lr_1e-05_coslr_0.1``
- Running DHC with a single configuration can be performed by:
``python disentangled_hyperspherical_cluster.py --filename={config_file} --train=true``
