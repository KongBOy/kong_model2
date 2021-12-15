#############################################################################################################################################################################################################
import os
code_exe_path = os.path.realpath(__file__)     ### 目前執行 step11.py 的 path
code_exe_dir = os.path.dirname(code_exe_path)  ### 目前執行 step11.py 的 dir
import sys                                     ### 把 step11 加入 的 dir sys.path， 在外部 import 這裡的 step11 才 import 的到 step10_a.py
sys.path.append(code_exe_dir)
#############################################################################################################################################################################################################
import step10_a_c_no_Bias as bce_block1

#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################
### no_Bias
bce_L4_block1_ch032_no_Bias = [
    bce_block1.L4_ch032_sig_ep060_bce_s001_no_Bias.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s020_no_Bias.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s040_no_Bias.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s060_no_Bias.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s080_no_Bias.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s100_no_Bias.build(),
]
bce_L4_block1_ch032_no_Bias = [ exp.result_obj for exp in bce_L4_block1_ch032_no_Bias]
####################
### no_Bias
bce_L4_block1_ch016_no_Bias = [
    bce_block1.L4_ch016_sig_ep060_bce_s001_no_Bias.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s020_no_Bias.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s040_no_Bias.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s060_no_Bias.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s080_no_Bias.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s100_no_Bias.build(),
]
bce_L4_block1_ch016_no_Bias = [ exp.result_obj for exp in bce_L4_block1_ch016_no_Bias]
### no_Bias
bce_L4_block1_ch008_no_Bias = [
    bce_block1.L4_ch008_sig_ep060_bce_s001_no_Bias.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s020_no_Bias.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s040_no_Bias.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s060_no_Bias.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s080_no_Bias.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s100_no_Bias.build(),
]
bce_L4_block1_ch008_no_Bias = [ exp.result_obj for exp in bce_L4_block1_ch008_no_Bias]
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################
### L2345_no_Bias
L2345_block1_bce_no_Bias = [
    [bce_block1.L2_ch128_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L2_ch064_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L2_ch032_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L2_ch016_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L2_ch008_sig_ep060_bce_s001_no_Bias.build().result_obj],
    [bce_block1.L3_ch128_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L3_ch064_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L3_ch032_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L3_ch016_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L3_ch008_sig_ep060_bce_s001_no_Bias.build().result_obj],
    [bce_block1.L4_ch064_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L4_ch032_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L4_ch016_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L4_ch008_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L4_ch004_sig_ep060_bce_s001_no_Bias.build().result_obj],
    [bce_block1.L5_ch032_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L5_ch016_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L5_ch008_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L5_ch004_sig_ep060_bce_s001_no_Bias.build().result_obj,
     bce_block1.L5_ch002_sig_ep060_bce_s001_no_Bias.build().result_obj],
]
