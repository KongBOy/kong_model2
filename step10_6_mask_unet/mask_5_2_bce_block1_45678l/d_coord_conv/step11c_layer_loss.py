#############################################################################################################################################################################################################
import os
code_exe_path = os.path.realpath(__file__)     ### 目前執行 step11.py 的 path
code_exe_dir = os.path.dirname(code_exe_path)  ### 目前執行 step11.py 的 dir
import sys                                     ### 把 step11 加入 的 dir sys.path， 在外部 import 這裡的 step11 才 import 的到 step10_a.py
sys.path.append(code_exe_dir)
#############################################################################################################################################################################################################
import step10_a_d_coord_conv as bce_block1

### L2345_coord_conv
L23456_block1_bce_coord_conv = [
    [bce_block1.L2_ch128_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L2_ch064_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L2_ch032_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L2_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L2_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj],
    [bce_block1.L3_ch128_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L3_ch064_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L3_ch032_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L3_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L3_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj],
    [bce_block1.L4_ch064_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L4_ch032_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L4_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L4_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L4_ch004_sig_ep060_bce_s001_coord_conv.build().result_obj],
    [bce_block1.L5_ch032_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L5_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L5_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L5_ch004_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L5_ch002_sig_ep060_bce_s001_coord_conv.build().result_obj],
    [bce_block1.L6_ch064_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L6_ch032_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L6_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L6_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L6_ch004_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L6_ch002_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L6_ch001_sig_ep060_bce_s001_coord_conv.build().result_obj]]

L2345678_block1_bce_coord_conv = [] + L23456_block1_bce_coord_conv
L2345678_block1_bce_coord_conv += [
    [bce_block1.L7_ch032_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L7_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L7_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L7_ch004_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L7_ch002_sig_ep060_bce_s001_coord_conv.build().result_obj],
    [bce_block1.L8_ch016_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L8_ch008_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L8_ch004_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L8_ch002_sig_ep060_bce_s001_coord_conv.build().result_obj,
     bce_block1.L8_ch001_sig_ep060_bce_s001_coord_conv.build().result_obj]]
