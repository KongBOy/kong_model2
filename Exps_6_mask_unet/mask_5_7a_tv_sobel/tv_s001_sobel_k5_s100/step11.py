
import os
code_exe_path = os.path.realpath(__file__)     ### 目前執行 step11.py 的 path
code_exe_dir = os.path.dirname(code_exe_path)  ### 目前執行 step11.py 的 dir
import sys                                     ### 把 step11 加入 的 dir sys.path， 在外部 import 這裡的 step11 才 import 的到 step10_a.py
sys.path.append(code_exe_dir)
#############################################################################################################################################################################################################
import step10_a_mask_5_7_tv_s001_sobel_k5_s100 as tv_s001_sobel_k5_s100

####################################################################################################################################
### 2-6l_s01-1_ch
mask_tv_s001_sobel_k5_s100_ch = [
    tv_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_tv_s001_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_tv_s001_sobel_k5_s100_ch]
############################################
### 2-6l_s01-2_ep
mask_tv_s001_sobel_k5_s100_ep = [
    tv_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep200.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep200.build()]
mask_tv_s001_sobel_k5_s100_ep = [ exp.result_obj for exp in mask_tv_s001_sobel_k5_s100_ep]
############################################
### 2-6l_s01-4_no-concat_and_add
mask_tv_s001_sobel_k5_s100_noC_and_add =  [
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
    tv_s001_sobel_k5_s100.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_tv_s001_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_tv_s001_sobel_k5_s100_noC_and_add]
