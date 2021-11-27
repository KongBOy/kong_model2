#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
exp_dir = code_exe_path_element[5][7:] + "/" + code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
# print("    exp_dir:", exp_dir)  ### 舉例：exp_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_e2_mask_unet2_obj import *
from step09_b_loss_info_obj import *
from step10_b_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
############################  have_bg  #################################
### ch064
mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s120_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_1", describe_end="mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s140_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s140_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_2", describe_end="mask_h_bg_ch064_sig_sobel_k5_s140_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s160_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s160_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_3", describe_end="mask_h_bg_ch064_sig_sobel_k5_s160_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s180_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s180_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_4", describe_end="mask_h_bg_ch064_sig_sobel_k5_s180_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s200_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s200_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_5", describe_end="mask_h_bg_ch064_sig_sobel_k5_s200_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s220_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s220_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_6", describe_end="mask_h_bg_ch064_sig_sobel_k5_s220_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s240_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s240_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_7", describe_end="mask_h_bg_ch064_sig_sobel_k5_s240_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s260_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s260_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_8", describe_end="mask_h_bg_ch064_sig_sobel_k5_s260_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
