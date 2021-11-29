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
### 1a. ch
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s001_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s001_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_1_1", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_1_1-20211102_110454-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s001_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s001_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s001_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_1_2", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_1_2-20211102_113858-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s020_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s001_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s001_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_1_3", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_1_3-20211102_121308-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s040_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s001_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s001_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_1_4", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_1_4-20211102_124704-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s060_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s001_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s001_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_1_5", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_1_5-20211102_132110-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s080_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s001_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s001_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_1_6", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s100_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_1_6-20211102_135506-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s001_sobel_k5_s100_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s020_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s020_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_2_1", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_2_1-20211102_142908-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s001_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s020_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s020_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_2_2", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_2_2-20211102_150317-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s020_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s020_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s020_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_2_3", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_2_3-20211102_153734-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s040_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s020_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s020_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_2_4", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_2_4-20211102_161145-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s060_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s020_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s020_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_2_5", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_2_5-20211102_164555-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s080_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s020_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s020_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_2_6", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s100_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_2_6-20211102_172009-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s020_sobel_k5_s100_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s040_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s040_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_3_1", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_3_1-20211102_175417-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s001_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s040_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s040_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_3_2", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_3_2-20211102_182828-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s020_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s040_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s040_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_3_3", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_3_3-20211102_190243-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s040_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s040_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s040_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_3_4", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_3_4-20211102_193658-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s060_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s040_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s040_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_3_5", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_3_5-20211102_201105-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s080_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s040_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s040_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_3_6", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s100_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_3_6-20211102_204518-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s040_sobel_k5_s100_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s060_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s060_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_4_1", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_4_1-20211102_211929-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s001_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s060_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s060_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_4_2", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_4_2-20211102_215345-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s020_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s060_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s060_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_4_3", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_4_3-20211102_222801-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s040_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s060_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s060_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_4_4", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_4_4-20211102_230216-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s060_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s060_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s060_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_4_5", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_4_5-20211102_233630-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s080_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s060_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s060_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_4_6", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s100_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_4_6-20211103_001041-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s060_sobel_k5_s100_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s080_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s080_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_5_1", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_5_1-20211103_004451-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s001_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s080_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s080_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_5_2", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_5_2-20211103_011911-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s020_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s080_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s080_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_5_3", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_5_3-20211103_015326-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s040_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s080_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s080_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_5_4", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_5_4-20211103_022737-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s060_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s080_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s080_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_5_5", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_5_5-20211103_030153-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s080_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s080_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s080_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_5_6", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s100_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_5_6-20211103_033611-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s080_sobel_k5_s100_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s100_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_6_1", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_6_1-20211103_041022-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s001_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s100_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_6_2", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_6_2-20211103_044436-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s020_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s100_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_6_3", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_6_3-20211103_051850-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s040_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s100_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_6_4", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_6_4-20211103_055309-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s060_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s100_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_6_5", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_6_5-20211103_062729-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s080_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_bce_s100_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="8b_6_6", describe_end="mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s100_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-8b_6_6-20211103_070148-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s100_6l_ep060")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch032_sig_L6_ep060_tv_s100_bce_s100_sobel_k5_s100.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
