#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
# print("    kong_to_py_layer:", kong_to_py_layer)
if  (kong_to_py_layer == 0): template_dir = ""
elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][0:]  ### [7:] 是為了去掉 step1x_， 後來覺得好像改有意義的名字不去掉也行所以 改 0
elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子， 後來覺得 自動排的順序也可以接受， 所以 改0
elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])
# print("    template_dir:", template_dir)  ### 舉例： template_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
exp_dir = template_dir
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_e2_mask_unet2_obj import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder
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
mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s001_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s001_sobel_k5_s001", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s001_sobel_k5_s001-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_152909")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s001_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s001_sobel_k5_s020", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s001_sobel_k5_s020-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_160101")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s001_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s001_sobel_k5_s040", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s001_sobel_k5_s040-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_163245")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s001_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s001_sobel_k5_s060", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s001_sobel_k5_s060-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_170429")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s001_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s001_sobel_k5_s080", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s001_sobel_k5_s080-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_173619")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s001_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s001_sobel_k5_s100", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s001_sobel_k5_s100-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_180806")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s020_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s020_sobel_k5_s001", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s020_sobel_k5_s001-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_183949")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s020_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s020_sobel_k5_s020", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s020_sobel_k5_s020-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_191136")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s020_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s020_sobel_k5_s040", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s020_sobel_k5_s040-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_194320")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s020_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s020_sobel_k5_s060", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s020_sobel_k5_s060-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_201505")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s020_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s020_sobel_k5_s080", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s020_sobel_k5_s080-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_204653")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s020_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s020_sobel_k5_s100", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s020_sobel_k5_s100-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_211841")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s040_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s040_sobel_k5_s001", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s040_sobel_k5_s001-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_215026")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s040_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s040_sobel_k5_s020", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s040_sobel_k5_s020-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_222212")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s040_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s040_sobel_k5_s040", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s040_sobel_k5_s040-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_225403")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s040_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s040_sobel_k5_s060", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s040_sobel_k5_s060-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_232548")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s040_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s040_sobel_k5_s080", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s040_sobel_k5_s080-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211104_235732")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s040_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s040_sobel_k5_s100", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s040_sobel_k5_s100-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_000256")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s060_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s060_sobel_k5_s001", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s060_sobel_k5_s001-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_003443")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s060_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s060_sobel_k5_s020", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s060_sobel_k5_s020-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_010627")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s060_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s060_sobel_k5_s040", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s060_sobel_k5_s040-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_013814")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s060_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s060_sobel_k5_s060", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s060_sobel_k5_s060-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_020958")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s060_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s060_sobel_k5_s080", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s060_sobel_k5_s080-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_024153")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s060_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s060_sobel_k5_s100", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s060_sobel_k5_s100-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_031341")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s080_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s080_sobel_k5_s001", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s080_sobel_k5_s001-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_034526")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s080_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s080_sobel_k5_s020", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s080_sobel_k5_s020-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_041715")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s080_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s080_sobel_k5_s040", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s080_sobel_k5_s040-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_044900")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s080_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s080_sobel_k5_s060", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s080_sobel_k5_s060-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_052044")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s080_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s080_sobel_k5_s080", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s080_sobel_k5_s080-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_055225")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s080_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s080_sobel_k5_s100", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s080_sobel_k5_s100-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_062405")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_sobel_k5_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s100_sobel_k5_s001", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s100_sobel_k5_s001-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_065546")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_sobel_k5_s020_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s100_sobel_k5_s020", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s100_sobel_k5_s020-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_072733")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_sobel_k5_s040_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s100_sobel_k5_s040", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s100_sobel_k5_s040-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_075921")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_sobel_k5_s060_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s100_sobel_k5_s060", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s100_sobel_k5_s060-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_083106")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_sobel_k5_s080_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s100_sobel_k5_s080", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s100_sobel_k5_s080-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_090253")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s100 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s100_sobel_k5_s100_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="7b_tv_s100_sobel_k5_s100", describe_end="mask_h_bg_ch032_sig_L6_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-7b_tv_s100_sobel_k5_s100-flow_unet-mask_h_bg_ch032_sig_L6_ep060-20211105_093441")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s100
if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s001.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])