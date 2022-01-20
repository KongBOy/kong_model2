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
from step09_h3_multi_unet2_obj_I_w_Mgt_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder


import Exps_8_multi_unet.I_w_Mgt_to_Wx_Wy_Wz_try_mul_M_focus .a_normal .step10_a as I_to_Wxyz_exp_builder
import Exps_8_multi_unet.I_w_Mgt_to_Wx_Wy_Wz_try_mul_M_focus .b_limit  .step10_a as I_to_Wxyz_limit_exp_builder
import Exps_8_multi_unet.W_w_Mgt_to_Cx_Cy_try_mul_M_focus    .a_normal .step10_a as W_to_Cxy_exp_builder
import Exps_8_multi_unet.W_w_Mgt_to_Cx_Cy_try_mul_M_focus    .b_limit  .step10_a as W_to_Cxy_limit_exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

use_db_obj = type8_blender_dis_wc_flow_try_mul_M
use_loss_obj = [G_mae_s001_loss_info_builder.set_loss_target("UNet_z").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_y").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_x").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cx").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cy").copy()]  ### z, y, x 順序是看 step07_b_0b_Multi_UNet 來對應的喔
#############################################################
L2_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch128, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch128).set_result_name(result_name="I_to_Wxyz_L2_ch128_and_I_to_Cxy_L2_ch128")
L2_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch064, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch064).set_result_name(result_name="I_to_Wxyz_L2_ch064_and_I_to_Cxy_L2_ch064")
L2_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch032, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch032).set_result_name(result_name="I_to_Wxyz_L2_ch032_and_I_to_Cxy_L2_ch032")
L2_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch016).set_result_name(result_name="I_to_Wxyz_L2_ch016_and_I_to_Cxy_L2_ch016")
L2_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch008).set_result_name(result_name="I_to_Wxyz_L2_ch008_and_I_to_Cxy_L2_ch008")
L2_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch004).set_result_name(result_name="I_to_Wxyz_L2_ch004_and_I_to_Cxy_L2_ch004")
L2_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch002).set_result_name(result_name="I_to_Wxyz_L2_ch002_and_I_to_Cxy_L2_ch002")
L2_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L2_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L2_ch001).set_result_name(result_name="I_to_Wxyz_L2_ch001_and_I_to_Cxy_L2_ch001")

L3_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch128, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch128).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch128_block1_sig_out_1_&&_I_to_Wy_L3_ch128_block1_sig_out_1_&&_I_to_Wz_L3_ch128_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch128_block1_sig_out_1_&&_I_to_Cy_L3_ch128_block1_sig_out_1-20220119_093005")
L3_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch064, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch064).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch064_block1_sig_out_1_&&_I_to_Wy_L3_ch064_block1_sig_out_1_&&_I_to_Wz_L3_ch064_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch064_block1_sig_out_1_&&_I_to_Cy_L3_ch064_block1_sig_out_1-20220118_221137")
L3_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch032, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch032).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch032_block1_sig_out_1_&&_I_to_Wy_L3_ch032_block1_sig_out_1_&&_I_to_Wz_L3_ch032_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch032_block1_sig_out_1_&&_I_to_Cy_L3_ch032_block1_sig_out_1-20220119_032551")
L3_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch016).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch016_block1_sig_out_1_&&_I_to_Wy_L3_ch016_block1_sig_out_1_&&_I_to_Wz_L3_ch016_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch016_block1_sig_out_1_&&_I_to_Cy_L3_ch016_block1_sig_out_1-20220119_054816")
L3_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch008).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch008_block1_sig_out_1_&&_I_to_Wy_L3_ch008_block1_sig_out_1_&&_I_to_Wz_L3_ch008_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch008_block1_sig_out_1_&&_I_to_Cy_L3_ch008_block1_sig_out_1-20220119_070540")
L3_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch004).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch004_block1_sig_out_1_&&_I_to_Wy_L3_ch004_block1_sig_out_1_&&_I_to_Wz_L3_ch004_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch004_block1_sig_out_1_&&_I_to_Cy_L3_ch004_block1_sig_out_1-20220119_075734")
L3_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch002).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch002_block1_sig_out_1_&&_I_to_Wy_L3_ch002_block1_sig_out_1_&&_I_to_Wz_L3_ch002_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch002_block1_sig_out_1_&&_I_to_Cy_L3_ch002_block1_sig_out_1-20220119_084430")
L3_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L3_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L3_ch001).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L3_ch001_block1_sig_out_1_&&_I_to_Wy_L3_ch001_block1_sig_out_1_&&_I_to_Wz_L3_ch001_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L3_ch001_block1_sig_out_1_&&_I_to_Cy_L3_ch001_block1_sig_out_1-20220119_233420")

L4_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch128, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch128).set_result_name(result_name="I_to_Wxyz_L4_ch128_and_I_to_Cxy_L4_ch128")  ### 超出記憶體
L4_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch064, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch064).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch064_block1_sig_out_1_&&_I_to_Wy_L4_ch064_block1_sig_out_1_&&_I_to_Wz_L4_ch064_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch064_block1_sig_out_1_&&_I_to_Cy_L4_ch064_block1_sig_out_1-20220118_221255")
L4_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch032, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch032).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch032_block1_sig_out_1_&&_I_to_Wy_L4_ch032_block1_sig_out_1_&&_I_to_Wz_L4_ch032_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch032_block1_sig_out_1_&&_I_to_Cy_L4_ch032_block1_sig_out_1-20220119_032620")
L4_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch016).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch016_block1_sig_out_1_&&_I_to_Wy_L4_ch016_block1_sig_out_1_&&_I_to_Wz_L4_ch016_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch016_block1_sig_out_1_&&_I_to_Cy_L4_ch016_block1_sig_out_1-20220119_054304")
L4_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch008).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch008_block1_sig_out_1_&&_I_to_Wy_L4_ch008_block1_sig_out_1_&&_I_to_Wz_L4_ch008_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch008_block1_sig_out_1_&&_I_to_Cy_L4_ch008_block1_sig_out_1-20220119_065700")
L4_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch004).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch004_block1_sig_out_1_&&_I_to_Wy_L4_ch004_block1_sig_out_1_&&_I_to_Wz_L4_ch004_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch004_block1_sig_out_1_&&_I_to_Cy_L4_ch004_block1_sig_out_1-20220119_074732")
L4_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch002).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch002_block1_sig_out_1_&&_I_to_Wy_L4_ch002_block1_sig_out_1_&&_I_to_Wz_L4_ch002_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch002_block1_sig_out_1_&&_I_to_Cy_L4_ch002_block1_sig_out_1-20220119_083319")
L4_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L4_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L4_ch001).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L4_ch001_block1_sig_out_1_&&_I_to_Wy_L4_ch001_block1_sig_out_1_&&_I_to_Wz_L4_ch001_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L4_ch001_block1_sig_out_1_&&_I_to_Cy_L4_ch001_block1_sig_out_1-20220119_091851")

L5_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch128, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch128).set_result_name(result_name="I_to_Wxyz_L5_ch128_and_I_to_Cxy_L5_ch128")  ### I_to_Wxyz本身就已經超出記憶體
L5_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch064, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch064).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch064_block1_sig_out_1_&&_I_to_Wy_L5_ch064_block1_sig_out_1_&&_I_to_Wz_L5_ch064_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch064_block1_sig_out_1_&&_I_to_Cy_L5_ch064_block1_sig_out_1-20220118_223711")
L5_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch032, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch032).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch032_block1_sig_out_1_&&_I_to_Wy_L5_ch032_block1_sig_out_1_&&_I_to_Wz_L5_ch032_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1-20220119_063115")
L5_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch016).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch016_block1_sig_out_1_&&_I_to_Wy_L5_ch016_block1_sig_out_1_&&_I_to_Wz_L5_ch016_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch016_block1_sig_out_1_&&_I_to_Cy_L5_ch016_block1_sig_out_1-20220119_092535")
L5_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch008).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch008_block1_sig_out_1_&&_I_to_Wy_L5_ch008_block1_sig_out_1_&&_I_to_Wz_L5_ch008_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch008_block1_sig_out_1_&&_I_to_Cy_L5_ch008_block1_sig_out_1-20220119_105142")
L5_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch004).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch004_block1_sig_out_1_&&_I_to_Wy_L5_ch004_block1_sig_out_1_&&_I_to_Wz_L5_ch004_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch004_block1_sig_out_1_&&_I_to_Cy_L5_ch004_block1_sig_out_1-20220119_114858")
L5_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch002).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch002_block1_sig_out_1_&&_I_to_Wy_L5_ch002_block1_sig_out_1_&&_I_to_Wz_L5_ch002_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch002_block1_sig_out_1_&&_I_to_Cy_L5_ch002_block1_sig_out_1-20220119_123724")
L5_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L5_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L5_ch001).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L5_ch001_block1_sig_out_1_&&_I_to_Wy_L5_ch001_block1_sig_out_1_&&_I_to_Wz_L5_ch001_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L5_ch001_block1_sig_out_1_&&_I_to_Cy_L5_ch001_block1_sig_out_1-20220119_140637")

L6_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch064, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch064).set_result_name(result_name="I_to_Wxyz_L6_ch064_and_I_to_Cxy_L6_ch064")  ### I_to_Wxyz本身就已經超出記憶體
L6_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch032, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch032).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L6_ch032_block1_sig_out_1_&&_I_to_Wy_L6_ch032_block1_sig_out_1_&&_I_to_Wz_L6_ch032_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L6_ch032_block1_sig_out_1_&&_I_to_Cy_L6_ch032_block1_sig_out_1-20220118_221639")
L6_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch016).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L6_ch016_block1_sig_out_1_&&_I_to_Wy_L6_ch016_block1_sig_out_1_&&_I_to_Wz_L6_ch016_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L6_ch016_block1_sig_out_1_&&_I_to_Cy_L6_ch016_block1_sig_out_1-20220119_031940")
L6_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch008).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L6_ch008_block1_sig_out_1_&&_I_to_Wy_L6_ch008_block1_sig_out_1_&&_I_to_Wz_L6_ch008_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L6_ch008_block1_sig_out_1_&&_I_to_Cy_L6_ch008_block1_sig_out_1-20220119_051758")
L6_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch004).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L6_ch004_block1_sig_out_1_&&_I_to_Wy_L6_ch004_block1_sig_out_1_&&_I_to_Wz_L6_ch004_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L6_ch004_block1_sig_out_1_&&_I_to_Cy_L6_ch004_block1_sig_out_1-20220119_062739")
L6_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch002).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L6_ch002_block1_sig_out_1_&&_I_to_Wy_L6_ch002_block1_sig_out_1_&&_I_to_Wz_L6_ch002_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L6_ch002_block1_sig_out_1_&&_I_to_Cy_L6_ch002_block1_sig_out_1-20220119_071828")
L6_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L6_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L6_ch001).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L6_ch001_block1_sig_out_1_&&_I_to_Wy_L6_ch001_block1_sig_out_1_&&_I_to_Wz_L6_ch001_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L6_ch001_block1_sig_out_1_&&_I_to_Cy_L6_ch001_block1_sig_out_1-20220119_080707")

L7_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L7_ch032, W_to_Cx_Cy=W_to_Cxy_exp_builder.L7_ch032).set_result_name(result_name="I_to_Wxyz_L7_ch032_and_I_to_Cxy_L7_ch032")  ### 組合起來超出記憶體
L7_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L7_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L7_ch016).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L7_ch016_block1_sig_out_1_&&_I_to_Wy_L7_ch016_block1_sig_out_1_&&_I_to_Wz_L7_ch016_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L7_ch016_block1_sig_out_1_&&_I_to_Cy_L7_ch016_block1_sig_out_1-20220118_221647")
L7_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L7_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L7_ch008).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L7_ch008_block1_sig_out_1_&&_I_to_Wy_L7_ch008_block1_sig_out_1_&&_I_to_Wz_L7_ch008_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L7_ch008_block1_sig_out_1_&&_I_to_Cy_L7_ch008_block1_sig_out_1-20220119_020753")
L7_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L7_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L7_ch004).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L7_ch004_block1_sig_out_1_&&_I_to_Wy_L7_ch004_block1_sig_out_1_&&_I_to_Wz_L7_ch004_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L7_ch004_block1_sig_out_1_&&_I_to_Cy_L7_ch004_block1_sig_out_1-20220119_034012")
L7_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L7_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L7_ch002).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L7_ch002_block1_sig_out_1_&&_I_to_Wy_L7_ch002_block1_sig_out_1_&&_I_to_Wz_L7_ch002_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L7_ch002_block1_sig_out_1_&&_I_to_Cy_L7_ch002_block1_sig_out_1-20220119_043831")
L7_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L7_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L7_ch001).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L7_ch001_block1_sig_out_1_&&_I_to_Wy_L7_ch001_block1_sig_out_1_&&_I_to_Wz_L7_ch001_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L7_ch001_block1_sig_out_1_&&_I_to_Cy_L7_ch001_block1_sig_out_1-20220119_053357")

L8_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L8_ch016, W_to_Cx_Cy=W_to_Cxy_exp_builder.L8_ch016).set_result_name(result_name="I_to_Wxyz_L8_ch016_and_I_to_Cxy_L8_ch016")
L8_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L8_ch008, W_to_Cx_Cy=W_to_Cxy_exp_builder.L8_ch008).set_result_name(result_name="I_to_Wxyz_L8_ch008_and_I_to_Cxy_L8_ch008")
L8_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L8_ch004, W_to_Cx_Cy=W_to_Cxy_exp_builder.L8_ch004).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L8_ch004_block1_sig_out_1_&&_I_to_Wy_L8_ch004_block1_sig_out_1_&&_I_to_Wz_L8_ch004_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L8_ch004_block1_sig_out_1_&&_I_to_Cy_L8_ch004_block1_sig_out_1-20220120_114013")
L8_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L8_ch002, W_to_Cx_Cy=W_to_Cxy_exp_builder.L8_ch002).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L8_ch002_block1_sig_out_1_&&_I_to_Wy_L8_ch002_block1_sig_out_1_&&_I_to_Wz_L8_ch002_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L8_ch002_block1_sig_out_1_&&_I_to_Cy_L8_ch002_block1_sig_out_1-20220120_113946")
L8_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.L8_ch001, W_to_Cx_Cy=W_to_Cxy_exp_builder.L8_ch001).set_result_name(result_name="type8_blender-I_to_Wx_Wy_Wz_I_to_Wx_L8_ch001_block1_sig_out_1_&&_I_to_Wy_L8_ch001_block1_sig_out_1_&&_I_to_Wz_L8_ch001_block1_sig_out_1_&&_W_to_Cx_Cy_I_to_Cx_L8_ch001_block1_sig_out_1_&&_I_to_Cy_L8_ch001_block1_sig_out_1-20220120_124021")


if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L2_ch016.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
