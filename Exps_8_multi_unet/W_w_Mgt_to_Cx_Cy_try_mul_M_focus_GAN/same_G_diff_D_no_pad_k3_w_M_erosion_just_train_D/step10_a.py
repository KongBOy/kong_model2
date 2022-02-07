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
from step09_i3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus_GAN_G_L5_ch032_D_all_no_pad_k3_w_M_erosion import *
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

use_db_obj = type8_blender_wc_flow_try_mul_M
use_loss_obj = [G_mae_s001_loss_info_builder.set_loss_target("UNet_Cx").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cy").copy(), GAN_s001_loss_info_builder.set_loss_target("D_Cxy").copy()]  ### z, y, x 順序是看 step07_b_0b_Multi_UNet 來對應的喔
#############################################################
L1_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch128-20220206_083531")
L1_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch064-20220206_042906")
L1_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch032-20220206_022553")
L1_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch016-20220206_002353")
L1_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch008-20220205_222154")
L1_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch004-20220205_202021")
L1_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch002-20220205_181854")
L1_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L1_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L1_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L1_ch001-20220206_063403")

L2_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch128-20220206_053639")
L2_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch064-20220206_021936")
L2_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch032-20220206_004151")
L2_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch016-20220205_230536")
L2_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch008-20220205_213016")
L2_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch004-20220205_195459")
L2_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch002-20220205_181934")
L2_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L2_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L2_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L2_ch001-20220206_040119")

L3_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch128-20220206_054538")
L3_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch064-20220206_022432")
L3_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch032-20220206_004524")
L3_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch016-20220205_230819")
L3_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch008-20220205_213208")
L3_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch004-20220205_195605")
L3_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch002-20220205_182010")
L3_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L3_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L3_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L3_ch001-20220206_041009")

L4_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch128-20220206_062215")
L4_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch064-20220206_024841")
L4_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch032-20220206_010408")
L4_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch016-20220205_232242")
L4_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch008-20220205_214145")
L4_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch004-20220205_200118")
L4_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch002-20220205_182046")
L4_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L4_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L4_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L4_ch001-20220206_044202")

L5_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch128-20220206_054933")
L5_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch064-20220206_022353")
L5_ch032 = Exp_builder().set_basic("test_train", use_db_obj, disc_L5_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch032-20220206_004348")
L5_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch016-20220205_230719")
L5_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch008-20220205_213139")
L5_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch004-20220205_195609")
L5_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch002-20220205_182110")
L5_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L5_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L5_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L5_ch001-20220206_041435")

L6_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch128-20220206_123337")
L6_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch064-20220206_192404")
L6_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch032-20220206_190310")
L6_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch016-20220206_165649")
L6_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch008-20220206_145138")
L6_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch004-20220206_124708")
L6_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch002-20220206_104348")
L6_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L6_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L6_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L6_ch001-20220206_174238")

L7_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")  ### 訓練超出記憶體
L7_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch064-20220206_101051")
L7_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch032-20220206_140809")
L7_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch016-20220206_122326")
L7_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch008-20220206_104407")
L7_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch004-20220206_090610")
L7_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch002-20220206_072811")
L7_ch001 = Exp_builder().set_basic("train", use_db_obj, disc_L7_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L7_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L7_ch001-20220206_083451")

L8_ch128 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")  ### 訓練超出記憶體
L8_ch064 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")  ### 訓練超出記憶體
L8_ch032 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L8_ch032-20220206_145657")
L8_ch016 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L8_ch016-20220206_124925")
L8_ch008 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L8_ch008-20220206_110501")
L8_ch004 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L8_ch004-20220206_092546")
L8_ch002 = Exp_builder().set_basic("train", use_db_obj, disc_L8_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L8_ch002-20220206_074652")
L8_ch001 = Exp_builder().set_basic("train_reload", use_db_obj, disc_L8_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=disc_L8_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1_&&_Disc_L8_ch001-20220206_192900")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L5_ch032.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
