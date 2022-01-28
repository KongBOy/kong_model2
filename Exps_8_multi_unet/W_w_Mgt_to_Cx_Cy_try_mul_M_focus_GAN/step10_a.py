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
from step09_i3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus_GAN import *
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
L2_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch128_block1_sig_out_1_&&_I_to_Cy_L2_ch128_block1_sig_out_1-20220126_002949")
L2_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch064_block1_sig_out_1_&&_I_to_Cy_L2_ch064_block1_sig_out_1-20220125_172841")
L2_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch032_block1_sig_out_1_&&_I_to_Cy_L2_ch032_block1_sig_out_1-20220125_194810")
L2_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch016_block1_sig_out_1_&&_I_to_Cy_L2_ch016_block1_sig_out_1-20220125_211128")
L2_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch008_block1_sig_out_1_&&_I_to_Cy_L2_ch008_block1_sig_out_1-20220125_221126")
L2_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch004_block1_sig_out_1_&&_I_to_Cy_L2_ch004_block1_sig_out_1-20220125_230208")
L2_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch002_block1_sig_out_1_&&_I_to_Cy_L2_ch002_block1_sig_out_1-20220125_234718")
L2_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L2_ch001_block1_sig_out_1_&&_I_to_Cy_L2_ch001_block1_sig_out_1-20220126_052827")

L3_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch128_block1_sig_out_1_&&_I_to_Cy_L3_ch128_block1_sig_out_1-20220126_140829")
L3_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch064_block1_sig_out_1_&&_I_to_Cy_L3_ch064_block1_sig_out_1-20220126_060952")
L3_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch032_block1_sig_out_1_&&_I_to_Cy_L3_ch032_block1_sig_out_1-20220126_092525")
L3_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch016_block1_sig_out_1_&&_I_to_Cy_L3_ch016_block1_sig_out_1-20220126_110754")
L3_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch008_block1_sig_out_1_&&_I_to_Cy_L3_ch008_block1_sig_out_1-20220126_113709")
L3_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch004_block1_sig_out_1_&&_I_to_Cy_L3_ch004_block1_sig_out_1-20220126_123343")
L3_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch002_block1_sig_out_1_&&_I_to_Cy_L3_ch002_block1_sig_out_1-20220126_132255")
L3_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L3_ch001_block1_sig_out_1_&&_I_to_Cy_L3_ch001_block1_sig_out_1-20220126_222956")

L4_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch128_block1_sig_out_1_&&_I_to_Cy_L4_ch128_block1_sig_out_1-20220126_011625")
L4_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch064_block1_sig_out_1_&&_I_to_Cy_L4_ch064_block1_sig_out_1-20220125_172341")
L4_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch032_block1_sig_out_1_&&_I_to_Cy_L4_ch032_block1_sig_out_1-20220125_203028")
L4_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch016_block1_sig_out_1_&&_I_to_Cy_L4_ch016_block1_sig_out_1-20220125_220254")
L4_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch008_block1_sig_out_1_&&_I_to_Cy_L4_ch008_block1_sig_out_1-20220125_230424")
L4_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch004_block1_sig_out_1_&&_I_to_Cy_L4_ch004_block1_sig_out_1-20220125_235309")
L4_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch002_block1_sig_out_1_&&_I_to_Cy_L4_ch002_block1_sig_out_1-20220126_003620")
L4_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L4_ch001_block1_sig_out_1_&&_I_to_Cy_L4_ch001_block1_sig_out_1-20220126_093753")

L5_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch128_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch128_block1_sig_out_1_&&_I_to_Cy_L5_ch128_block1_sig_out_1-20220126_033213")
L5_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch064_block1_sig_out_1_&&_I_to_Cy_L5_ch064_block1_sig_out_1-20220125_173019")
L5_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch032_block1_sig_out_1_&&_I_to_Cy_L5_ch032_block1_sig_out_1-20220125_220008")
L5_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch016_block1_sig_out_1_&&_I_to_Cy_L5_ch016_block1_sig_out_1-20220125_235631")
L5_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch008_block1_sig_out_1_&&_I_to_Cy_L5_ch008_block1_sig_out_1-20220126_010621")
L5_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch004_block1_sig_out_1_&&_I_to_Cy_L5_ch004_block1_sig_out_1-20220126_020012")
L5_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch002_block1_sig_out_1_&&_I_to_Cy_L5_ch002_block1_sig_out_1-20220126_024747")
L5_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L5_ch001_block1_sig_out_1_&&_I_to_Cy_L5_ch001_block1_sig_out_1-20220126_182956")

L6_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch064_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch064_block1_sig_out_1_&&_I_to_Cy_L6_ch064_block1_sig_out_1-20220126_001049")
L6_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch032_block1_sig_out_1_&&_I_to_Cy_L6_ch032_block1_sig_out_1-20220125_173116")
L6_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch016_block1_sig_out_1_&&_I_to_Cy_L6_ch016_block1_sig_out_1-20220125_201934")
L6_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch008_block1_sig_out_1_&&_I_to_Cy_L6_ch008_block1_sig_out_1-20220125_213925")
L6_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch004_block1_sig_out_1_&&_I_to_Cy_L6_ch004_block1_sig_out_1-20220125_223651")
L6_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch002_block1_sig_out_1_&&_I_to_Cy_L6_ch002_block1_sig_out_1-20220125_232620")
L6_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch001_block1_sig_out_1_&&_I_to_Cy_L6_ch001_block1_sig_out_1-20220126_095122")

L7_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch032_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L7_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L7_ch016_block1_sig_out_1_&&_I_to_Cy_L7_ch016_block1_sig_out_1-20220125_225125")
L7_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L7_ch008_block1_sig_out_1_&&_I_to_Cy_L7_ch008_block1_sig_out_1-20220126_011024")
L7_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L7_ch004_block1_sig_out_1_&&_I_to_Cy_L7_ch004_block1_sig_out_1-20220126_022113")
L7_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L7_ch002_block1_sig_out_1_&&_I_to_Cy_L7_ch002_block1_sig_out_1-20220126_031510")
L7_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L7_ch001_block1_sig_out_1_&&_I_to_Cy_L7_ch001_block1_sig_out_1-20220126_120545")

L8_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch016_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L8_ch016_block1_sig_out_1_&&_I_to_Cy_L8_ch016_block1_sig_out_1-20220126_173838")
L8_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch008_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L8_ch008_block1_sig_out_1_&&_I_to_Cy_L8_ch008_block1_sig_out_1-20220126_151937")
L8_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch004_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L8_ch004_block1_sig_out_1_&&_I_to_Cy_L8_ch004_block1_sig_out_1-20220126_123353")
L8_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch002_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L8_ch002_block1_sig_out_1_&&_I_to_Cy_L8_ch002_block1_sig_out_1-20220126_133933")
L8_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch001_sig.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L8_ch001_block1_sig_out_1_&&_I_to_Cy_L8_ch001_block1_sig_out_1-20220126_143159")



L5_ch128_limit = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch128_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

L6_ch128_limit = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch128_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender-I_to_Cx_L6_ch128_block1_sig_out_1_&&_I_to_Cy_L6_ch128_block1_sig_out_1_limit-20220126_103335")
L6_ch064_limit = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch064_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

L7_ch128_limit = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch128_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L7_ch064_limit = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch064_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L7_ch032_limit = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch032_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch032_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

L8_ch128_limit = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch128_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch064_limit = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch064_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch032_limit = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch032_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch032_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch016_limit = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch016_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch016_sig_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")


if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L2_ch002.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
