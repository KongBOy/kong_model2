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
elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][7:]  ### [7:] 是為了去掉 step1x_
elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
# print("    template_dir:", template_dir)  ### 舉例： template_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
exp_dir = template_dir
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_e2_mask_unet2_obj import *
from step09_b_loss_info_obj import *
from step10_b_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet2_block1/自己命的名字"，那 result_dir 就都在：
    6_mask_unet2_block1/自己命的名字/result_a
    6_mask_unet2_block1/自己命的名字/result_b
    6_mask_unet2_block1/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
############################  have_bg  #################################
### 1a. ch
mask_h_bg_ch128_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_1", describe_end="mask_h_bg_block1_ch128_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_1-flow_unet2-mask_h_bg_block1_ch128_sig_bce_ep060-20211107_012545")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_2", describe_end="mask_h_bg_block1_ch064_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_2-flow_unet2-mask_h_bg_block1_ch064_sig_bce_ep060-20211106_181134")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_3", describe_end="mask_h_bg_block1_ch032_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_3-flow_unet2-mask_h_bg_block1_ch032_sig_bce_ep060-20211106_195944")  #.change_result_name_v1_to_v2()
mask_h_bg_ch016_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_4", describe_end="mask_h_bg_block1_ch016_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_4-flow_unet2-mask_h_bg_block1_ch016_sig_bce_ep060-20211106_210109")  #.change_result_name_v1_to_v2()
mask_h_bg_ch008_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_5", describe_end="mask_h_bg_block1_ch008_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_5-flow_unet2-mask_h_bg_block1_ch008_sig_bce_ep060-20211106_213803")  #.change_result_name_v1_to_v2()
mask_h_bg_ch004_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_6", describe_end="mask_h_bg_block1_ch004_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_6-flow_unet2-mask_h_bg_block1_ch004_sig_bce_ep060-20211106_220905")  #.change_result_name_v1_to_v2()
mask_h_bg_ch002_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_7", describe_end="mask_h_bg_block1_ch002_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_7-flow_unet2-mask_h_bg_block1_ch002_sig_bce_ep060-20211106_223849")  #.change_result_name_v1_to_v2()
mask_h_bg_ch001_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1_8", describe_end="mask_h_bg_block1_ch001_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1_8-flow_unet2-mask_h_bg_block1_ch001_sig_bce_ep060-20211106_230804")  #.change_result_name_v1_to_v2()
### 1b. ch and epoch
mask_h_bg_ch128_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_1", describe_end="mask_h_bg_block1_ch128_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_1-flow_unet2-mask_h_bg_block1_ch128_sig_bce_ep200-20211107_200234")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_2", describe_end="mask_h_bg_block1_ch064_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_2-flow_unet2-mask_h_bg_block1_ch064_sig_bce_ep200-20211107_044610")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_3", describe_end="mask_h_bg_block1_ch032_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_3-flow_unet2-mask_h_bg_block1_ch032_sig_bce_ep200-20211107_103813")  #.change_result_name_v1_to_v2()
mask_h_bg_ch016_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_4", describe_end="mask_h_bg_block1_ch016_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_4-flow_unet2-mask_h_bg_block1_ch016_sig_bce_ep200-20211107_135908")  #.change_result_name_v1_to_v2()
mask_h_bg_ch008_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_5", describe_end="mask_h_bg_block1_ch008_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_5-flow_unet2-mask_h_bg_block1_ch008_sig_bce_ep200-20211107_160017")  #.change_result_name_v1_to_v2()
mask_h_bg_ch004_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_6", describe_end="mask_h_bg_block1_ch004_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_6-flow_unet2-mask_h_bg_block1_ch004_sig_bce_ep200-20211107_174309")  #.change_result_name_v1_to_v2()
mask_h_bg_ch002_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_7", describe_end="mask_h_bg_block1_ch002_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_7-flow_unet2-mask_h_bg_block1_ch002_sig_bce_ep200-20211107_181024")  #.change_result_name_v1_to_v2()
mask_h_bg_ch001_sig_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="1b_8", describe_end="mask_h_bg_block1_ch001_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-1b_8-flow_unet2-mask_h_bg_block1_ch001_sig_bce_ep200-20211107_162446")  #.change_result_name_v1_to_v2()

### 2. level
mask_h_bg_ch032_L2_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L2, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_1", describe_end="mask_h_bg_block1_ch032_2l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_1-flow_unet2-mask_h_bg_block1_ch032_2l_sig_bce_ep060-20211106_181334")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L3_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L3, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_2", describe_end="mask_h_bg_block1_ch032_3l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_2-flow_unet2-mask_h_bg_block1_ch032_3l_sig_bce_ep060-20211106_184439")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L4_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L4, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_3", describe_end="mask_h_bg_block1_ch032_4l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_3-flow_unet2-mask_h_bg_block1_ch032_4l_sig_bce_ep060-20211106_191714")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L5_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L5, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_4", describe_end="mask_h_bg_block1_ch032_5l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_4-flow_unet2-mask_h_bg_block1_ch032_5l_sig_bce_ep060-20211106_195106")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L6_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L6, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_5", describe_end="mask_h_bg_block1_ch032_6l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_5-flow_unet2-mask_h_bg_block1_ch032_6l_sig_bce_ep060-20211106_203118")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_6", describe_end="mask_h_bg_block1_ch032_7l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_6-flow_unet2-mask_h_bg_block1_ch032_7l_sig_bce_ep060-20211106_212204")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L8_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_L8, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_7", describe_end="mask_h_bg_block1_ch032_8l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-2_7-flow_unet2-mask_h_bg_block1_ch032_8l_sig_bce_ep060-20211106_222337")  #.change_result_name_v1_to_v2()

### 3. no-concat
mask_h_bg_ch032_L7_2to2noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to2noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_1", describe_end="mask_h_bg_block1_ch032_7l_2to2noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_1-flow_unet2-mask_h_bg_block1_ch032_7l_2to2noC_sig_bce_ep060-20211106_233532")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to3noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to3noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_2", describe_end="mask_h_bg_block1_ch032_7l_2to3noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_2-flow_unet2-mask_h_bg_block1_ch032_7l_2to3noC_sig_bce_ep060-20211107_003609")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to4noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to4noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_3", describe_end="mask_h_bg_block1_ch032_7l_2to4noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_3-flow_unet2-mask_h_bg_block1_ch032_7l_2to4noC_sig_bce_ep060-20211107_013553")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to5noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to5noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_4", describe_end="mask_h_bg_block1_ch032_7l_2to5noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_4-flow_unet2-mask_h_bg_block1_ch032_7l_2to5noC_sig_bce_ep060-20211107_023434")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to6noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to6noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_5", describe_end="mask_h_bg_block1_ch032_7l_2to6noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_5-flow_unet2-mask_h_bg_block1_ch032_7l_2to6noC_sig_bce_ep060-20211107_033153")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to7noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to7noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_6", describe_end="mask_h_bg_block1_ch032_7l_2to7noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_6-flow_unet2-mask_h_bg_block1_ch032_7l_2to7noC_sig_bce_ep060-20211107_042540")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to8noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_L7_ch32_2to8noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="3_7", describe_end="mask_h_bg_block1_ch032_7l_2to8noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-3_7-flow_unet2-mask_h_bg_block1_ch032_7l_2to8noC_sig_bce_ep060-20211107_051839")  #.change_result_name_v1_to_v2()

### 4. skip use add
mask_h_bg_ch032_L2_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_2_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_1", describe_end="mask_h_bg_block1_ch032_2l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_1-flow_unet2-mask_h_bg_block1_ch032_2l_skipAdd_sig_bce_ep060-20211107_061219")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L3_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_3_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_2", describe_end="mask_h_bg_block1_ch032_3l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_2-flow_unet2-mask_h_bg_block1_ch032_3l_skipAdd_sig_bce_ep060-20211107_065021")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L4_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_4_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_3", describe_end="mask_h_bg_block1_ch032_4l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_3-flow_unet2-mask_h_bg_block1_ch032_4l_skipAdd_sig_bce_ep060-20211107_074312")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L5_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_5_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_4", describe_end="mask_h_bg_block1_ch032_5l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_4-flow_unet2-mask_h_bg_block1_ch032_5l_skipAdd_sig_bce_ep060-20211107_060347")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L6_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_6_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_5", describe_end="mask_h_bg_block1_ch032_6l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_5-flow_unet2-mask_h_bg_block1_ch032_6l_skipAdd_sig_bce_ep060-20211107_072210")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L7_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_7_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_6", describe_end="mask_h_bg_block1_ch032_7l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_6-flow_unet2-mask_h_bg_block1_ch032_7l_skipAdd_sig_bce_ep060-20211107_085130")  #.change_result_name_v1_to_v2()
mask_h_bg_ch032_L8_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_8_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="4_7", describe_end="mask_h_bg_block1_ch032_8l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-4_7-flow_unet2-mask_h_bg_block1_ch032_8l_skipAdd_sig_bce_ep060-20211107_103147")  #.change_result_name_v1_to_v2()

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch128_sig_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
