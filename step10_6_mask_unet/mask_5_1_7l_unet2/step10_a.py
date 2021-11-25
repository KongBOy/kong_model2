#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print("step10a")
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
exp_dir = code_exe_path_element[5][7:] + "/" + code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
print("    exp_dir:", exp_dir)  ### 舉例：exp_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_e2_mask_unet2_obj import *
from step09_b_loss_info_obj import *
from step10_b_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet2/自己命的名字"，那 result_dir 就都在：
    6_mask_unet2/自己命的名字/result_a
    6_mask_unet2/自己命的名字/result_b
    6_mask_unet2/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
############################  have_bg  #################################
### 1a. ch
mask_h_bg_ch128_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_ch128_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="mask_h_bg_ch128_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_1-flow_unet2-mask_h_bg_ch128_sig_bce_ep060-20211106_213808")  #.result_name_v1_to_v2()
mask_h_bg_ch064_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_ch064_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_2", describe_end="mask_h_bg_ch064_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_2-flow_unet2-mask_h_bg_ch064_sig_bce_ep060-20211106_175325")  #.result_name_v1_to_v2()
mask_h_bg_ch032_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_3-flow_unet2-mask_h_bg_ch032_sig_bce_ep060-20211106_183823")  #.result_name_v1_to_v2()
mask_h_bg_ch016_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch016_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_4", describe_end="mask_h_bg_ch016_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_4-flow_unet2-mask_h_bg_ch016_sig_bce_ep060-20211106_191129")  #.result_name_v1_to_v2()
mask_h_bg_ch008_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch008_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_5", describe_end="mask_h_bg_ch008_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_5-flow_unet2-mask_h_bg_ch008_sig_bce_ep060-20211106_194156")  #.result_name_v1_to_v2()
mask_h_bg_ch004_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch004_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_6", describe_end="mask_h_bg_ch004_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_6-flow_unet2-mask_h_bg_ch004_sig_bce_ep060-20211106_201103")  #.result_name_v1_to_v2()
mask_h_bg_ch002_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch002_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_7", describe_end="mask_h_bg_ch002_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_7-flow_unet2-mask_h_bg_ch002_sig_bce_ep060-20211106_204003")  #.result_name_v1_to_v2()
mask_h_bg_ch001_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch001_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_8", describe_end="mask_h_bg_ch001_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_8-flow_unet2-mask_h_bg_ch001_sig_bce_ep060-20211106_210910")  #.result_name_v1_to_v2()
### 1b. ch and epoch
mask_h_bg_ch128_sig_ep200 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_ch128_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_1", describe_end="mask_h_bg_ch128_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_1-flow_unet2-mask_h_bg_ch128_sig_bce_ep200-20211107_112121")  #.result_name_v1_to_v2()
mask_h_bg_ch064_sig_ep200 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_ch064_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_2", describe_end="mask_h_bg_ch064_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_2-flow_unet2-mask_h_bg_ch064_sig_bce_ep200-20211106_225648")  #.result_name_v1_to_v2()
mask_h_bg_ch032_sig_ep200 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_3", describe_end="mask_h_bg_ch032_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_3-flow_unet2-mask_h_bg_ch032_sig_bce_ep200-20211107_012509")  #.result_name_v1_to_v2()
mask_h_bg_ch016_sig_ep200 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_ch016_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_4", describe_end="mask_h_bg_ch016_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_4-flow_unet2-mask_h_bg_ch016_sig_bce_ep200-20211107_031433")  #.result_name_v1_to_v2()
mask_h_bg_ch008_sig_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_ch008_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_5", describe_end="mask_h_bg_ch008_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_5-flow_unet2-mask_h_bg_ch008_sig_bce_ep200-20211107_045522")  #.result_name_v1_to_v2()
mask_h_bg_ch004_sig_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_ch004_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_6", describe_end="mask_h_bg_ch004_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_6-flow_unet2-mask_h_bg_ch004_sig_bce_ep200-20211107_063215")  #.result_name_v1_to_v2()
mask_h_bg_ch002_sig_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_ch002_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_7", describe_end="mask_h_bg_ch002_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_7-flow_unet2-mask_h_bg_ch002_sig_bce_ep200-20211107_080820")  #.result_name_v1_to_v2()
mask_h_bg_ch001_sig_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_ch001_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_8", describe_end="mask_h_bg_ch001_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_8-flow_unet2-mask_h_bg_ch001_sig_bce_ep200-20211107_094452")  #.result_name_v1_to_v2()

### 2. level
mask_h_bg_ch032_L2_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L2_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_1", describe_end="mask_h_bg_ch032_L2_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_1-flow_unet2-mask_h_bg_ch032_L2_sig_bce_ep060-20211107_154433")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L3_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L3_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_2", describe_end="mask_h_bg_ch032_L3_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_2-flow_unet2-mask_h_bg_ch032_L3_sig_bce_ep060-20211107_161322")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L4_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L4_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_3", describe_end="mask_h_bg_ch032_L4_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_3-flow_unet2-mask_h_bg_ch032_L4_sig_bce_ep060-20211107_164235")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L5_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L5_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_4", describe_end="mask_h_bg_ch032_L5_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_4-flow_unet2-mask_h_bg_ch032_L5_sig_bce_ep060-20211107_171204")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L6_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L6_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_5", describe_end="mask_h_bg_ch032_L6_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_5-flow_unet2-mask_h_bg_ch032_L6_sig_bce_ep060-20211107_174204")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L7_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_6", describe_end="mask_h_bg_ch032_L7_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_6-flow_unet2-mask_h_bg_ch032_L7_sig_bce_ep060-20211107_181318")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L8_sig_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet2_L8_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_7", describe_end="mask_h_bg_ch032_L8_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_2_7-flow_unet2-mask_h_bg_ch032_L8_sig_bce_ep060-20211107_184619")  #.result_name_v1_to_v2()

### 3. no-concat
mask_h_bg_ch032_L7_2to2noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to2noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_1", describe_end="mask_h_bg_ch032_7l_2to2noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_1-flow_unet2-mask_h_bg_ch032_7l_2to2noC_sig_bce_ep060-20211107_192105")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to3noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to3noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_2", describe_end="mask_h_bg_ch032_7l_2to3noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_2-flow_unet2-mask_h_bg_ch032_7l_2to3noC_sig_bce_ep060-20211107_195409")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to4noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to4noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_3", describe_end="mask_h_bg_ch032_7l_2to4noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_3-flow_unet2-mask_h_bg_ch032_7l_2to4noC_sig_bce_ep060-20211107_202709")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to5noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to5noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_4", describe_end="mask_h_bg_ch032_7l_2to5noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_4-flow_unet2-mask_h_bg_ch032_7l_2to5noC_sig_bce_ep060-20211107_210005")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to6noC_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to6noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_5", describe_end="mask_h_bg_ch032_7l_2to6noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_5-flow_unet2-mask_h_bg_ch032_7l_2to6noC_sig_bce_ep060-20211107_213259")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to7noC_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to7noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_6", describe_end="mask_h_bg_ch032_7l_2to7noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_6-flow_unet2-mask_h_bg_ch032_7l_2to7noC_sig_bce_ep060-20211107_220449")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_2to8noC_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_IN_L7_ch32_2to8noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_7", describe_end="mask_h_bg_ch032_7l_2to8noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_7-flow_unet2-mask_h_bg_ch032_7l_2to8noC_sig_bce_ep060-20211107_223644")  #.result_name_v1_to_v2()

### 4. skip use add
mask_h_bg_ch032_L2_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L2_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_1", describe_end="mask_h_bg_ch032_2l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_1-flow_unet2-mask_h_bg_ch032_2l_skipAdd_sig_bce_ep060-20211107_230850")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L3_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L3_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_2", describe_end="mask_h_bg_ch032_3l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_2-flow_unet2-mask_h_bg_ch032_3l_skipAdd_sig_bce_ep060-20211107_233828")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L4_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L4_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_3", describe_end="mask_h_bg_ch032_4l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_3-flow_unet2-mask_h_bg_ch032_4l_skipAdd_sig_bce_ep060-20211108_000916")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L5_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L5_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_4", describe_end="mask_h_bg_ch032_5l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_4-flow_unet2-mask_h_bg_ch032_5l_skipAdd_sig_bce_ep060-20211108_004127")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L6_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L6_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_5", describe_end="mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_5-flow_unet2-mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060-20211108_011532")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L7_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L7_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_6", describe_end="mask_h_bg_ch032_7l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_6-flow_unet2-mask_h_bg_ch032_7l_skipAdd_sig_bce_ep060-20211108_015402")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L8_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet2_L8_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_7", describe_end="mask_h_bg_ch032_8l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_7-flow_unet2-mask_h_bg_ch032_8l_skipAdd_sig_bce_ep060-20211108_023857")  #.result_name_v1_to_v2()

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
