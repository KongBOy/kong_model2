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
# print("    exp_dir:", exp_dir)  ### 舉例：exp_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
from step06_a_datas_obj import *
from step09_e5_flow_unet2_obj_I_with_Mgt_to_C import *
from step09_b_loss_info_obj import *
from step10_b_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "7_flow_unet2_block1/自己命的名字"，那 result_dir 就都在：
    7_flow_unet2_block1/自己命的名字/result_a
    7_flow_unet2_block1/自己命的名字/result_b
    7_flow_unet2_block1/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
#################################################################################################################################################################################################################################################################################################################################################################################################
L2_ch128_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch128", describe_end="block1_L2_ch128_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch064_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch064", describe_end="block1_L2_ch064_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch032", describe_end="block1_L2_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch016", describe_end="block1_L2_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch008", describe_end="block1_L2_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch004", describe_end="block1_L2_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch002", describe_end="block1_L2_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L2_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch001", describe_end="block1_L2_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
#################################################################################################################################################################################################################################################################################################################################################################################################
L3_ch128_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch128", describe_end="block1_L3_ch128_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch128-flow_unet2-block1_L3_ch128_mae_s001-20211125_180625")
L3_ch064_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch064", describe_end="block1_L3_ch064_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch064-flow_unet2-block1_L3_ch064_mae_s001-20211125_201641")
L3_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch032", describe_end="block1_L3_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch032-flow_unet2-block1_L3_ch032_mae_s001-20211125_210855")
L3_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch016", describe_end="block1_L3_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch016-flow_unet2-block1_L3_ch016_mae_s001-20211125_214042")
L3_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch008", describe_end="block1_L3_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch008-flow_unet2-block1_L3_ch008_mae_s001-20211125_221014")
L3_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch004", describe_end="block1_L3_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch004-flow_unet2-block1_L3_ch004_mae_s001-20211125_223858")
L3_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch002", describe_end="block1_L3_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch002-flow_unet2-block1_L3_ch002_mae_s001-20211125_230720")
L3_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch001", describe_end="block1_L3_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch001-flow_unet2-block1_L3_ch001_mae_s001-20211125_233533")
#################################################################################################################################################################################################################################################################################################################################################################################################
L4_ch064_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch064", describe_end="block1_L4_ch064_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch064-flow_unet2-block1_L4_ch064_mae_s001-20211125_170234")
L4_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch032", describe_end="block1_L4_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch032-flow_unet2-block1_L4_ch032_mae_s001-20211125_181054")
L4_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch016", describe_end="block1_L4_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch016-flow_unet2-block1_L4_ch016_mae_s001-20211125_184428")
L4_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch008", describe_end="block1_L4_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch008-flow_unet2-block1_L4_ch008_mae_s001-20211125_191455")
L4_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch004", describe_end="block1_L4_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch004-flow_unet2-block1_L4_ch004_mae_s001-20211125_194422")
L4_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch002", describe_end="block1_L4_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch002-flow_unet2-block1_L4_ch002_mae_s001-20211125_201310")
L4_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch001", describe_end="block1_L4_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch001-flow_unet2-block1_L4_ch001_mae_s001-20211125_204154")
#################################################################################################################################################################################################################################################################################################################################################################################################
L5_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch032", describe_end="block1_L5_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch032-flow_unet2-block1_L5_ch032_mae_s001-20211125_170346")
L5_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch016", describe_end="block1_L5_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch016-flow_unet2-block1_L5_ch016_mae_s001-20211125_174349")
L5_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch008", describe_end="block1_L5_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch008-flow_unet2-block1_L5_ch008_mae_s001-20211125_181438")
L5_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch004", describe_end="block1_L5_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch004-flow_unet2-block1_L5_ch004_mae_s001-20211125_184406")
L5_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch002", describe_end="block1_L5_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch002-flow_unet2-block1_L5_ch002_mae_s001-20211125_191303")
L5_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch001", describe_end="block1_L5_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch001-flow_unet2-block1_L5_ch001_mae_s001-20211125_194155")
#################################################################################################################################################################################################################################################################################################################################################################################################
L6_ch128_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L6      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch128",       describe_end="block1_L6_ch128_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L6_ch128_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L6_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch128_limit", describe_end="block1_L6_ch128_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L6_ch064_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L6      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch064",       describe_end="block1_L6_ch064_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L6_ch064_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L6_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch064_limit", describe_end="block1_L6_ch064_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L6_ch032_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L6      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch032",       describe_end="block1_L6_ch032_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch032-flow_unet2-block1_L6_ch032_mae_s001-20211125_213807")
L6_ch032_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L6_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch032_limit", describe_end="block1_L6_ch032_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch032_limit-flow_unet2-block1_L6_ch032_limit_mae_s001-20211125_222742")
L6_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch016", describe_end="block1_L6_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch016-flow_unet2-block1_L6_ch016_mae_s001-20211125_231726")
L6_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch008", describe_end="block1_L6_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch008-flow_unet2-block1_L6_ch008_mae_s001-20211125_234937")
L6_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch004", describe_end="block1_L6_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch004-flow_unet2-block1_L6_ch004_mae_s001-20211126_001937")
L6_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch002", describe_end="block1_L6_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch002-flow_unet2-block1_L6_ch002_mae_s001-20211126_004907")
L6_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch001", describe_end="block1_L6_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch001-flow_unet2-block1_L6_ch001_mae_s001-20211126_011819")
#################################################################################################################################################################################################################################################################################################################################################################################################
L7_ch128_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L7      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch128"      , describe_end="block1_L7_ch128_mae_s001"      ) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L7_ch128_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L7_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch128_limit", describe_end="block1_L7_ch128_mae_s001_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L7_ch064_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L7      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch064"      , describe_end="block1_L7_ch064_mae_s001"      ) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L7_ch064_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L7_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch064_limit", describe_end="block1_L7_ch064_mae_s001_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L7_ch032_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L7      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch032"      , describe_end="block1_L7_ch032_mae_s001"      ) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch032-flow_unet2-block1_L7_ch032_mae_s001-20211125_213833")
L7_ch032_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L7_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch032_limit", describe_end="block1_L7_ch032_mae_s001_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch032_limit-flow_unet2-block1_L7_ch032_mae_s001_limit-20211125_223941")
L7_ch016_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L7      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch016"      , describe_end="block1_L7_ch016_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch016-flow_unet2-block1_L7_ch016_mae_s001-20211125_234055")
L7_ch016_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L7_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch016_limit", describe_end="block1_L7_ch016_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch016_limit-flow_unet2-block1_L7_ch016_mae_s001-20211126_001840")
L7_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L7, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch008", describe_end="block1_L7_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch008-flow_unet2-block1_L7_ch008_mae_s001-20211126_005620")
L7_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L7, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch004", describe_end="block1_L7_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch004-flow_unet2-block1_L7_ch004_mae_s001-20211126_012721")
L7_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L7, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch002", describe_end="block1_L7_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch002-flow_unet2-block1_L7_ch002_mae_s001-20211126_015655")
L7_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L7, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch001", describe_end="block1_L7_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch001-flow_unet2-block1_L7_ch001_mae_s001-20211126_022606")
#################################################################################################################################################################################################################################################################################################################################################################################################
L8_ch128_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L8      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch128"      , describe_end="block1_L8_ch128_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L8_ch128_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L8_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch128_limit", describe_end="block1_L8_ch128_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L8_ch064_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L8      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch064"      , describe_end="block1_L8_ch064_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L8_ch064_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L8_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch064_limit", describe_end="block1_L8_ch064_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
L8_ch032_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L8      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch032"      , describe_end="block1_L8_ch032_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch032-flow_unet2-block1_L8_ch032_mae_s001-20211126_000345")
L8_ch032_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L8_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch032_limit", describe_end="block1_L8_ch032_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch032_limit-flow_unet2-block1_L8_ch032_limit_mae_s001-20211126_011445")
L8_ch016_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L8      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch016"      , describe_end="block1_L8_ch016_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch016-flow_unet2-block1_L8_ch016_mae_s001-20211126_022545")
L8_ch016_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L8_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch016_limit", describe_end="block1_L8_ch016_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch016_limit-flow_unet2-block1_L8_ch016_limit_mae_s001-20211126_031321")
L8_ch008_mae_s001       = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L8      , G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch008"      , describe_end="block1_L8_ch008_mae_s001")       .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch008-flow_unet2-block1_L8_ch008_mae_s001-20211126_040046")
L8_ch008_mae_s001_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L8_limit, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch008_limit", describe_end="block1_L8_ch008_limit_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch008_limit-flow_unet2-block1_L8_ch008_limit_mae_s001-20211126_043457")
L8_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L8, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch004", describe_end="block1_L8_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch004-flow_unet2-block1_L8_ch004_mae_s001-20211126_050852")
L8_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L8, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch002", describe_end="block1_L8_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch002-flow_unet2-block1_L8_ch002_mae_s001-20211126_053923")
L8_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L8, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch001", describe_end="block1_L8_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch001-flow_unet2-block1_L8_ch001_mae_s001-20211126_060902")


if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L2_ch016_mae_s001.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
