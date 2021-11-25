#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
print("step10a")
print("    code_exe_path:", code_exe_path)
print("    code_exe_path_element:", code_exe_path_element)
print("    kong_layer:", kong_layer)
print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
exp_dir = code_exe_path_element[5][7:] + "/" + code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
print("    exp_dir:", exp_dir)  ### 舉例：exp_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
from step06_a_datas_obj import *
from step09_e3_flow_unet2_obj import *
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
L2_ch128_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch128", describe_end="block1_L2_ch128_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch128-flow_unet2-block1_L2_ch128_mae_s001-20211118_210714")
L2_ch064_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch064", describe_end="block1_L2_ch064_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch064-flow_unet2-block1_L2_ch064_mae_s001-20211118_223136")
L2_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch032", describe_end="block1_L2_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch032-flow_unet2-block1_L2_ch032_mae_s001-20211118_230931")
L2_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch016", describe_end="block1_L2_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch016-flow_unet2-block1_L2_ch016_mae_s001-20211118_234035")
L2_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch008", describe_end="block1_L2_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch008-flow_unet2-block1_L2_ch008_mae_s001-20211119_001010")
L2_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch004", describe_end="block1_L2_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch004-flow_unet2-block1_L2_ch004_mae_s001-20211119_003910")
L2_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch002", describe_end="block1_L2_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch002-flow_unet2-block1_L2_ch002_mae_s001-20211119_010752")
L2_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L2, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L2_ch001", describe_end="block1_L2_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L2_ch001-flow_unet2-block1_L2_ch001_mae_s001-20211119_013628")
#################################################################################################################################################################################################################################################################################################################################################################################################
L3_ch128_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch128", describe_end="block1_L3_ch128_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch128-flow_unet2-block1_L3_ch128_mae_s001-20211118_210741")
L3_ch064_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch064", describe_end="block1_L3_ch064_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch064-flow_unet2-block1_L3_ch064_mae_s001-20211118_232048")
L3_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch032", describe_end="block1_L3_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch032-flow_unet2-block1_L3_ch032_mae_s001-20211119_001403")
L3_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch016", describe_end="block1_L3_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch016-flow_unet2-block1_L3_ch016_mae_s001-20211119_004652")
L3_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch008", describe_end="block1_L3_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch008-flow_unet2-block1_L3_ch008_mae_s001-20211119_011719")
L3_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch004", describe_end="block1_L3_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch004-flow_unet2-block1_L3_ch004_mae_s001-20211119_014659")
L3_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch002", describe_end="block1_L3_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch002-flow_unet2-block1_L3_ch002_mae_s001-20211119_021606")
L3_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L3, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L3_ch001", describe_end="block1_L3_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L3_ch001-flow_unet2-block1_L3_ch001_mae_s001-20211119_024513")
#################################################################################################################################################################################################################################################################################################################################################################################################
L4_ch064_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch064", describe_end="block1_L4_ch064_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch064-flow_unet2-block1_L4_ch064_mae_s001-20211118_210751")
L4_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch032", describe_end="block1_L4_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch032-flow_unet2-block1_L4_ch032_mae_s001-20211118_221705")
L4_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch016", describe_end="block1_L4_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch016-flow_unet2-block1_L4_ch016_mae_s001-20211118_225104")
L4_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch008", describe_end="block1_L4_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch008-flow_unet2-block1_L4_ch008_mae_s001-20211118_232142")
L4_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch004", describe_end="block1_L4_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch004-flow_unet2-block1_L4_ch004_mae_s001-20211118_235120")
L4_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch002", describe_end="block1_L4_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch002-flow_unet2-block1_L4_ch002_mae_s001-20211119_002030")
L4_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L4, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L4_ch001", describe_end="block1_L4_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L4_ch001-flow_unet2-block1_L4_ch001_mae_s001-20211119_004928")
#################################################################################################################################################################################################################################################################################################################################################################################################
L5_ch032_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch032", describe_end="block1_L5_ch032_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch032-flow_unet2-block1_L5_ch032_mae_s001-20211118_210803")
L5_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch016", describe_end="block1_L5_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch016-flow_unet2-block1_L5_ch016_mae_s001-20211118_215841")
L5_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch008", describe_end="block1_L5_ch008_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch008-flow_unet2-block1_L5_ch008_mae_s001-20211118_223222")
L5_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch004", describe_end="block1_L5_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch004-flow_unet2-block1_L5_ch004_mae_s001-20211118_230409")
L5_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch002", describe_end="block1_L5_ch002_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch002-flow_unet2-block1_L5_ch002_mae_s001-20211118_233506")
L5_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L5, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L5_ch001", describe_end="block1_L5_ch001_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L5_ch001-flow_unet2-block1_L5_ch001_mae_s001-20211119_000553")
#################################################################################################################################################################################################################################################################################################################################################################################################
L6_ch016_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch016", describe_end="block1_L6_ch016_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch016-flow_unet2-block1_L6_ch016_mae_s001-20211118_213447")
L6_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch008", describe_end="block1_ch008_mae_s001_L6") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch008-flow_unet2-block1_L6_ch008_mae_s001-20211118_220855")
L6_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch004", describe_end="block1_ch004_mae_s001_L6") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch004-flow_unet2-block1_L6_ch004_mae_s001-20211118_224114")
L6_ch002_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch002_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch002", describe_end="block1_ch002_mae_s001_L6") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch002-flow_unet2-block1_L6_ch002_mae_s001-20211118_231235")
L6_ch001_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch001_sig_L6, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L6_ch001", describe_end="block1_ch001_mae_s001_L6") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L6_ch001-flow_unet2-block1_L6_ch001_mae_s001-20211118_234351")
#################################################################################################################################################################################################################################################################################################################################################################################################
L7_ch008_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L7, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L7_ch008", describe_end="block1_ch008_mae_s001_L7") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L7_ch008-flow_unet2-block1_L7_ch008_mae_s001-20211118_213539")
#################################################################################################################################################################################################################################################################################################################################################################################################
L8_ch004_mae_s001 = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L8, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch004", describe_end="block1_L8_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch004-flow_unet2-block1_L8_ch004_mae_s001-20211118_221004")
L8_ch004_mae_s001_copy = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch004_sig_L8, G_mae_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="2_L8_ch004", describe_end="block1_L8_ch004_mae_s001") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-2_L8_ch004-flow_unet2-block1_L8_ch004_mae_s001-20211118_221004_copy")


if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L2_ch001_mae_s001.build().run()
        # L3_ch128_mae_s001_copy.build().run()

        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
