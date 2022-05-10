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
from step09_e7_flow_unet2_obj_M_w_Mgt_to_Cx_Cy import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "7_block1/自己命的名字"，那 result_dir 就都在：
    7_block1/自己命的名字/result_a
    7_block1/自己命的名字/result_b
    7_block1/自己命的名字/...
'''

use_db_obj = type8_blender_wc_flow
use_loss_obj = [G_mae_s001_loss_info_builder.set_loss_target("UNet_Cx"), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cy")]  ### x, y 順序是看 step07_b_0b_Multi_UNet 來對應的喔
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################
L2_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch128_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch128_block1-20220108_025933")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch064_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch064_block1-20220107_231759")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch032_block1-20220108_001634")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch016_block1-20220108_005204")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch008_block1-20220108_012435")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch004_block1-20220108_015646")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch002_block1-20220108_022818")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L2_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L2_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L2_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L2_ch001_block1-20220108_051114")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
##########################################################################################################################################################################################################################################################################################################################################################
L3_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch128_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch128_block1-20220108_095945")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch064_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch064_block1-20220108_054218")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch032_block1-20220108_070751")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch016_block1-20220108_074926")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch008_block1-20220108_082352")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch004_block1-20220108_085626")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch002_block1-20220108_092810")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L3_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L3_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L3_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L3_ch001_block1-20220108_133739")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
##########################################################################################################################################################################################################################################################################################################################################################
L4_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch128_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch128_block1-20220108_032854")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch064_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch064_block1-20220107_232055")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch032_block1-20220108_004548")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch016_block1-20220108_012459")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch008_block1-20220108_015724")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch004_block1-20220108_022823")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch002_block1-20220108_025837")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L4_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L4_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch001_block1-20220108_072717")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
##########################################################################################################################################################################################################################################################################################################################################################
L5_ch128 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch128_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch128_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch128_block1-20220108_043644")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch064_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch064_block1-20220107_232221")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch032_block1-20220108_013421")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch016_block1-20220108_022910")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch008_block1-20220108_030245")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch004_block1-20220108_033436")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch002_block1-20220108_040555")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L5_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch001_block1-20220108_121306")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
##########################################################################################################################################################################################################################################################################################################################################################
L6_ch064 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch064_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch064_block1-20220108_025941")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch032_block1-20220107_232359")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch016_block1-20220108_004834")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch008_block1-20220108_012420")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch004_block1-20220108_015702")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch002_block1-20220108_022831")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L6_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch001_block1-20220108_080347")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
##########################################################################################################################################################################################################################################################################################################################################################
L7_ch032 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch032_block1-20220108_022130")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch016_block1-20220107_232559")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch008_block1-20220108_003139")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch004_block1-20220108_011007")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch002_block1-20220108_014612")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L7_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch001_block1-20220108_064919")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
##########################################################################################################################################################################################################################################################################################################################################################
L8_ch016 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch016_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch016_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch016_block1-20220108_103546")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch008 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch008_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch008_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch008_block1-20220108_075746")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch004 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch004_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch004_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch004_block1-20220108_085752")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch002 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch002_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch002_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch002_block1-20220108_093130")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch001 = Exp_builder().set_basic("train", use_db_obj, block1_L8_ch001_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch001_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch001_block1-20220108_100346")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################
L6_ch064 = Exp_builder().set_basic("test_see", use_db_obj, block1_L6_ch064_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch064_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch064_block1-20220108_025941")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch032 = Exp_builder().set_basic("test_see", use_db_obj, block1_L6_ch032_sig, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch032_sig.kong_model.model_describe).set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch032_block1-20220107_232359")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L6_ch064.build().run()
        L6_ch032.build().run()
        # L2_ch002.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
