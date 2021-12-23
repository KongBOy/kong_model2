#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer
# print("    kong_to_py_layer:", kong_to_py_layer)
if  (kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][7:]  ### [7:] 是為了去掉 step1x_
elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
# print("    template_dir:", template_dir)  ### 舉例： template_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
exp_dir = template_dir
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
use_loss_obj = mae_s001_sobel_k5_s001_tv_s001_loss_info_builder
################################################################################################################################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################
L4_ch128_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L4_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch128_sig_L4_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
################################################################################################################################################################################################################################################################################################################################################################################################
L5_ch128_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L5_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch128_sig_L5_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L5_ch064_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L5_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch064_sig_L5_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
################################################################################################################################################################################################################################################################################################################################################################################################
L6_ch128_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L6_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch128_sig_L6_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L6_ch064_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L6_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch064_sig_L6_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L6_ch032_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L6_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch032_sig_L6_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
################################################################################################################################################################################################################################################################################################################################################################################################
L7_ch128_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L7_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch128_sig_L7_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L7_ch064_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L7_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch064_sig_L7_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L7_ch032_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L7_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch032_sig_L7_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L7_ch016_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L7_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch016_sig_L7_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
################################################################################################################################################################################################################################################################################################################################################################################################
L8_ch128_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch128_sig_L8_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch128_sig_L8_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch064_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch064_sig_L8_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch064_sig_L8_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch032_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch032_sig_L8_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch032_sig_L8_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch016_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch016_sig_L8_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch016_sig_L8_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
L8_ch008_limit = Exp_builder().set_basic("train", use_db_obj, flow_unet2_block1_ch008_sig_L8_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=flow_unet2_block1_ch008_sig_L8_limit.kong_model.model_describe + "_limit") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
################################################################################################################################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        # L8_ch008_mae_s001_limit.build().run()
        # L2_ch001_mae_s001.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
