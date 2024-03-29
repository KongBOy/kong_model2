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
from step09_e5_flow_unet2_obj_I_to_M import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder

rm_paths = [path for path in sys.path if code_dir in path]
for rm_path in rm_paths: sys.path.remove(rm_path)
rm_moduless = [module for module in sys.modules if "step09" in module]
for rm_module in rm_moduless: del sys.modules[rm_module]
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "7_block1/自己命的名字"，那 result_dir 就都在：
    7_block1/自己命的名字/result_a
    7_block1/自己命的名字/result_b
    7_block1/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
use_loss_obj = G_sobel_k5_loss_info_builder
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################
L4_ch128_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L4_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L4_ch128_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L4_ch128_block1_sig_out_1_limit-20211220_170305")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
#######################################################################################################################################################################################################################################################################################################################################################################################
L5_ch128_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L5_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch128_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch128_block1_sig_out_1_limit-20211220_203635")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L5_ch064_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L5_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L5_ch064_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L5_ch064_block1_sig_out_1_limit-20211221_003209")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
#######################################################################################################################################################################################################################################################################################################################################################################################
L6_ch128_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L6_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch128_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch128_block1_sig_out_1_limit-20211221_022008")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch064_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L6_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch064_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch064_block1_sig_out_1_limit-20211221_063144")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L6_ch032_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L6_ch032_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L6_ch032_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L6_ch032_block1_sig_out_1_limit-20211221_083532")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
#######################################################################################################################################################################################################################################################################################################################################################################################
L7_ch128_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L7_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch128_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch128_block1_sig_out_1_limit-20211220_170212")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch064_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L7_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch064_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch064_block1_sig_out_1_limit-20211220_202408")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch032_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L7_ch032_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch032_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch032_block1_sig_out_1_limit-20211220_204151")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L7_ch016_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L7_ch016_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L7_ch016_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L7_ch016_block1_sig_out_1_limit-20211220_214240")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
#######################################################################################################################################################################################################################################################################################################################################################################################
L8_ch128_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L8_ch128_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch128_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch128_block1_sig_out_1_limit-20211220_221959")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch064_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L8_ch064_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch064_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch064_block1_sig_out_1_limit-20211221_015049")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch032_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L8_ch032_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch032_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch032_block1_sig_out_1_limit-20211221_034721")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch016_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L8_ch016_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch016_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch016_block1_sig_out_1_limit-20211221_045904")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
L8_ch008_limit = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, block1_L8_ch008_sig_limit, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=block1_L8_ch008_sig_limit.kong_model.model_describe + "_limit").set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="L8_ch008_block1_sig_out_1_limit-20211221_054728")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################################################################################################################################################################################################################################################################################################################################################################################

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        L6_ch064_limit.build().run()
        # L2_ch001.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
