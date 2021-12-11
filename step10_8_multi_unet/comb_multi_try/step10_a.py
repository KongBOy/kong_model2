#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
exp_dir = code_exe_path_element[-3][7:] + "/" + code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
# print("    exp_dir:", exp_dir)  ### 舉例：exp_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_f1_multi_unet2_obj_I_to_M_w_I_to_C import *
from step09_b_loss_info_obj import *
from step10_b_exp_builder import Exp_builder

import step10_6_mask_unet.mask_5_2_bce_block1_45678l.step10_a as I_to_M_exp_builder
import step10_7_flow_unet.mask_5_2_mae_block1_45678l_I_with_Mgt_to_C.step10_a as M_w_I_to_C_exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
############################  have_bg  #################################
### 1a. ch
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch032_ep060 = Exp_builder().set_basic("train", use_db_obj, I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch032, [G_bce_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="I_to_M_w_I_to_C") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_M=I_to_M_exp_builder.L4_ch032_sig_ep060_bce_s001, M_w_I_to_C=M_w_I_to_C_exp_builder.L5_ch032_mae_s001).set_result_name(result_name="")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch032_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
