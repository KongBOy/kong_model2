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
###############################################################################################################################################################################################################
# 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
    os.chdir(code_exe_dir)
# print("current_path:", os.getcwd())
###############################################################################################################################################################################################################
### 所有 指令 統一寫這邊
from step10_c_exp_command import *
######################################################################################################################
import subprocess as sb

### I_w_M_to_C woDiv 不需要看 LD/SSIM， 因為沒有rec
### W_w_M_to_C woDiv
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C_ch016_L5_normal       .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C_ch016_L5_noFocus_ChBg .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])

##### 一起訓練
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_normal__I_w_M_to_W_normal__W_w_M_to_C_normal                .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_ChBg__I_w_M_to_W_ChBg__W_w_M_to_C_normal                    .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_noFocus__I_w_M_to_W_noFocus_W_w_M_to_C_noFocus_ChBg         .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_noFocus_ChBg__I_w_M_to_W_noFocus_W_w_M_to_C_noFocus_ChBg__1 .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_noFocus_ChBg__I_w_M_to_W_noFocus_W_w_M_to_C_noFocus_ChBg__2 .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_noFocus_ChBg__I_w_M_to_W_noFocus_W_w_M_to_C_noFocus_ChBg__3 .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_noFocus_ChBg__I_w_M_to_W_noFocus_W_w_M_to_C_noFocus_ChBg__4 .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"exp_L5_gather_noFocus_ChBg__I_w_M_to_W_noFocus_W_w_M_to_C_noFocus_ChBg__5 .{test % ('test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA', 'knpy_save=False')}"])
